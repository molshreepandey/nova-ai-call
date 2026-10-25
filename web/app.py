"""Browser UI: a hands-free voice call with the same agent, served by FastAPI.

    python -m web.app          # then open http://127.0.0.1:8000

The page detects speech in the browser (voice activity detection), sends each
utterance here as WAV, and plays back the spoken reply. The server reuses the
CLI's components: Groq Whisper STT, CallSession + Orchestrator (local RAG tool +
MCP tools) and edge-tts. Each browser call gets its own session, so language
choice and conversation memory are per caller.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from agent.call import CallReply, CallSession
from agent.language import Language
from agent.llm_client import GroqLLMClient
from agent.mcp_client import MCPClient
from agent.orchestrator import Orchestrator
from config import settings, setup_logging
from rag.retriever import Retriever
from rag.tool import KnowledgeBaseTool
from voice.stt import GroqWhisperSTT, is_probable_hallucination
from voice.tts import EdgeTTS

logger = logging.getLogger("web")

STATIC_DIR = Path(__file__).parent / "static"
MAX_SESSIONS = 50
MAX_AUDIO_BYTES = 10 * 1024 * 1024


class Session:
    """One caller's call; the lock serialises that caller's turns."""

    def __init__(self, call: CallSession) -> None:
        self.call = call
        self.lock = asyncio.Lock()


class AgentService:
    """Process-wide components plus per-call sessions."""

    def __init__(self) -> None:
        self.llm = GroqLLMClient(
            api_key=settings.require_groq_key(),
            model=settings.llm_model,
            reasoning_effort=settings.reasoning_effort,
            fallback_model=settings.llm_fallback_model,
        )
        self.stt = GroqWhisperSTT(
            api_key=settings.require_groq_key(),
            model=settings.stt_model,
            models_by_language={"hi": settings.stt_model_hi},
        )
        self.tts = EdgeTTS()
        self.retriever = Retriever()
        self.kb_tool = KnowledgeBaseTool(self.retriever)
        self.mcp = MCPClient()
        self._sessions: OrderedDict[str, Session] = OrderedDict()

    def create(self) -> tuple[str, Session]:
        """Start a new call session, evicting the least recently used beyond the cap."""
        orchestrator = Orchestrator(
            self.llm,
            local_tools=[self.kb_tool],
            tool_server=self.mcp,
            max_iterations=settings.max_tool_iterations,
            max_history_turns=settings.max_history_turns,
            require_tool_first=settings.require_tool_first,
        )
        session_id = uuid.uuid4().hex
        self._sessions[session_id] = Session(CallSession(orchestrator))
        if len(self._sessions) > MAX_SESSIONS:
            self._sessions.popitem(last=False)
        return session_id, self._sessions[session_id]

    def get(self, session_id: str) -> Session:
        """Look up a live call."""
        session = self._sessions.get(session_id)
        if session is None:
            raise HTTPException(404, "call not found or already ended")
        self._sessions.move_to_end(session_id)
        return session

    def end(self, session_id: str) -> None:
        """Forget a call (hang up)."""
        self._sessions.pop(session_id, None)

    async def speech(self, text: str, language: Language | None) -> str | None:
        """MP3 for ``text`` as base64 in the call language's voice; None if TTS fails."""
        try:
            mp3 = await self.tts.synthesize_parallel(text, settings.tts_voice(language))
            return base64.b64encode(mp3).decode()
        except Exception as exc:  # noqa: BLE001 - TTS is best-effort; text still shows
            logger.warning("[TTS] synthesis failed: %s", exc)
            return None


service: AgentService | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Start the MCP server and warm the embedding model once for the whole app.

    The MCP session is entered and exited in this same task, as anyio requires.
    """
    global service
    setup_logging()
    service = AgentService()
    if service.retriever.count() == 0:
        raise RuntimeError("The knowledge base is empty. Run `python -m rag.ingest` first.")
    async with service.mcp:
        await asyncio.to_thread(service.retriever.warm_up)
        logger.info("[WEB] ready on http://127.0.0.1:8000")
        yield


app = FastAPI(title="NovaWealth Voice Agent", lifespan=lifespan)


def _service() -> AgentService:
    if service is None:
        raise HTTPException(503, "agent is starting")
    return service


class ReplyOut(BaseModel):
    """What the browser needs to show and play one agent reply."""

    session_id: str
    transcript: str = ""
    reply: str = ""
    language: Language | None = None
    audio: str | None = None
    tools: list[str] = []
    sources: list[str] = []
    ended: bool = False
    ignored: bool = False


async def _reply_out(svc: AgentService, session_id: str, transcript: str, reply: CallReply) -> ReplyOut:
    turn = reply.turn
    return ReplyOut(
        session_id=session_id,
        transcript=transcript,
        reply=reply.text,
        language=reply.language,
        audio=await svc.speech(reply.text, reply.language),
        tools=[c.name for c in turn.tool_calls] if turn else [],
        sources=turn.sources if turn else [],
        ended=reply.ended,
    )


@app.get("/")
async def index() -> FileResponse:
    """The single-page call UI."""
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/call")
async def start_call() -> ReplyOut:
    """Begin a call: new session plus the spoken opening (the language question)."""
    svc = _service()
    session_id, session = svc.create()
    opening = CallReply(session.call.opening(), session.call.language)
    return await _reply_out(svc, session_id, "", opening)


@app.post("/api/turn")
async def turn(session_id: str = Form(...), audio: UploadFile = File(...)) -> ReplyOut:
    """One caller utterance (WAV from the browser's voice detector): STT, agent, TTS.

    Noise that Whisper turns into nothing or a stock hallucination is reported as
    ``ignored`` so the browser silently keeps listening instead of re-prompting.
    """
    svc = _service()
    session = svc.get(session_id)
    data = await audio.read()
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(413, "recording too long")

    async with session.lock:
        call = session.call
        try:
            transcript = await svc.stt.transcribe(data, filename=audio.filename or "speech.wav", language=call.language)
        except Exception as exc:  # noqa: BLE001 - STT outage shouldn't end the call
            logger.warning("[STT] failed: %s", exc)
            return await _reply_out(svc, session_id, "", CallReply(call.not_heard, call.language))
        if is_probable_hallucination(transcript):
            return ReplyOut(session_id=session_id, language=call.language, ignored=True)
        reply = await call.handle(transcript)

    if reply.ended:
        svc.end(session_id)
    return await _reply_out(svc, session_id, transcript, reply)


@app.post("/api/end")
async def end_call(session_id: str = Form(...)) -> dict[str, Any]:
    """Hang up from the UI."""
    _service().end(session_id)
    return {"status": "ended"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web.app:app", host="127.0.0.1", port=8000)
