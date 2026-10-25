"""Optional browser UI: the same agent behind a small FastAPI app.

    python -m web.app          # then open http://127.0.0.1:8000

The browser records the microphone (MediaRecorder) and posts each utterance here.
The server reuses exactly the CLI's components: Groq Whisper STT, the Orchestrator
with the local RAG tool + MCP tools, and edge-tts. Each browser tab gets its own
conversation (session id), so memory works per caller.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from groq import RateLimitError
from pydantic import BaseModel

from agent.llm_client import GroqLLMClient
from agent.mcp_client import MCPClient
from agent.orchestrator import Orchestrator, TurnResult
from agent.prompts import GOODBYE, GREETING, is_hangup
from config import settings, setup_logging
from rag.retriever import Retriever
from rag.tool import KnowledgeBaseTool
from voice.stt import GroqWhisperSTT
from voice.tts import EdgeTTS

logger = logging.getLogger("web")

STATIC_DIR = Path(__file__).parent / "static"
MAX_SESSIONS = 50
MAX_AUDIO_BYTES = 10 * 1024 * 1024


class Session:
    """One caller's conversation; the lock serialises that caller's turns."""

    def __init__(self, orchestrator: Orchestrator) -> None:
        self.orchestrator = orchestrator
        self.lock = asyncio.Lock()


class AgentService:
    """Shared, process-wide components plus per-session orchestrators."""

    def __init__(self) -> None:
        self.llm = GroqLLMClient(
            api_key=settings.require_groq_key(),
            model=settings.llm_model,
            reasoning_effort=settings.reasoning_effort,
        )
        self.stt = GroqWhisperSTT(api_key=settings.require_groq_key(), model=settings.stt_model)
        self.tts = EdgeTTS(settings.tts_voice)
        self.retriever = Retriever()
        self.kb_tool = KnowledgeBaseTool(self.retriever)
        self.mcp = MCPClient()
        self._sessions: OrderedDict[str, Session] = OrderedDict()

    def session(self, session_id: str) -> Session:
        """Get or create the session, evicting the least recently used beyond the cap."""
        if session_id in self._sessions:
            self._sessions.move_to_end(session_id)
            return self._sessions[session_id]
        session = Session(
            Orchestrator(
                self.llm,
                local_tools=[self.kb_tool],
                tool_server=self.mcp,
                max_iterations=settings.max_tool_iterations,
                max_history_turns=settings.max_history_turns,
            )
        )
        self._sessions[session_id] = session
        if len(self._sessions) > MAX_SESSIONS:
            self._sessions.popitem(last=False)
        return session

    def end(self, session_id: str) -> None:
        """Forget a session (hang up)."""
        self._sessions.pop(session_id, None)

    async def speech(self, text: str) -> str | None:
        """MP3 for ``text`` as base64, or None if TTS is unavailable (text still shows)."""
        try:
            return base64.b64encode(await self.tts.synthesize(text)).decode()
        except Exception as exc:  # noqa: BLE001 - TTS is best-effort
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


app = FastAPI(title="NovaTech Voice Agent", lifespan=lifespan)


def _service() -> AgentService:
    if service is None:
        raise HTTPException(503, "agent is starting")
    return service


class StartResponse(BaseModel):
    session_id: str
    greeting: str
    audio: str | None


class TurnResponse(BaseModel):
    transcript: str
    answer: str
    tool_calls: list[dict[str, Any]]
    sources: list[str]
    audio: str | None
    ended: bool = False


@app.get("/")
async def index() -> FileResponse:
    """The single-page UI."""
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/start")
async def start() -> StartResponse:
    """Begin a call: new session plus the spoken greeting."""
    svc = _service()
    session_id = uuid.uuid4().hex
    svc.session(session_id)
    return StartResponse(session_id=session_id, greeting=GREETING, audio=await svc.speech(GREETING))


@app.post("/api/turn")
async def turn(
    session_id: str = Form(...),
    text: str | None = Form(None),
    audio: UploadFile | None = File(None),
) -> TurnResponse:
    """One caller turn from recorded audio (or typed text): STT, agent, TTS."""
    svc = _service()
    if audio is not None:
        data = await audio.read()
        if len(data) > MAX_AUDIO_BYTES:
            raise HTTPException(413, "recording too long")
        transcript = await svc.stt.transcribe(data, filename=audio.filename or "speech.webm")
    else:
        transcript = (text or "").strip()

    if not transcript:
        reply = "Sorry, I didn't catch that. Could you say it again?"
        return TurnResponse(transcript="", answer=reply, tool_calls=[], sources=[], audio=await svc.speech(reply))
    if is_hangup(transcript):
        svc.end(session_id)
        return TurnResponse(
            transcript=transcript, answer=GOODBYE, tool_calls=[], sources=[],
            audio=await svc.speech(GOODBYE), ended=True,
        )

    session = svc.session(session_id)
    async with session.lock:
        try:
            result: TurnResult = await session.orchestrator.run_turn(transcript)
        except RateLimitError:
            result = TurnResult(answer="Sorry, I'm handling a lot of requests right now. Please ask again in a few seconds.")
        except Exception:
            logger.exception("turn failed")
            result = TurnResult(answer="Sorry, I ran into a problem answering that. Could you try again?")
    return TurnResponse(
        transcript=transcript,
        answer=result.answer,
        tool_calls=[asdict(c) for c in result.tool_calls],
        sources=result.sources,
        audio=await svc.speech(result.answer),
    )


@app.post("/api/end")
async def end(session_id: str = Form(...)) -> dict[str, str]:
    """Hang up from the UI."""
    _service().end(session_id)
    return {"status": "ended"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web.app:app", host="127.0.0.1", port=8000)
