"""NovaWealth voice agent: CLI entry point and the "call" loop.

    python main.py                 # voice call, push-to-talk (Enter to start/stop)
    python main.py --duration 6    # voice call, fixed 6-second recordings
    python main.py --text          # type instead of speaking (add --tts to hear answers)
    python main.py --lang hi       # skip the language question (en or hi)
    python main.py --list-devices  # show audio devices

Every call opens by asking "English or Hindi?"; the rest of the call follows that choice.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from contextlib import AsyncExitStack
from typing import Protocol

from agent.call import CallSession
from agent.language import LANGUAGES, Language
from agent.llm_client import GroqLLMClient
from agent.mcp_client import MCPClient
from agent.orchestrator import Orchestrator
from config import ConfigError, settings, setup_logging
from display import console, show_agent, show_banner, show_error, show_turn, show_user
from rag.retriever import Retriever
from rag.tool import KnowledgeBaseTool
from voice.recorder import MicrophoneError, Recorder
from voice.stt import GroqWhisperSTT, SpeechToText, is_probable_hallucination
from voice.tts import EdgeTTS, SilentTTS, TextToSpeech

logger = logging.getLogger("main")


class Listener(Protocol):
    """Source of caller utterances."""

    async def listen(self, language: Language | None) -> str | None:
        """Return the next utterance, "" if nothing usable was heard, None to hang up.

        ``language`` is the call language so far (None = not chosen yet).
        """
        ...


class TextListener:
    """Reads typed (or piped) lines from stdin."""

    async def listen(self, language: Language | None) -> str | None:
        """Read one line; ``None`` on EOF (Ctrl-D / end of piped input)."""
        try:
            if console.is_terminal:
                return (await asyncio.to_thread(console.input, "\n[bold green]You:[/] ")).strip()
            line = (await asyncio.to_thread(input)).strip()
            show_user(line)  # piped input isn't echoed by the terminal, so print it
            return line
        except EOFError:
            return None


class VoiceListener:
    """Records from the microphone and transcribes with an STT provider."""

    def __init__(self, recorder: Recorder, stt: SpeechToText, fixed_seconds: float | None = None) -> None:
        self._recorder = recorder
        self._stt = stt
        self._fixed_seconds = fixed_seconds

    async def listen(self, language: Language | None) -> str | None:
        """Capture one utterance (push-to-talk or fixed duration) and return its transcript."""
        try:
            if self._fixed_seconds:
                console.print(f"\n[bold green]● Listening for {self._fixed_seconds:g}s...[/] speak now")
                clip = await asyncio.to_thread(self._recorder.record_fixed, self._fixed_seconds)
            else:
                await asyncio.to_thread(console.input, "\n[bold green]Press Enter to talk[/] ")
                console.print("[bold red]● Recording...[/] press Enter to stop")
                clip = await asyncio.to_thread(self._recorder.record_until, input)
        except EOFError:
            return None

        if clip.is_silent:
            return ""
        with console.status("Transcribing..."):
            text = await self._stt.transcribe(clip.wav_bytes, language=language)
        if is_probable_hallucination(text):
            return ""
        show_user(text)
        return text


async def say(tts: TextToSpeech, text: str, language: Language | None) -> None:
    """Print a line from the agent and speak it in the call language's voice."""
    show_agent(text)
    await tts.speak(text, settings.tts_voice(language))


def build_io(args: argparse.Namespace) -> tuple[Listener, TextToSpeech, str]:
    """Pick the input source and speaker for the chosen mode."""
    if args.text:
        tts: TextToSpeech = EdgeTTS() if args.tts else SilentTTS()
        return TextListener(), tts, "text" + (" + tts" if args.tts else "")

    recorder = Recorder(sample_rate=settings.sample_rate)
    mic = recorder.check_device()
    stt = GroqWhisperSTT(
        api_key=settings.require_groq_key(), model=settings.stt_model, models_by_language={"hi": settings.stt_model_hi}
    )
    mode = f"voice ({'fixed ' + format(args.duration, 'g') + 's' if args.duration else 'push-to-talk'}, mic: {mic})"
    return VoiceListener(recorder, stt, args.duration), EdgeTTS(), mode


async def run_call(args: argparse.Namespace) -> None:
    """Connect all components and run the conversation until the caller hangs up."""
    listener, tts, mode = build_io(args)
    llm = GroqLLMClient(
        api_key=settings.require_groq_key(),
        model=settings.llm_model,
        reasoning_effort=settings.reasoning_effort,
        fallback_model=settings.llm_fallback_model,
    )
    retriever = Retriever()
    if retriever.count() == 0:
        raise ConfigError("The knowledge base is empty. Run `python -m rag.ingest` first.")
    kb_tool = KnowledgeBaseTool(retriever)

    # The MCP session must be opened and closed in this task (anyio cancel scopes), so
    # only the embedding warm-up runs concurrently.
    async with AsyncExitStack() as stack:
        with console.status("Loading embedding model and starting MCP server..."):
            warm_up = asyncio.create_task(asyncio.to_thread(retriever.warm_up))
            mcp = await stack.enter_async_context(MCPClient())
            await warm_up

        orchestrator = Orchestrator(
            llm,
            local_tools=[kb_tool],
            tool_server=mcp,
            max_iterations=settings.max_tool_iterations,
            max_history_turns=settings.max_history_turns,
            require_tool_first=settings.require_tool_first,
        )
        call = CallSession(orchestrator, language=args.lang)
        show_banner(settings.llm_model, [kb_tool.name, *mcp.tool_names], mode)
        await say(tts, call.opening(), call.language)

        while True:
            try:
                user_text = await listener.listen(call.language)
            except Exception as exc:  # noqa: BLE001 - e.g. STT API error; keep the call alive
                logger.exception("listening failed")
                show_error(f"{type(exc).__name__}: {exc}")
                user_text = ""
            if user_text is None:
                break
            if not user_text:
                await say(tts, call.not_heard, call.language)
                continue

            with console.status("Thinking..."):
                reply = await call.handle(user_text)
            if reply.turn:
                show_turn(reply.turn)
                await tts.speak(reply.text, settings.tts_voice(reply.language))
            else:
                await say(tts, reply.text, reply.language)
            if reply.ended:
                return

        await say(tts, call.goodbye, call.language)  # input ended (EOF) without a hang-up phrase


def parse_args() -> argparse.Namespace:
    """Parse command-line flags."""
    parser = argparse.ArgumentParser(description="NovaWealth voice investor-support agent (RAG + MCP).")
    parser.add_argument("--text", action="store_true", help="type queries instead of speaking")
    parser.add_argument("--tts", action="store_true", help="in --text mode, also speak the answers")
    parser.add_argument(
        "--duration",
        type=float,
        metavar="SECONDS",
        help="record fixed-length clips instead of push-to-talk",
    )
    parser.add_argument("--lang", choices=["en", "hi"], help="call language; skips the language question")
    parser.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    if args.list_devices:
        import sounddevice as sd

        console.print(sd.query_devices())
        return
    setup_logging(console=console)
    try:
        asyncio.run(run_call(args))
    except (ConfigError, MicrophoneError) as exc:
        show_error(str(exc))
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        show_agent(LANGUAGES["en"].goodbye)


if __name__ == "__main__":
    main()
