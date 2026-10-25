"""NovaTech voice agent: CLI entry point and the "call" loop.

    python main.py --text          # type instead of speaking (debugging)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from contextlib import AsyncExitStack

from agent.llm_client import GroqLLMClient
from agent.mcp_client import MCPClient
from agent.orchestrator import Orchestrator
from agent.prompts import GOODBYE, GREETING, HANGUP_PHRASES
from config import ConfigError, settings, setup_logging
from display import console, show_agent, show_banner, show_error, show_turn, show_user
from rag.retriever import Retriever
from rag.tool import KnowledgeBaseTool

logger = logging.getLogger("main")


def is_hangup(text: str) -> bool:
    """True if the caller asked to end the call."""
    normalised = " ".join(text.lower().replace(".", " ").replace("!", " ").split())
    return any(phrase in normalised for phrase in HANGUP_PHRASES)


async def read_text() -> str | None:
    """Read one typed line; ``None`` on EOF (Ctrl-D / end of piped input)."""
    try:
        if console.is_terminal:
            return (await asyncio.to_thread(console.input, "\n[bold green]You:[/] ")).strip()
        line = (await asyncio.to_thread(input)).strip()
        show_user(line)  # piped input isn't echoed by the terminal, so print it
        return line
    except EOFError:
        return None


async def run_call(args: argparse.Namespace) -> None:
    """Connect all components and run the conversation until the caller hangs up."""
    llm = GroqLLMClient(
        api_key=settings.require_groq_key(),
        model=settings.llm_model,
        reasoning_effort=settings.reasoning_effort,
    )
    retriever = Retriever()
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
        )
        show_banner(settings.llm_model, [kb_tool.name, *mcp.tool_names], "text")
        show_agent(GREETING)

        while True:
            user_text = await read_text()
            if user_text is None:
                break
            if not user_text:
                continue
            if is_hangup(user_text):
                break
            try:
                result = await orchestrator.run_turn(user_text)
            except Exception as exc:  # noqa: BLE001 - keep the call alive on API errors
                logger.exception("turn failed")
                show_error(f"I hit a problem answering that ({type(exc).__name__}). Please try again.")
                continue
            show_turn(result)

        show_agent(GOODBYE)


def parse_args() -> argparse.Namespace:
    """Parse command-line flags."""
    parser = argparse.ArgumentParser(description="NovaTech voice support agent (RAG + MCP).")
    parser.add_argument("--text", action="store_true", help="type queries instead of speaking")
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    setup_logging()
    try:
        asyncio.run(run_call(args))
    except ConfigError as exc:
        show_error(str(exc))
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        show_agent(GOODBYE)


if __name__ == "__main__":
    main()
