"""The agent's brain: LLM tool-calling loop, tool routing and conversation memory.

Routing is done by the LLM. Each turn the model sees every tool (the local RAG
tool plus whatever the MCP server advertises) and decides which, if any, to call.
The orchestrator executes the calls, feeds results back, and repeats until the
model answers in plain text or the iteration budget runs out.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from agent.llm_client import LLMClient, ToolCall
from agent.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

MAX_TOOL_RESULT_CHARS = 6000
SOURCE_SCORE_MARGIN = 0.15
STALE_TOOL_RESULT = '{"note":"result from an earlier turn, removed; call the tool again if needed"}'
ToolProvider = Literal["local", "mcp"]


class LocalTool(Protocol):
    """A tool implemented inside the agent process (e.g. the RAG knowledge base)."""

    name: str

    def definition(self) -> dict[str, Any]:
        """Function-calling schema for the LLM."""
        ...

    async def call(self, arguments: dict[str, Any]) -> str:
        """Execute the tool and return its result as text."""
        ...


class ToolServer(Protocol):
    """A remote tool provider, such as :class:`agent.mcp_client.MCPClient`."""

    @property
    def tool_names(self) -> list[str]: ...

    def openai_tools(self) -> list[dict[str, Any]]: ...

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str: ...


@dataclass(frozen=True)
class ToolCallRecord:
    """Trace entry for one executed tool call."""

    name: str
    provider: ToolProvider | Literal["unknown"]
    arguments: dict[str, Any]
    result: str
    ok: bool
    duration_ms: int


@dataclass
class TurnResult:
    """Everything produced while answering one user utterance."""

    answer: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    iterations: int = 0


class Orchestrator:
    """Runs conversation turns against an LLM with local and MCP tools."""

    def __init__(
        self,
        llm: LLMClient,
        local_tools: list[LocalTool],
        tool_server: ToolServer | None = None,
        max_iterations: int = 5,
        max_history_turns: int = 8,
        system_prompt: str = SYSTEM_PROMPT,
        require_tool_first: bool = False,
    ) -> None:
        """
        Args:
            llm: The routing LLM.
            local_tools: In-process tools such as the RAG knowledge base.
            tool_server: Remote tool provider (the MCP client), if any.
            max_iterations: LLM <-> tool round-trips per turn before forcing an answer.
            max_history_turns: Conversation turns kept in memory.
            system_prompt: Initial system prompt.
            require_tool_first: Force at least one tool call on each turn's first step,
                so every answer is grounded in a tool result. Prompting alone proved
                unreliable for this (e.g. in Hindi the model answered from memory).
        """
        self._llm = llm
        self._local = {t.name: t for t in local_tools}
        self._server = tool_server
        self._max_iterations = max_iterations
        self._max_history_turns = max_history_turns
        self._system_prompt = system_prompt
        self._require_tool_first = require_tool_first

        remote = tool_server.tool_names if tool_server else []
        clash = set(self._local) & set(remote)
        if clash:
            raise ValueError(f"tool names defined both locally and on the MCP server: {sorted(clash)}")
        self._tool_defs = [t.definition() for t in local_tools] + (
            tool_server.openai_tools() if tool_server else []
        )
        self.messages: list[dict[str, Any]] = []
        self.reset()
        logger.info("[ROUTER] tools available: local=%s mcp=%s", list(self._local), remote)

    def reset(self) -> None:
        """Start a fresh conversation (keeps only the system prompt)."""
        self.messages = [{"role": "system", "content": self._system_prompt}]

    def set_system_prompt(self, prompt: str) -> None:
        """Replace the system prompt mid-call (e.g. after the caller picks a language)."""
        self._system_prompt = prompt
        self.messages[0] = {"role": "system", "content": prompt}

    async def run_turn(self, user_text: str) -> TurnResult:
        """Answer one user message, calling tools as the LLM requests.

        On an unrecoverable error (e.g. the LLM API is down) the turn's messages are
        rolled back so history stays valid, and the exception propagates.

        Afterwards the turn's tool results are compacted (see :meth:`_compact_turn`).
        """
        checkpoint = len(self.messages)
        self.messages.append({"role": "user", "content": user_text})
        result = TurnResult(answer="")
        try:
            for iteration in range(1, self._max_iterations + 1):
                result.iterations = iteration
                choice = "required" if self._require_tool_first and iteration == 1 else "auto"
                response = await self._llm.complete(self.messages, self._tool_defs, tool_choice=choice)
                self.messages.append(response.message)

                if not response.tool_calls:
                    result.answer = response.content
                    logger.info("[ROUTER] final answer after %d LLM call(s)", iteration)
                    break

                logger.info(
                    "[ROUTER] iteration %d: LLM chose tool(s): %s",
                    iteration,
                    ", ".join(tc.name for tc in response.tool_calls),
                )
                records = await asyncio.gather(*(self._execute(tc) for tc in response.tool_calls))
                for tc, record in zip(response.tool_calls, records):
                    result.tool_calls.append(record)
                    self.messages.append({"role": "tool", "tool_call_id": tc.id, "content": record.result})
            else:
                result.answer = await self._force_answer()
                result.iterations += 1
        except Exception:
            del self.messages[checkpoint:]
            raise

        if not result.answer:
            result.answer = "Sorry, I couldn't come up with an answer to that. Could you rephrase?"
        result.sources = _collect_sources(result.tool_calls)
        self._compact_turn(checkpoint)
        return result

    def _compact_turn(self, checkpoint: int) -> None:
        """Shrink the finished turn's tool results, then drop the oldest turns beyond the cap.

        Raw tool results would be resent on every later call (costly on Groq's
        tokens-per-minute limits) and tempt the model to reuse stale data. Removing the
        tool-call messages entirely is worse: the model then imitates the history and
        answers from memory without tools. So the calls stay and only results shrink.
        """
        for message in self.messages[checkpoint:]:
            if message["role"] == "tool":
                message["content"] = STALE_TOOL_RESULT

        user_turns = [i for i, m in enumerate(self.messages) if m["role"] == "user"]
        if len(user_turns) > self._max_history_turns:
            del self.messages[1 : user_turns[-self._max_history_turns]]

    async def _force_answer(self) -> str:
        """Called when the tool budget is exhausted: ask for an answer with tools disabled."""
        logger.warning("[ROUTER] hit max of %d tool iterations; forcing a final answer", self._max_iterations)
        self.messages.append(
            {
                "role": "system",
                "content": "Tool budget exhausted. Answer now using only the information above.",
            }
        )
        response = await self._llm.complete(self.messages, tools=None)
        self.messages.append(response.message)
        return response.content

    async def _execute(self, call: ToolCall) -> ToolCallRecord:
        """Run one tool call. Failures become error results for the LLM; they never raise."""
        provider: ToolProvider | Literal["unknown"]
        if call.name in self._local:
            provider = "local"
        elif self._server and call.name in self._server.tool_names:
            provider = "mcp"
        else:
            provider = "unknown"

        start = time.perf_counter()
        ok = False
        try:
            if call.parse_error:
                raise ValueError(call.parse_error)
            if provider == "local":
                logger.info("[RAG] calling %s(%s)", call.name, call.arguments)
                output = await self._local[call.name].call(call.arguments)
            elif provider == "mcp":
                assert self._server is not None
                output = await self._server.call_tool(call.name, call.arguments)
            else:
                raise LookupError(f"unknown tool {call.name!r}")
            ok = True
        except Exception as exc:  # noqa: BLE001 - any tool failure is reported back to the LLM
            logger.warning("[TOOL] %s failed: %s", call.name, exc)
            output = json.dumps({"error": f"{type(exc).__name__}: {exc}"})

        if len(output) > MAX_TOOL_RESULT_CHARS:
            output = output[:MAX_TOOL_RESULT_CHARS] + "... [truncated]"
        return ToolCallRecord(
            name=call.name,
            provider=provider,
            arguments=call.arguments,
            result=output,
            ok=ok,
            duration_ms=int((time.perf_counter() - start) * 1000),
        )


def _collect_sources(records: list[ToolCallRecord]) -> list[str]:
    """Unique "file § section" citations from successful knowledge-base results, in order.

    Only hits scoring close to each search's best hit are cited; the tail of the
    top-k is usually loosely related context, not what the answer came from.
    """
    seen: dict[str, None] = {}
    for record in records:
        if record.name != "search_knowledge_base" or not record.ok:
            continue
        try:
            hits = json.loads(record.result).get("results", [])
        except (json.JSONDecodeError, AttributeError):
            continue
        if not hits:
            continue
        best = max(hit["score"] for hit in hits)
        for hit in hits:
            if hit["score"] >= best - SOURCE_SCORE_MARGIN:
                seen.setdefault(f"{hit['source']} § {hit['section']}", None)
    return list(seen)
