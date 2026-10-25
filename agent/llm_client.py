"""LLM client interface and the Groq (OpenAI-compatible) implementation."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

import time

from groq import AsyncGroq, BadRequestError, RateLimitError

logger = logging.getLogger(__name__)

# Some reasoning models emit their chain of thought inline, sometimes without the
# opening tag; it must never be spoken.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?।])\s+|\n+")


def _trim_repetition(text: str) -> str:
    """Cut a degenerate answer at the first repeated sentence.

    Models occasionally loop ("X, no, X... sorry, X...") on hard number phrasing.
    A looping answer must never be read aloud, so keep only the text before the
    first sentence that repeats an earlier one.
    """
    seen: set[str] = set()
    kept: list[str] = []
    for sentence in _SENTENCE_SPLIT.split(text):
        key = " ".join(sentence.lower().split())
        if not key:
            continue
        if key in seen:
            break
        seen.add(key)
        kept.append(sentence.strip())
    return " ".join(kept) if kept else text


def _strip_reasoning(text: str) -> str:
    text = _THINK_RE.sub("", text)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.strip()


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation requested by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]
    # Set when the model produced arguments that are not valid JSON.
    parse_error: str | None = None


@dataclass
class LLMResponse:
    """A normalised chat completion: text, tool calls and the message to store in history."""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    message: dict[str, Any] = field(default_factory=dict)


class LLMClient(Protocol):
    """Anything that can run one chat-completion step with optional tools."""

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
    ) -> LLMResponse:
        """Return the model's next message for ``messages``.

        ``tool_choice`` is "auto" (model decides) or "required" (must call a tool).
        """
        ...


PRIMARY_COOLDOWN_S = 60.0


def _rate_limit_reason(exc: RateLimitError) -> str:
    """Short human-readable reason, e.g. "tokens per day (TPD)"."""
    match = re.search(r"on ([a-z ]+\([A-Z]+\))", str(exc))
    return match.group(1) if match else "rate limit"


def _parse_arguments(raw: str | None) -> tuple[dict[str, Any], str | None]:
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        return {}, f"invalid JSON arguments: {exc}"
    if not isinstance(parsed, dict):
        return {}, "tool arguments must be a JSON object"
    return parsed, None


class GroqLLMClient:
    """Chat completions with tool calling via Groq's API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        reasoning_effort: str = "",
        temperature: float = 0.2,
        max_tokens: int = 512,
        fallback_model: str = "",
    ) -> None:
        """
        Args:
            api_key: Groq API key.
            model: Groq model id, e.g. "qwen/qwen3.8-27b".
            reasoning_effort: Passed through to reasoning models when non-empty.
            temperature: Sampling temperature; low keeps tool routing deterministic.
            max_tokens: Completion cap. Spoken answers are short, and a small cap
                stretches the free-tier tokens-per-minute budget.
            fallback_model: Model to switch to when ``model`` is rate-limited (Groq
                quotas are per model, so a second model keeps the call going when the
                first hits its per-minute or per-day token limit). Empty disables it.
        """
        self._client = AsyncGroq(api_key=api_key, max_retries=2)
        self.model = model
        self.fallback_model = fallback_model
        self._reasoning_effort = reasoning_effort
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._primary_blocked_until = 0.0

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
    ) -> LLMResponse:
        """Run one completion step.

        If Groq rejects a malformed tool call the model generated (``tool_use_failed``),
        the request is retried once, since this is usually transient.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        if self._reasoning_effort:
            kwargs["reasoning_effort"] = self._reasoning_effort
        if tools:
            kwargs.update(tools=tools, tool_choice=tool_choice)

        if self.fallback_model and time.monotonic() < self._primary_blocked_until:
            completion = await self._create(self._for_fallback(kwargs))
        else:
            try:
                completion = await self._create(kwargs)
            except RateLimitError as exc:
                if not self.fallback_model:
                    raise
                # Skip the primary for a minute instead of hitting its limit on every call.
                self._primary_blocked_until = time.monotonic() + PRIMARY_COOLDOWN_S
                logger.warning(
                    "[LLM] %s rate-limited (%s); switching to %s",
                    self.model, _rate_limit_reason(exc), self.fallback_model,
                )
                completion = await self._create(self._for_fallback(kwargs))

        if completion.usage:
            logger.debug(
                "[LLM] tokens: prompt=%d completion=%d",
                completion.usage.prompt_tokens,
                completion.usage.completion_tokens,
            )
        msg = completion.choices[0].message
        content = _trim_repetition(_strip_reasoning(msg.content or ""))
        tool_calls: list[ToolCall] = []
        for tc in msg.tool_calls or []:
            args, error = _parse_arguments(tc.function.arguments)
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args, parse_error=error))

        history_msg: dict[str, Any] = {"role": "assistant", "content": content}
        if msg.tool_calls:
            history_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"},
                }
                for tc in msg.tool_calls
            ]
        return LLMResponse(content=content, tool_calls=tool_calls, message=history_msg)

    async def _create(self, kwargs: dict[str, Any]) -> Any:
        """One API call; a malformed tool call the model generated is retried once."""
        try:
            return await self._client.chat.completions.create(**kwargs)
        except BadRequestError as exc:
            if "tool_use_failed" not in str(exc):
                raise
            logger.warning("[LLM] model produced a malformed tool call; retrying once")
            return await self._client.chat.completions.create(**kwargs)

    def _for_fallback(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Same request for the fallback model; the reasoning setting is model-specific."""
        fallback = {k: v for k, v in kwargs.items() if k != "reasoning_effort"}
        fallback["model"] = self.fallback_model
        return fallback
