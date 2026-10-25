"""Orchestrator routing loop, driven by a scripted fake LLM (no network)."""

import json
from typing import Any

import pytest

from agent.llm_client import LLMResponse, ToolCall
from agent.orchestrator import STALE_TOOL_RESULT, Orchestrator


def tool_step(*calls: tuple[str, dict[str, Any]]) -> LLMResponse:
    """An LLM response that requests the given tool calls."""
    tcs = [ToolCall(id=f"call_{i}", name=name, arguments=args) for i, (name, args) in enumerate(calls)]
    return LLMResponse(
        content="",
        tool_calls=tcs,
        message={
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                for tc in tcs
            ],
        },
    )


def answer(text: str) -> LLMResponse:
    return LLMResponse(content=text, message={"role": "assistant", "content": text})


class ScriptedLLM:
    """Returns pre-baked responses in order and records what it was sent."""

    def __init__(self, *responses: LLMResponse | Exception) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeKnowledgeBase:
    name = "search_knowledge_base"

    def __init__(self) -> None:
        self.queries: list[str] = []

    def definition(self) -> dict[str, Any]:
        return {"type": "function", "function": {"name": self.name, "parameters": {"type": "object"}}}

    async def call(self, arguments: dict[str, Any]) -> str:
        self.queries.append(arguments["query"])
        return json.dumps({"results": [
            {"source": "warranty_policy.md", "section": "Standard Warranty Periods", "score": 0.8,
             "text": "NovaBand X200: 12 months"},
            {"source": "shipping_faq.md", "section": "Overview", "score": 0.3, "text": "unrelated"},
        ]})


class FakeMCP:
    """Stands in for MCPClient."""

    def __init__(self, fail: set[str] | None = None) -> None:
        self.fail = fail or set()
        self.calls: list[tuple[str, dict[str, Any]]] = []

    @property
    def tool_names(self) -> list[str]:
        return ["get_current_time", "days_between"]

    def openai_tools(self) -> list[dict[str, Any]]:
        return [{"type": "function", "function": {"name": n, "parameters": {"type": "object"}}}
                for n in self.tool_names]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        self.calls.append((name, arguments))
        if name in self.fail:
            raise RuntimeError("server exploded")
        if name == "get_current_time":
            return '{"date":"2026-10-25"}'
        return '{"days":238}'


def make(llm: ScriptedLLM, mcp: FakeMCP | None = None, **kwargs: Any) -> tuple[Orchestrator, FakeKnowledgeBase, FakeMCP]:
    kb, mcp = FakeKnowledgeBase(), mcp or FakeMCP()
    return Orchestrator(llm, local_tools=[kb], tool_server=mcp, **kwargs), kb, mcp


async def test_answer_without_tools() -> None:
    orch, kb, mcp = make(ScriptedLLM(answer("Hi! How can I help?")))
    result = await orch.run_turn("hello")
    assert result.answer == "Hi! How can I help?"
    assert result.tool_calls == [] and result.iterations == 1
    assert not kb.queries and not mcp.calls


async def test_rag_and_mcp_in_one_turn() -> None:
    """The key scenario: knowledge base + two MCP tools, results fed back to the LLM."""
    llm = ScriptedLLM(
        tool_step(("search_knowledge_base", {"query": "X200 warranty"}), ("get_current_time", {})),
        tool_step(("days_between", {"start_date": "2026-03-01", "end_date": "2026-10-25"})),
        answer("Yes, you're still covered."),
    )
    orch, kb, mcp = make(llm)
    result = await orch.run_turn("Is my X200 from March 1st still under warranty?")

    assert result.answer == "Yes, you're still covered."
    assert [(r.name, r.provider) for r in result.tool_calls] == [
        ("search_knowledge_base", "local"), ("get_current_time", "mcp"), ("days_between", "mcp")
    ]
    assert all(r.ok for r in result.tool_calls)
    assert kb.queries == ["X200 warranty"]
    assert [name for name, _ in mcp.calls] == ["get_current_time", "days_between"]
    # Every tool result reached the LLM, matched to its call id.
    final_request = llm.calls[-1]["messages"]
    tool_msgs = [m for m in final_request if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["call_0", "call_1", "call_0"]
    assert '"days":238' in tool_msgs[-1]["content"]
    # All tools (local + MCP) were offered to the LLM.
    offered = {t["function"]["name"] for t in llm.calls[0]["tools"]}
    assert offered == {"search_knowledge_base", "get_current_time", "days_between"}
    # Only strong hits are cited.
    assert result.sources == ["warranty_policy.md § Standard Warranty Periods"]


async def test_tool_failure_is_returned_to_llm_not_raised() -> None:
    llm = ScriptedLLM(tool_step(("get_current_time", {})), answer("Sorry, I couldn't get the date."))
    orch, _, _ = make(llm, FakeMCP(fail={"get_current_time"}))
    result = await orch.run_turn("what's the date?")
    record = result.tool_calls[0]
    assert not record.ok
    assert "server exploded" in record.result
    assert "server exploded" in llm.calls[1]["messages"][-1]["content"]
    assert result.answer == "Sorry, I couldn't get the date."


async def test_unknown_tool_and_bad_arguments_become_errors() -> None:
    bad_args = tool_step(("days_between", {}))
    bad_args.tool_calls[0] = ToolCall(id="call_0", name="days_between", arguments={}, parse_error="invalid JSON")
    llm = ScriptedLLM(tool_step(("launch_rockets", {})), bad_args, answer("done"))
    orch, _, mcp = make(llm)
    result = await orch.run_turn("do things")
    assert [r.ok for r in result.tool_calls] == [False, False]
    assert result.tool_calls[0].provider == "unknown"
    assert "invalid JSON" in result.tool_calls[1].result
    assert mcp.calls == []  # neither call reached the server


async def test_iteration_limit_forces_final_answer_without_tools() -> None:
    llm = ScriptedLLM(*[tool_step(("get_current_time", {})) for _ in range(3)], answer("Best effort answer."))
    orch, _, mcp = make(llm, max_iterations=3)
    result = await orch.run_turn("loop forever")
    assert result.answer == "Best effort answer."
    assert len(mcp.calls) == 3
    assert llm.calls[-1]["tools"] is None  # the forced call can't request more tools
    assert result.iterations == 4


async def test_llm_failure_rolls_back_history() -> None:
    llm = ScriptedLLM(tool_step(("get_current_time", {})), RuntimeError("API down"))
    orch, _, _ = make(llm)
    before = list(orch.messages)
    with pytest.raises(RuntimeError, match="API down"):
        await orch.run_turn("what's the date?")
    assert orch.messages == before


async def test_history_keeps_calls_but_stubs_old_tool_results() -> None:
    llm = ScriptedLLM(
        tool_step(("search_knowledge_base", {"query": "warranty"})), answer("12 months."),
        answer("You're welcome!"),
    )
    orch, _, _ = make(llm)
    await orch.run_turn("How long is the warranty?")
    await orch.run_turn("Thanks!")

    second_request = llm.calls[-1]["messages"]
    roles = [m["role"] for m in second_request]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "user"]
    assert second_request[2]["tool_calls"][0]["function"]["name"] == "search_knowledge_base"
    assert second_request[3]["content"] == STALE_TOOL_RESULT
    assert second_request[4]["content"] == "12 months."


async def test_history_is_capped_to_recent_turns() -> None:
    llm = ScriptedLLM(*[answer(f"answer {i}") for i in range(5)])
    orch, _, _ = make(llm, max_history_turns=2)
    for i in range(5):
        await orch.run_turn(f"question {i}")
    assert [m["content"] for m in orch.messages[1:]] == ["question 3", "answer 3", "question 4", "answer 4"]
    assert orch.messages[0]["role"] == "system"


async def test_empty_llm_answer_gets_fallback() -> None:
    orch, _, _ = make(ScriptedLLM(answer("")))
    assert "rephrase" in (await orch.run_turn("???")).answer


def test_duplicate_tool_names_are_rejected() -> None:
    class ClashingMCP(FakeMCP):
        @property
        def tool_names(self) -> list[str]:
            return ["search_knowledge_base"]

    with pytest.raises(ValueError, match="both locally and on the MCP server"):
        make(ScriptedLLM(), ClashingMCP())
