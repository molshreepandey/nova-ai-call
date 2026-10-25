"""MCP server tools, called directly and through a real stdio MCP session."""

from datetime import datetime
from pathlib import Path

import pytest

import mcp_server.server as server
from agent.mcp_client import MCPClient, MCPToolError, mcp_tool_to_openai
from mcp_server.finance import IST, market_status, project_investment


# --- get_current_time -------------------------------------------------------


def test_current_time_defaults_to_india() -> None:
    result = server.get_current_time()
    assert result["timezone"] == "Asia/Kolkata"
    assert result["utc_offset"] == "+0530"
    assert datetime.fromisoformat(result["iso"]).date().isoformat() == result["date"]


def test_current_time_rejects_unknown_timezone() -> None:
    with pytest.raises(ValueError, match="unknown timezone"):
        server.get_current_time("Mars/Olympus_Mons")


# --- days_between -----------------------------------------------------------


@pytest.mark.parametrize(
    ("start", "end", "days"),
    [("2026-03-01", "2026-10-25", 238), ("2026-10-25", "2026-03-01", -238), ("2024-02-28", "2024-03-01", 2)],
)
def test_days_between(start: str, end: str, days: int) -> None:
    assert server.days_between(start, end)["days"] == days


def test_days_between_requires_iso_dates() -> None:
    with pytest.raises(ValueError, match="ISO"):
        server.days_between("1st March 2026", "2026-10-25")


# --- read_system_log --------------------------------------------------------


def test_read_system_log_returns_tail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log = tmp_path / "agent.log"
    log.write_text("\n".join(f"line {i}" for i in range(50)) + "\n", encoding="utf-8")
    monkeypatch.setattr(server, "LOG_FILE", log)
    assert server.read_system_log(3)["lines"] == ["line 47", "line 48", "line 49"]


def test_read_system_log_clamps_line_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log = tmp_path / "agent.log"
    log.write_text("\n".join(str(i) for i in range(500)), encoding="utf-8")
    monkeypatch.setattr(server, "LOG_FILE", log)
    assert len(server.read_system_log(10_000)["lines"]) == server.MAX_LOG_LINES
    assert len(server.read_system_log(-5)["lines"]) == 1


def test_read_system_log_takes_no_path_argument() -> None:
    """Security: the tool's only parameter is a line count; no path can be supplied."""
    import inspect

    assert list(inspect.signature(server.read_system_log).parameters) == ["lines"]


# --- finance tools ----------------------------------------------------------


def test_sip_projection_matches_closed_form() -> None:
    result = server.investment_projection(annual_return_pct=12, years=10, monthly_sip=10_000)
    r, n = 0.01, 120
    expected = 10_000 * ((1 + r) ** n - 1) / r * (1 + r)  # annuity due
    assert result["total_invested_inr"] == 1_200_000
    assert result["estimated_value_inr"] == pytest.approx(expected, abs=0.01)  # rounded to paise
    assert "not guaranteed" in result["disclaimer"]


def test_lumpsum_and_step_up_projection() -> None:
    assert project_investment(lumpsum=100_000, annual_return_pct=12, years=10).estimated_value == pytest.approx(  # noqa: E501
        100_000 * 1.01**120, abs=0.01
    )
    stepped = project_investment(monthly_sip=5_000, annual_return_pct=12, years=2, annual_step_up_pct=10)
    assert stepped.total_invested == 5_000 * 12 + 5_500 * 12


@pytest.mark.parametrize(
    "kwargs",
    [
        {"annual_return_pct": 12, "years": 10},  # no amount
        {"monthly_sip": -1, "annual_return_pct": 12, "years": 10},
        {"monthly_sip": 1000, "annual_return_pct": 95, "years": 10},
        {"monthly_sip": 1000, "annual_return_pct": 12, "years": 500},
    ],
)
def test_projection_rejects_bad_input(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        project_investment(**kwargs)


@pytest.mark.parametrize(
    ("when", "status", "next_open_day"),
    [
        (datetime(2026, 10, 26, 10, 0), "open", ""),
        (datetime(2026, 10, 26, 9, 5), "pre-open session", "Monday"),
        (datetime(2026, 10, 23, 16, 0), "closed", "Monday"),  # Friday evening
        (datetime(2026, 10, 25, 12, 0), "closed", "Monday"),  # Sunday
    ],
)
def test_market_status(when: datetime, status: str, next_open_day: str) -> None:
    result = market_status(when.replace(tzinfo=IST))
    assert result["status"] == status
    assert result["next_open"].startswith(next_open_day)


# --- schema conversion ------------------------------------------------------


def test_mcp_schema_converts_to_function_format() -> None:
    from mcp.types import Tool

    tool = Tool(
        name="days_between",
        description="Count days.",
        inputSchema={"type": "object", "title": "days_betweenArguments",
                     "properties": {"start_date": {"type": "string", "title": "Start Date"}},
                     "required": ["start_date"]},
    )
    converted = mcp_tool_to_openai(tool)
    assert converted["type"] == "function"
    assert converted["function"]["name"] == "days_between"
    assert converted["function"]["parameters"] == {
        "type": "object", "properties": {"start_date": {"type": "string"}}, "required": ["start_date"]
    }


# --- real stdio round-trip --------------------------------------------------


@pytest.mark.integration
async def test_stdio_session_lists_and_calls_tools() -> None:
    async with MCPClient() as mcp:
        assert set(mcp.tool_names) == {
            "get_current_time", "get_market_status", "investment_projection",
            "calculate", "days_between", "read_system_log",
        }
        assert all(t["function"]["parameters"]["type"] == "object" for t in mcp.openai_tools())
        assert '"result":147.0' in await mcp.call_tool("calculate", {"expression": "sqrt(1764) * 3.5"})
        with pytest.raises(MCPToolError, match="unsupported"):
            await mcp.call_tool("calculate", {"expression": "__import__('os')"})
