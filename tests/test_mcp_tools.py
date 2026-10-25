"""MCP server tools, called directly and through a real stdio MCP session."""

from datetime import datetime
from pathlib import Path

import httpx
import pytest

import mcp_server.server as server
from agent.mcp_client import MCPClient, MCPToolError, mcp_tool_to_openai
from mcp_server.weather import fetch_current_weather


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


# --- get_weather (network mocked) -------------------------------------------


def _mock_open_meteo(request: httpx.Request) -> httpx.Response:
    if "geocoding" in request.url.host:
        if request.url.params["name"] == "Nowhere":
            return httpx.Response(200, json={})
        return httpx.Response(
            200,
            json={"results": [{"name": "Mumbai", "admin1": "Maharashtra", "country": "India",
                               "latitude": 19.07, "longitude": 72.88}]},
        )
    return httpx.Response(
        200,
        json={"current": {"time": "2026-10-25T21:45", "temperature_2m": 30.3, "apparent_temperature": 32.8,
                          "relative_humidity_2m": 52, "weather_code": 0, "wind_speed_10m": 7.7,
                          "wind_direction_10m": 357}},
    )


async def test_weather_parses_open_meteo_response() -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(_mock_open_meteo)) as client:
        weather = await fetch_current_weather("Mumbai", client)
    assert weather["city"] == "Mumbai"
    assert weather["temperature_c"] == 30.3
    assert weather["conditions"] == "clear sky"


async def test_weather_unknown_city() -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(_mock_open_meteo)) as client:
        with pytest.raises(ValueError, match="could not find"):
            await fetch_current_weather("Nowhere", client)


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
            "get_current_time", "get_weather", "calculate", "days_between", "read_system_log"
        }
        assert all(t["function"]["parameters"]["type"] == "object" for t in mcp.openai_tools())
        assert '"result":147.0' in await mcp.call_tool("calculate", {"expression": "sqrt(1764) * 3.5"})
        with pytest.raises(MCPToolError, match="unsupported"):
            await mcp.call_tool("calculate", {"expression": "__import__('os')"})
