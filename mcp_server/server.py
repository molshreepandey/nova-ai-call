"""NovaTech system-tools MCP server (FastMCP, stdio transport).

The agent spawns this as a subprocess and talks MCP over stdin/stdout, so this
process must never print to stdout. Logs go to logs/mcp_server.log.

Run standalone (e.g. with the MCP Inspector):
    python -m mcp_server.server
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from mcp.server.fastmcp import FastMCP

from config import LOG_DIR, LOG_FILE
from mcp_server.calculator import safe_eval
from mcp_server.weather import fetch_current_weather

MAX_LOG_LINES = 200

LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / "mcp_server.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("mcp_server")

mcp = FastMCP("novatech-system-tools")


@mcp.tool()
def get_current_time(timezone: str = "Asia/Kolkata") -> dict[str, Any]:
    """Get the current date and time in an IANA timezone.

    Use this whenever the answer depends on today's date or the current time.

    Args:
        timezone: IANA timezone name, e.g. "Asia/Kolkata", "America/New_York", "UTC".
    """
    try:
        tz = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"unknown timezone {timezone!r}; use an IANA name like 'Asia/Kolkata'") from exc
    now = datetime.now(tz)
    logger.info("get_current_time(%s)", timezone)
    return {
        "timezone": timezone,
        "iso": now.isoformat(timespec="seconds"),
        "date": now.date().isoformat(),
        "time": now.strftime("%H:%M"),
        "weekday": now.strftime("%A"),
        "utc_offset": now.strftime("%z"),
    }


@mcp.tool()
async def get_weather(city: str) -> dict[str, Any]:
    """Get the current weather for a city (temperature, conditions, humidity, wind).

    Args:
        city: City name, e.g. "Mumbai" or "London".
    """
    logger.info("get_weather(%s)", city)
    return await fetch_current_weather(city)


@mcp.tool()
def calculate(expression: str) -> dict[str, Any]:
    """Safely evaluate a math expression.

    Supports + - * / // % ** (or ^), parentheses, constants pi/e/tau and functions
    sqrt, sin, cos, tan, asin, acos, atan, log (natural, or log(x, base)), log10,
    log2, exp, abs, round, floor, ceil, factorial, radians, degrees.
    Trig functions use radians. For differences between dates use days_between.

    Args:
        expression: The expression, e.g. "sqrt(1764) * 3.5".
    """
    result = safe_eval(expression)
    logger.info("calculate(%s) = %s", expression, result)
    return {"expression": expression, "result": result}


@mcp.tool()
def days_between(start_date: str, end_date: str) -> dict[str, Any]:
    """Count the days from start_date to end_date (negative if end is before start).

    Args:
        start_date: ISO date, YYYY-MM-DD.
        end_date: ISO date, YYYY-MM-DD.
    """
    try:
        start, end = date.fromisoformat(start_date.strip()), date.fromisoformat(end_date.strip())
    except ValueError as exc:
        raise ValueError("dates must be ISO formatted, e.g. 2026-03-01") from exc
    days = (end - start).days
    logger.info("days_between(%s, %s) = %d", start, end, days)
    return {"start_date": start.isoformat(), "end_date": end.isoformat(), "days": days}


@mcp.tool()
def read_system_log(lines: int = 20) -> dict[str, Any]:
    """Read the last N lines of the voice agent's own log file (logs/agent.log).

    Only this one file can be read; no other paths are accessible.

    Args:
        lines: How many trailing lines to return (1-200).
    """
    lines = max(1, min(int(lines), MAX_LOG_LINES))
    if not LOG_FILE.exists():
        return {"file": "logs/agent.log", "lines": []}
    with LOG_FILE.open(encoding="utf-8", errors="replace") as fh:
        tail = [line.rstrip("\n") for line in deque(fh, maxlen=lines)]
    logger.info("read_system_log(%d)", lines)
    return {"file": "logs/agent.log", "lines": tail}


if __name__ == "__main__":
    mcp.run(transport="stdio")
