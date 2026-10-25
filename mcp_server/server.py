"""NovaWealth system-tools MCP server (FastMCP, stdio transport).

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
from mcp_server.finance import format_inr, market_status, project_investment

MAX_LOG_LINES = 200

LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / "mcp_server.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("mcp_server")

mcp = FastMCP("novawealth-system-tools")


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
def get_market_status() -> dict[str, str]:
    """Check whether the Indian stock market (NSE/BSE equity) is open right now,
    and when it next opens. Based on regular trading hours; holidays not included."""
    logger.info("get_market_status()")
    return market_status()


@mcp.tool()
def investment_projection(
    annual_return_pct: float,
    years: float,
    monthly_sip: float = 0,
    lumpsum: float = 0,
    annual_step_up_pct: float = 0,
) -> dict[str, Any]:
    """Illustrative future value of a monthly SIP and/or a one-time lump sum in rupees,
    at an ASSUMED constant annual return (not a prediction or guarantee).

    Args:
        annual_return_pct: Assumed annual return in percent, e.g. 12.
        years: Investment period in years (up to 50).
        monthly_sip: Monthly SIP amount in rupees.
        lumpsum: One-time investment in rupees.
        annual_step_up_pct: Yearly percentage increase of the SIP amount (0-50).
    """
    result = project_investment(monthly_sip, lumpsum, annual_return_pct, years, annual_step_up_pct)
    logger.info("investment_projection(sip=%s, lumpsum=%s, %s%%, %sy) = %s",
                monthly_sip, lumpsum, annual_return_pct, years, result)
    return {
        "assumed_annual_return_pct": annual_return_pct,
        "years": years,
        "total_invested_inr": result.total_invested,
        "estimated_value_inr": result.estimated_value,
        "estimated_gain_inr": result.estimated_gain,
        "total_invested_display": format_inr(result.total_invested),
        "estimated_value_display": format_inr(result.estimated_value),
        "estimated_gain_display": format_inr(result.estimated_gain),
        "disclaimer": "Illustrative only; actual returns vary and are not guaranteed.",
    }


@mcp.tool()
def calculate(expression: str) -> dict[str, Any]:
    """Safely evaluate a math expression.

    Supports + - * / % ** (or ^), parentheses, pi, e and sqrt, sin, cos, tan, log,
    log10, exp, abs, round, floor, ceil, factorial (trig in radians).

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
