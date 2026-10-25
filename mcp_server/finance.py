"""Pure investment helpers behind the finance MCP tools (no I/O, easy to test)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
PRE_OPEN = time(9, 0)
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)

MAX_YEARS = 50
MAX_AMOUNT = 1_000_000_000  # ₹100 crore; keeps results sane
RETURN_RANGE = (-20.0, 30.0)


@dataclass(frozen=True)
class Projection:
    """Outcome of an illustrative SIP / lump-sum projection."""

    total_invested: float
    estimated_value: float
    estimated_gain: float


def format_inr(amount: float) -> str:
    """Rupees the way Indians say them: ₹23.23 lakh, ₹1.5 crore, ₹45,000."""
    def short(value: float) -> str:
        return f"{value:.2f}".rstrip("0").rstrip(".")

    if amount >= 1e7:
        return f"₹{short(amount / 1e7)} crore"
    if amount >= 1e5:
        return f"₹{short(amount / 1e5)} lakh"
    whole = f"{round(amount):d}"
    head, tail = whole[:-3], whole[-3:]
    groups = [head[max(i - 2, 0):i] for i in range(len(head), 0, -2)][::-1]
    return "₹" + ",".join([*groups, tail]) if head else f"₹{tail}"


def project_investment(
    monthly_sip: float = 0.0,
    lumpsum: float = 0.0,
    annual_return_pct: float = 12.0,
    years: float = 10.0,
    annual_step_up_pct: float = 0.0,
) -> Projection:
    """Project the future value of a monthly SIP and/or a one-time lump sum.

    Uses monthly compounding at ``annual_return_pct / 12``. SIP instalments are
    invested at the start of each month, and the SIP amount rises by
    ``annual_step_up_pct`` at the start of each new year.

    Raises:
        ValueError: on out-of-range inputs.
    """
    if monthly_sip < 0 or lumpsum < 0:
        raise ValueError("amounts cannot be negative")
    if monthly_sip == 0 and lumpsum == 0:
        raise ValueError("provide a monthly_sip, a lumpsum, or both")
    if max(monthly_sip, lumpsum) > MAX_AMOUNT:
        raise ValueError("amount too large for this calculator")
    if not 0 < years <= MAX_YEARS:
        raise ValueError(f"years must be between 0 and {MAX_YEARS}")
    if not RETURN_RANGE[0] <= annual_return_pct <= RETURN_RANGE[1]:
        raise ValueError(f"annual_return_pct must be between {RETURN_RANGE[0]} and {RETURN_RANGE[1]}")
    if not 0 <= annual_step_up_pct <= 50:
        raise ValueError("annual_step_up_pct must be between 0 and 50")

    rate = annual_return_pct / 100 / 12
    months = round(years * 12)
    value, invested, sip = lumpsum, lumpsum, monthly_sip
    for month in range(months):
        if month and month % 12 == 0:
            sip *= 1 + annual_step_up_pct / 100
        value = (value + sip) * (1 + rate)
        invested += sip
    return Projection(
        total_invested=round(invested, 2),
        estimated_value=round(value, 2),
        estimated_gain=round(value - invested, 2),
    )


def market_status(now: datetime | None = None) -> dict[str, str]:
    """Whether the NSE/BSE equity cash market is open, based on regular IST hours.

    Exchange holidays are not known to this function, so on a holiday it may
    report "open"; callers should mention that.
    """
    now = (now or datetime.now(IST)).astimezone(IST)
    weekday = now.weekday() < 5
    t = now.time()
    if weekday and MARKET_OPEN <= t < MARKET_CLOSE:
        status = "open"
    elif weekday and PRE_OPEN <= t < MARKET_OPEN:
        status = "pre-open session"
    else:
        status = "closed"

    next_open = now
    if not weekday or t >= MARKET_OPEN:
        next_open += timedelta(days=1)
    while next_open.weekday() >= 5:
        next_open += timedelta(days=1)
    next_open = next_open.replace(hour=9, minute=15, second=0, microsecond=0)

    return {
        "status": status,
        "now_ist": now.strftime("%A %d %B %Y, %H:%M IST"),
        "regular_hours": "Monday to Friday, 9:15 AM to 3:30 PM IST (pre-open 9:00 to 9:15 AM)",
        "next_open": next_open.strftime("%A %d %B %Y, 9:15 AM IST") if status != "open" else "",
        "note": "Exchange holidays are not included; check the NSE holiday calendar.",
    }
