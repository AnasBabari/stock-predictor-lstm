"""Exchange-aware future trading-date generation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pandas_market_calendars as mcal

SUFFIX_CALENDARS = {
    ".L": "LSE",
    ".SW": "SIX",
    ".TO": "TSX",
    ".AX": "ASX",
    ".HK": "HKEX",
}


def latest_completed_trading_session(now: datetime | pd.Timestamp | None = None) -> pd.Timestamp:
    """Return the latest NYSE session whose regular close has passed."""
    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    instant = instant.tz_localize(UTC) if instant.tzinfo is None else instant.tz_convert(UTC)
    ny_now = instant.tz_convert(ZoneInfo("America/New_York"))
    calendar = mcal.get_calendar("NYSE")
    schedule = calendar.schedule(
        start_date=(ny_now - timedelta(days=14)).date(),
        end_date=ny_now.date(),
    )
    completed = schedule.loc[schedule["market_close"] <= instant]
    if completed.empty:
        raise ValueError("Could not determine the latest completed NYSE session.")
    return pd.Timestamp(completed.index[-1]).tz_localize(None).normalize()


def resolve_calendar(ticker: str) -> tuple[str, str]:
    """Return (calendar implementation, public identifier)."""
    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError("Ticker must be a non-empty string.")
    upper = ticker.upper()
    if upper.endswith(("-USD", "-GBP", "-EUR", "-USDT")):
        return "24/7", "CRYPTO_24_7"
    for suffix, calendar in SUFFIX_CALENDARS.items():
        if upper.endswith(suffix):
            return calendar, calendar
    if "." in upper:
        return "NYSE", "NYSE_FALLBACK"
    return "NYSE", "NYSE"


def future_trading_dates(
    ticker: str, current_date: pd.Timestamp, days: int
) -> tuple[list[str], str]:
    """Generate exactly ``days`` future dates for the instrument calendar."""
    if days < 1:
        raise ValueError("Forecast days must be positive.")

    calendar_name, identifier = resolve_calendar(ticker)
    current = pd.Timestamp(current_date).tz_localize(None).normalize()
    if calendar_name == "24/7":
        dates = pd.date_range(current + timedelta(days=1), periods=days, freq="D")
        return dates.strftime("%Y-%m-%d").tolist(), identifier

    try:
        calendar = mcal.get_calendar(calendar_name)
    except Exception:
        calendar = mcal.get_calendar("NYSE")
        identifier = "NYSE_FALLBACK"

    # Widen the search horizon if a holiday-heavy period yields too few sessions.
    future: list[pd.Timestamp] = []
    horizon = max(days * 4 + 14, 45)
    for _ in range(3):
        schedule = calendar.schedule(
            start_date=current + timedelta(days=1), end_date=current + timedelta(days=horizon)
        )
        future = [d for d in schedule.index if d > current][:days]
        if len(future) == days:
            break
        horizon *= 2
    if len(future) != days:
        raise ValueError(f"Could not generate {days} dates for calendar {identifier}.")
    if any(a >= b for a, b in zip(future, future[1:], strict=False)):
        raise ValueError(f"Calendar {identifier} produced non-chronological session dates.")
    future_dates = [d.strftime("%Y-%m-%d") for d in future]
    return future_dates, identifier
