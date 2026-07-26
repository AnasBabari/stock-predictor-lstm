"""Exchange-aware future trading-date generation."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pandas_market_calendars as mcal

SUFFIX_CALENDARS = {
    ".L": "LSE",
    ".SW": "SIX",
    ".TO": "TSX",
    ".AX": "ASX",
    ".HK": "HKEX",
}


def resolve_calendar(ticker: str) -> tuple[str, str]:
    """Return (calendar implementation, public identifier)."""
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

    end = current + timedelta(days=max(days * 4 + 14, 45))
    schedule = calendar.schedule(start_date=current + timedelta(days=1), end_date=end)
    future = [d.strftime("%Y-%m-%d") for d in schedule.index if d > current][:days]
    if len(future) != days:
        raise ValueError(f"Could not generate {days} dates for calendar {identifier}.")
    return future, identifier
