"""Exchange session calendar computing exact causal cutoff UTC timestamps (handling EDT, EST, and early closes)."""

from __future__ import annotations

import datetime
from zoneinfo import ZoneInfo

# America/New_York timezone handles Daylight Saving Time transitions automatically
NY_TZ = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")

# US Early close dates (13:00 ET close)
EARLY_CLOSE_DATES: set[str] = {
    "2021-11-26",
    "2022-11-25",
    "2023-07-03",
    "2023-11-24",
    "2024-07-03",
    "2024-11-29",
    "2024-12-24",
    "2025-07-03",
    "2025-11-28",
    "2025-12-24",
    "2026-11-27",
    "2026-12-24",
}


def get_session_close_utc(date_str: str) -> str:
    """Computes exact session close UTC ISO-8601 timestamp for a given US trading session.

    Regular close: 16:00 ET (20:00 UTC in EDT, 21:00 UTC in EST).
    Early close: 13:00 ET (17:00 UTC in EDT, 18:00 UTC in EST).
    """
    year, month, day = map(int, date_str.split("-"))
    close_hour = 13 if date_str in EARLY_CLOSE_DATES else 16
    close_minute = 0

    et_dt = datetime.datetime(year, month, day, close_hour, close_minute, 0, tzinfo=NY_TZ)
    utc_dt = et_dt.astimezone(UTC_TZ)
    return utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
