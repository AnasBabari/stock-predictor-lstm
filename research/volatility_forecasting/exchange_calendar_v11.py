"""Exchange session calendar with explicit holiday rejection and exact UTC market-close resolution."""

from __future__ import annotations

import datetime
from zoneinfo import ZoneInfo

NY_TZ = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")

# Fixed & Dynamic US Market Holidays (XNAS / XNYS)
US_MARKET_HOLIDAYS: set[str] = {
    # 2021
    "2021-01-01",
    "2021-01-18",
    "2021-02-15",
    "2021-04-02",
    "2021-05-31",
    "2021-07-05",
    "2021-09-06",
    "2021-11-25",
    "2021-12-24",
    # 2022
    "2022-01-17",
    "2022-02-21",
    "2022-04-15",
    "2022-05-30",
    "2022-06-20",
    "2022-07-04",
    "2022-09-05",
    "2022-11-24",
    "2022-12-26",
    # 2023
    "2023-01-02",
    "2023-01-16",
    "2023-02-20",
    "2023-04-07",
    "2023-05-29",
    "2023-06-19",
    "2023-07-04",
    "2023-09-04",
    "2023-11-23",
    "2023-12-25",
    # 2024
    "2024-01-01",
    "2024-01-15",
    "2024-02-19",
    "2024-03-29",
    "2024-05-27",
    "2024-06-19",
    "2024-07-04",
    "2024-09-02",
    "2024-11-28",
    "2024-12-25",
    # 2025
    "2025-01-01",
    "2025-01-20",
    "2025-02-17",
    "2025-04-18",
    "2025-05-26",
    "2025-06-19",
    "2025-07-04",
    "2025-09-01",
    "2025-11-27",
    "2025-12-25",
    # 2026
    "2026-01-01",
    "2026-01-19",
    "2026-02-16",
    "2026-04-03",
    "2026-05-25",
    "2026-06-19",
    "2026-07-03",
    "2026-09-07",
    "2026-11-26",
    "2026-12-25",
}

US_EARLY_CLOSE_DATES: set[str] = {
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


def get_session_close_utc(session_date: str, exchange_mic: str = "XNAS") -> str:
    """Returns exact UTC close timestamp for session_date on exchange_mic.

    Rejects non-trading dates (weekends, holidays) with ValueError.
    """
    year, month, day = map(int, session_date.split("-"))
    dt_obj = datetime.date(year, month, day)

    # Reject weekends (5=Saturday, 6=Sunday)
    if dt_obj.weekday() >= 5:
        raise ValueError(
            f"Date {session_date} is a weekend, not an active exchange trading session."
        )

    # Reject official exchange holidays
    if session_date in US_MARKET_HOLIDAYS:
        raise ValueError(f"Date {session_date} is an official exchange holiday on {exchange_mic}.")

    close_hour = 13 if session_date in US_EARLY_CLOSE_DATES else 16
    et_dt = datetime.datetime(year, month, day, close_hour, 0, 0, tzinfo=NY_TZ)
    utc_dt = et_dt.astimezone(UTC_TZ)
    return utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
