"""Unit tests for session_calendar_v11."""

from research.volatility_forecasting.session_calendar_v11 import get_session_close_utc


def test_session_close_utc_dst_transitions_and_early_close():
    # 1. Summer session (EDT: UTC-4) -> 16:00 ET is 20:00 UTC
    summer_utc = get_session_close_utc("2026-08-28")
    assert summer_utc == "2026-08-28T20:00:00Z"

    # 2. Winter session (EST: UTC-5) -> 16:00 ET is 21:00 UTC
    winter_utc = get_session_close_utc("2026-01-15")
    assert winter_utc == "2026-01-15T21:00:00Z"

    # 3. Early close session (Black Friday in November, EST: UTC-5) -> 13:00 ET is 18:00 UTC
    early_close_utc = get_session_close_utc("2024-11-29")
    assert early_close_utc == "2024-11-29T18:00:00Z"
