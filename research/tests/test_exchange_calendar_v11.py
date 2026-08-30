"""Unit tests for hardened exchange_calendar_v11."""

import pytest

from research.volatility_forecasting.exchange_calendar_v11 import (
    get_session_close_utc,
)


def test_exchange_calendar_validations():
    # 1. Active trading session in summer (EDT)
    assert get_session_close_utc("2026-08-28") == "2026-08-28T20:00:00Z"

    # 2. Active trading session in winter (EST)
    assert get_session_close_utc("2026-01-15") == "2026-01-15T21:00:00Z"

    # 3. Early close session (Black Friday 2024, EST) -> 13:00 ET = 18:00 UTC
    assert get_session_close_utc("2024-11-29") == "2024-11-29T18:00:00Z"

    # 4. Reject Weekend
    with pytest.raises(ValueError, match="weekend"):
        get_session_close_utc("2026-08-29")  # Saturday

    # 5. Reject Labor Day Holiday
    with pytest.raises(ValueError, match="holiday"):
        get_session_close_utc("2026-09-07")  # US Labor Day
