from datetime import UTC, datetime

import pandas as pd
import pytest

from calendars import future_trading_dates, latest_completed_trading_session, resolve_calendar


def test_latest_completed_session_lse():
    instant = datetime(2026, 9, 4, 18, 0, tzinfo=UTC)
    session = latest_completed_trading_session(instant, exchange="LSE")
    assert session == pd.Timestamp("2026-09-04")

    # Early in the morning before LSE close (10:00 UTC)
    morning = datetime(2026, 9, 4, 10, 0, tzinfo=UTC)
    session_prev = latest_completed_trading_session(morning, exchange="LSE")
    assert session_prev == pd.Timestamp("2026-09-03")


def test_exchange_resolution():
    assert resolve_calendar("AAPL") == ("NYSE", "NYSE")
    assert resolve_calendar("VOD.L") == ("LSE", "LSE")
    assert resolve_calendar("NESN.SW") == ("SIX", "SIX")
    assert resolve_calendar("BTC-USD") == ("24/7", "CRYPTO_24_7")
    assert resolve_calendar("UNKNOWN.XY") == ("NYSE", "NYSE_FALLBACK")


def test_lse_boxing_day_and_six_national_day():
    lse_dates, lse_id = future_trading_dates("VOD.L", pd.Timestamp("2025-12-24"), 2)
    six_dates, six_id = future_trading_dates("NESN.SW", pd.Timestamp("2025-07-31"), 2)
    assert lse_id == "LSE"
    assert lse_dates == ["2025-12-29", "2025-12-30"]
    assert six_id == "SIX"
    assert "2025-08-01" not in six_dates


def test_crypto_calendar_is_always_open():
    dates, calendar_id = future_trading_dates("BTC-USD", pd.Timestamp("2025-12-26"), 3)
    assert calendar_id == "CRYPTO_24_7"
    assert dates == ["2025-12-27", "2025-12-28", "2025-12-29"]


def test_resolve_calendar_rejects_empty_ticker():
    with pytest.raises(ValueError, match="non-empty"):
        resolve_calendar("")
    with pytest.raises(ValueError, match="non-empty"):
        resolve_calendar("   ")


def test_future_trading_dates_widens_horizon_for_sparse_calendars(monkeypatch):
    class GrowingCalendar:
        def __init__(self):
            self.calls = 0

        def schedule(self, start_date, end_date):
            self.calls += 1
            if self.calls == 1:
                return pd.DataFrame(index=pd.DatetimeIndex([start_date]))
            return pd.DataFrame(index=pd.bdate_range(start_date, end_date))

    growing = GrowingCalendar()
    monkeypatch.setattr("calendars.mcal.get_calendar", lambda name: growing)

    dates, _ = future_trading_dates("VOD.L", pd.Timestamp("2026-01-05"), 5)
    assert growing.calls > 1
    assert len(dates) == 5
    assert all(str(d) == d for d in dates)
    assert all(d > "2026-01-05" for d in dates)


def test_future_trading_dates_rejects_non_chronological_schedule(monkeypatch):
    class TiedCalendar:
        def schedule(self, start_date, end_date):
            return pd.DataFrame(index=pd.DatetimeIndex([start_date, start_date]))

    monkeypatch.setattr("calendars.mcal.get_calendar", lambda name: TiedCalendar())

    with pytest.raises(ValueError, match="non-chronological"):
        future_trading_dates("VOD.L", pd.Timestamp("2026-01-05"), 2)


def test_future_trading_dates_rejects_empty_schedule(monkeypatch):
    class EmptyCalendar:
        def schedule(self, start_date, end_date):
            return pd.DataFrame(index=pd.DatetimeIndex([]))

    monkeypatch.setattr("calendars.mcal.get_calendar", lambda name: EmptyCalendar())

    with pytest.raises(ValueError, match="Could not generate"):
        future_trading_dates("VOD.L", pd.Timestamp("2026-01-05"), 2)
