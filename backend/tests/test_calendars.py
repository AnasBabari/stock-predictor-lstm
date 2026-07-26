import pandas as pd

from calendars import future_trading_dates, resolve_calendar


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
