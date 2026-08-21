from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from features.market import MARKET_TICKERS, MarketContextUnavailable, add_market_context
from model import _unscale_close


def _batched_download_frame(dates, symbols=None):
    """Build a group_by='ticker' style frame (symbols on the columns axis)."""
    symbols = symbols or list(MARKET_TICKERS)
    frames = []
    for i, _symbol in enumerate(symbols):
        close = pd.DataFrame({"Close": np.linspace(99 + i, 110 + i, len(dates))}, index=dates)
        frames.append(close)
    return pd.concat(frames, keys=symbols, axis=1)


def test_add_market_context_success_single_call():
    dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
    df = pd.DataFrame({"Close": np.random.randn(10)}, index=dates)
    # One prior observation so the first target row has a real return.
    batched = _batched_download_frame(dates.insert(0, dates[0] - pd.Timedelta(days=1)))

    with patch("yfinance.download", return_value=batched) as mock_download:
        result, metadata = add_market_context(df)

    assert mock_download.call_count == 1  # batched, not one call per symbol
    assert "SPY_Return_1D" in result.columns
    assert "QQQ_Return_1D" in result.columns
    assert "VIX_Return_1D" in result.columns
    assert "TNX_Return_1D" in result.columns
    assert len(result) == 10
    assert metadata["status"] == "complete"
    assert metadata["schema_version"] == 2
    assert set(metadata["sources"]) == set(MARKET_TICKERS.values())


def test_add_market_context_missing_symbol_fails_closed():
    dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
    df = pd.DataFrame({"Close": np.random.randn(10)}, index=dates)
    extended = dates.insert(0, dates[0] - pd.Timedelta(days=1))
    partial = _batched_download_frame(extended, symbols=["SPY", "QQQ"])

    with (
        patch("yfinance.download", return_value=partial),
        pytest.raises(MarketContextUnavailable, match="VIX"),
    ):
        add_market_context(df)


def test_add_market_context_empty_download():
    dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
    df = pd.DataFrame({"Close": np.random.randn(10)}, index=dates)

    empty_df = pd.DataFrame()

    with (
        patch("yfinance.download", return_value=empty_df),
        pytest.raises(MarketContextUnavailable),
    ):
        add_market_context(df)


def test_add_market_context_malformed_download():
    dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
    df = pd.DataFrame({"Close": np.random.randn(10)}, index=dates)
    with (
        patch(
            "yfinance.download",
            return_value=pd.DataFrame({"Open": [1.0]}, index=dates[:1]),
        ),
        pytest.raises(MarketContextUnavailable),
    ):
        add_market_context(df)


def test_unscale_close_constant_price_guard():
    scaler = MagicMock()
    scaler.scale_ = np.array([0.0, 1.0, 0.5])
    scaler.data_min_ = np.array([150.0, 10.0, 5.0])

    with patch("model.FEATURES", ["Close", "High", "Low"]):
        scaled = np.array([0.5, 0.5, 0.5])
        unscaled = _unscale_close(scaled, scaler)
        assert (unscaled == 150.0).all()
