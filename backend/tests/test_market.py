from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from features.market import MarketContextUnavailable, add_market_context
from model import _unscale_close


def test_add_market_context_success():
    dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
    df = pd.DataFrame({"Close": np.random.randn(10)}, index=dates)

    market_dates = dates.insert(0, dates[0] - pd.Timedelta(days=1))
    mock_m_data = pd.DataFrame(
        {"Close": np.linspace(99, 110, 11)},
        index=market_dates,
    )

    with patch("yfinance.download", return_value=mock_m_data):
        result, metadata = add_market_context(df)

    assert "SPY_Return_1D" in result.columns
    assert "QQQ_Return_1D" in result.columns
    assert "VIX_Return_1D" in result.columns
    assert "TNX_Return_1D" in result.columns
    assert len(result) == 10
    assert metadata["status"] == "complete"
    assert metadata["schema_version"] == 2


def test_add_market_context_empty_download():
    dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
    df = pd.DataFrame({"Close": np.random.randn(10)}, index=dates)

    empty_df = pd.DataFrame()

    with (
        patch("yfinance.download", return_value=empty_df),
        pytest.raises(MarketContextUnavailable, match="SPY"),
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
        pytest.raises(MarketContextUnavailable, match="closing prices"),
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
