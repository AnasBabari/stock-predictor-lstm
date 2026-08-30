"""Unit tests for HistoricalPITDatasetBuilderV11."""

import numpy as np
import pandas as pd

from research.volatility_forecasting.historical_pit_dataset_builder_v11 import (
    HistoricalPITDatasetBuilderV11,
)


def test_historical_pit_panel_construction_and_causality():
    n_days = 100
    dates = pd.date_range("2022-01-01", periods=n_days, freq="B")
    rng = np.random.default_rng(42)

    def create_synthetic_df():
        c = np.exp(np.cumsum(rng.normal(0.0005, 0.015, size=n_days))) * 100.0
        h = c * 1.01
        low_p = c * 0.99
        o = c
        v = rng.uniform(1e6, 5e6, size=n_days)
        return pd.DataFrame(
            {"Open": o, "High": h, "Low": low_p, "Close": c, "Volume": v}, index=dates
        )

    equities = {
        "AAPL": create_synthetic_df(),
        "MSFT": create_synthetic_df(),
        "AMGN": create_synthetic_df(),
    }
    sector_df = create_synthetic_df()
    market_df = create_synthetic_df()

    panel = HistoricalPITDatasetBuilderV11.construct_panel_from_series(
        equities_ohlcv=equities,
        sector_ohlcv=sector_df,
        market_ohlcv=market_df,
        horizons=(1, 3, 5, 7),
        warmup_sessions=65,
    )

    # Verify output shapes and dimensions
    assert len(panel.dates) > 50
    assert panel.numeric_features.shape[1] == 34
    assert panel.news_features.shape[1] == 19
    assert panel.returns_targets.shape[1] == 4
    assert panel.rv_targets.shape[1] == 4
    assert len(panel.panel_sha256) == 64

    # Verify chronological ordering
    for i in range(len(panel.dates) - 1):
        assert panel.dates[i] <= panel.dates[i + 1]
