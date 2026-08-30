"""Unit tests for EnrichedFeatureExtractor."""

import numpy as np
import pandas as pd

from research.volatility_forecasting.multimodal_features_v2 import (
    EnrichedFeatureExtractor,
    EnrichedMarketFeatures,
)


def test_enriched_feature_extraction_completeness_and_finite():
    dates = pd.date_range("2024-01-01", periods=100, freq="B").strftime("%Y-%m-%d")
    rng = np.random.default_rng(42)

    c = np.exp(np.cumsum(rng.normal(0.0005, 0.015, size=100))) * 100.0
    h = c * (1.0 + rng.uniform(0.002, 0.02, size=100))
    low_p = c * (1.0 - rng.uniform(0.002, 0.02, size=100))
    o = (h + low_p) / 2.0
    v = rng.uniform(1e6, 5e6, size=100)

    target_df = pd.DataFrame(
        {"Open": o, "High": h, "Low": low_p, "Close": c, "Volume": v}, index=dates
    )
    sector_df = target_df.copy()
    market_df = target_df.copy()

    feats = EnrichedFeatureExtractor.extract_from_series(target_df, sector_df, market_df)
    arr = feats.to_array()
    assert len(arr) == len(EnrichedMarketFeatures.feature_names())
    assert np.isfinite(arr).all()
    assert feats.drawdown_20d <= 0.0  # Drawdown is non-positive
