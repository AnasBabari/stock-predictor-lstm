"""Unit tests for DevelopmentForecastRunner."""

import math

import numpy as np
import pytest

from research.volatility_forecasting.causal_dataset_v1 import (
    STATIONARY_FEATURE_COLUMNS_V1,
)
from scripts.forecast_development import DevelopmentForecastRunner


def test_development_forecast_anchors_exactly_at_p0():
    runner = DevelopmentForecastRunner()
    rng = np.random.default_rng(123)
    feats = rng.normal(0, 0.01, size=(60, len(STATIONARY_FEATURE_COLUMNS_V1)))

    p0 = 42.15
    out = runner.run_development_forecast(
        ticker="BP",
        base_date="2026-08-28",
        base_price=p0,
        recent_feature_window=feats,
        daily_volatility=0.018,
    )

    assert out.ticker == "BP"
    assert out.base_price == 42.15
    assert out.is_certified_production_claim is False
    assert out.status == "development_diagnostic_only"
    assert len(out.median_prices) == 7
    assert len(out.intervals_80pct) == 7

    # Verify that day 1 price is within reasonable bounds of P0
    p1 = out.median_prices[0]
    ret_1 = math.log(p1 / p0)
    assert abs(ret_1) < 0.05, f"Anchored return deviated excessively: {ret_1}"


def test_development_forecast_rejects_invalid_inputs():
    runner = DevelopmentForecastRunner()
    bad_feats = np.zeros((30, 5))
    with pytest.raises(ValueError, match="Feature window must have shape"):
        runner.run_development_forecast("BP", "2026-08-28", 42.15, bad_feats)

    good_feats = np.zeros((60, len(STATIONARY_FEATURE_COLUMNS_V1)))
    with pytest.raises(ValueError, match="Base price P0 must be strictly positive"):
        runner.run_development_forecast("BP", "2026-08-28", -10.0, good_feats)
