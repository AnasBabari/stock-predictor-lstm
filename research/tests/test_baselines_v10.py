"""Tests for matched causal volatility baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.volatility_forecasting.baselines_v10 import (
    CausalEWMABaseline,
    CausalHARBaseline,
    CausalPersistenceBaseline,
)


@pytest.fixture
def sample_rv_series() -> pd.Series:
    np.random.seed(42)
    # Generate log-AR(1) realized variance path
    n = 200
    log_rv = np.zeros(n)
    log_rv[0] = -7.0
    for t in range(1, n):
        log_rv[t] = -7.0 * 0.1 + 0.9 * log_rv[t - 1] + np.random.normal(0, 0.2)
    rv = np.exp(log_rv)
    return pd.Series(rv)


def test_persistence_baseline_scales_with_horizon(sample_rv_series: pd.Series) -> None:
    model = CausalPersistenceBaseline()
    history = sample_rv_series.to_numpy()[:50]
    p1 = model.predict(history, 1)
    p5 = model.predict(history, 5)
    assert np.isclose(p5, p1 * 5.0)
    assert p1 > 0


def test_ewma_baseline_positive_and_causal(sample_rv_series: pd.Series) -> None:
    model = CausalEWMABaseline(decay=0.94)
    history = sample_rv_series.to_numpy()[:50]
    p1 = model.predict(history, 1)
    p3 = model.predict(history, 3)
    assert p1 > 0
    assert p3 > p1


def test_har_baseline_fits_and_predicts_positive_variance(sample_rv_series: pd.Series) -> None:
    model = CausalHARBaseline()
    model.fit(sample_rv_series.iloc[:150], horizons=(1, 3, 5, 7))
    history = sample_rv_series.to_numpy()[:150]
    for h in (1, 3, 5, 7):
        pred = model.predict(history, h)
        assert np.isfinite(pred)
        assert pred > 0
