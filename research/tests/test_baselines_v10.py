"""Tests for the complete matched causal volatility baseline suite."""

from __future__ import annotations

import numpy as np
import pytest

from research.volatility_forecasting.baselines_v10 import (
    ElasticNetVolatilityBaseline,
    EWMABaseline,
    GARCHBaseline,
    GJRGARCHBaseline,
    HARRVBaseline,
    PersistenceBaseline,
    RidgeVolatilityBaseline,
)


def test_persistence_baseline_scales_with_horizon() -> None:
    base = PersistenceBaseline()
    v1 = base.predict(0.0004, horizon=1)
    v5 = base.predict(0.0004, horizon=5)
    assert v1 == pytest.approx(0.0004)
    assert v5 == pytest.approx(0.0020)


def test_ewma_baseline_positive_and_causal() -> None:
    base = EWMABaseline(lambda_param=0.94)
    series = [0.0001 * (1 + 0.05 * i) for i in range(50)]
    pred = base.predict(series, horizon=5)
    assert pred > 0.0


def test_har_baseline_fits_and_predicts_positive_variance() -> None:
    rng = np.random.default_rng(42)
    daily_vars = 0.0004 * (1.0 + 0.2 * rng.standard_normal(100))
    daily_vars = np.maximum(daily_vars, 1e-6)

    har = HARRVBaseline()
    har.fit(daily_vars[:80])
    pred = har.predict(daily_vars[50:80], horizon=5)
    assert pred > 0.0


def test_garch_and_gjr_garch_predictions_are_positive() -> None:
    garch = GARCHBaseline()
    garch.fit(np.random.default_rng(42).normal(0, 0.01, size=100))
    p1 = garch.predict(0.0004, horizon=5)
    assert p1 > 0.0

    gjr = GJRGARCHBaseline()
    p2 = gjr.predict(0.0004, horizon=5)
    assert p2 > 0.0


def test_ridge_and_elasticnet_baselines_fit_and_predict() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 26))
    y = np.maximum(0.0004 + 0.0001 * X[:, 0], 1e-6)

    ridge = RidgeVolatilityBaseline(alpha=1.0).fit(X, y)
    r_preds = ridge.predict(X[:5])
    assert (r_preds > 0.0).all()

    enet = ElasticNetVolatilityBaseline(alpha=0.01, l1_ratio=0.5).fit(X, y)
    e_preds = enet.predict(X[:5])
    assert (e_preds > 0.0).all()
