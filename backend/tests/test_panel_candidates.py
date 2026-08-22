"""Slice-8 tests: global candidates on synthetic pooled panels."""

from __future__ import annotations

import numpy as np
import pytest

from panel.candidates import REGISTRY, ElasticNetCandidate, RidgeCandidate, RollingMeanCandidate


def make_pooled(
    n_tickers: int = 6, rows: int = 120, window: int = 20, features: int = 4, seed: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Learnable panel: y depends linearly on the last window row."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (n_tickers * rows, window, features))
    weights = np.array([0.8, -0.5, 0.3, 0.1])
    y = x[:, -1, :] @ weights + rng.normal(0, 0.05, n_tickers * rows)
    return x.astype(np.float32), y.astype(np.float32)


def relative_mae(model) -> float:
    x, y = make_pooled()
    split = int(len(y) * 0.8)
    model.fit(x[:split], y[:split])
    prediction = model.predict(x[split:])
    mae = np.mean(np.abs(prediction.point - y[split:]))
    baseline = np.mean(np.abs(y[split:]))  # persistence = 0 forecast
    return float(mae / baseline)


def test_registry_contains_statistical_families() -> None:
    for name in (
        "persistence",
        "rolling_mean_shrunk",
        "ridge_global",
        "elastic_net_global",
        "dlinear_global",
    ):
        assert name in REGISTRY


def test_ridge_beats_persistence_on_learnable_panel() -> None:
    assert relative_mae(RidgeCandidate(alpha=1.0)) < 0.9


def test_elastic_net_beats_persistence_on_learnable_panel() -> None:
    assert relative_mae(ElasticNetCandidate()) < 0.95


def test_dlinear_beats_persistence_on_learnable_panel() -> None:
    from panel.candidates import DLinearGlobalCandidate

    assert relative_mae(DLinearGlobalCandidate(kernel=5)) < 0.95


def test_rolling_mean_shrinkage_pulls_toward_zero() -> None:
    x, y = make_pooled()
    model = RollingMeanCandidate(shrinkage=0.5).fit(x, y)
    raw_mean = float(np.mean(y))
    assert abs(model._mean) == pytest.approx(abs(raw_mean) / 2)


def test_persistence_quantiles_are_symmetric_and_flat() -> None:
    from panel.candidates import PersistenceCandidate

    out = PersistenceCandidate().predict(np.zeros((5, 20, 4)))
    np.testing.assert_array_equal(out.quantiles["0.5"], np.zeros(5))
    np.testing.assert_allclose(out.quantiles["0.1"], -out.quantiles["0.9"])


def test_prediction_shapes_match_rows() -> None:
    x, _ = make_pooled(rows=40)
    model = RidgeCandidate(alpha=1.0).fit(x[:80], np.zeros(80))
    out = model.predict(x[80:])
    assert out.point.shape == (len(x) - 80,)
