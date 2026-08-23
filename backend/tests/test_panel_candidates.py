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
        "momentum_mean_reversion",
        "volatility_scaled_drift",
        "hist_gradient_boost_global",
        "cross_sectional_momentum",
    ):
        assert name in REGISTRY


def test_new_candidate_models_train_and_predict() -> None:
    from panel.candidates import (
        CrossSectionalMomentumCandidate,
        HistGradientBoostCandidate,
        MomentumMeanReversionCandidate,
        VolatilityScaledDriftCandidate,
    )

    x, y = make_pooled(n_tickers=4, rows=50, window=20, features=4)
    split = int(len(y) * 0.8)

    # Momentum & Mean-Reversion
    mmr = MomentumMeanReversionCandidate().fit(x[:split], y[:split])
    p_mmr = mmr.predict(x[split:])
    assert p_mmr.return_point is not None
    assert np.isfinite(p_mmr.return_point).all()

    # Volatility Scaled Drift
    vsd = VolatilityScaledDriftCandidate().fit(x[:split], y[:split])
    p_vsd = vsd.predict(x[split:])
    assert p_vsd.return_point is not None
    assert np.isfinite(p_vsd.return_point).all()

    # Hist Gradient Boost
    hgb = HistGradientBoostCandidate(max_iter=10).fit(x[:split], y[:split])
    p_hgb = hgb.predict(x[split:])
    assert p_hgb.return_point is not None
    assert np.isfinite(p_hgb.return_point).all()

    # Cross-Sectional Momentum
    csm = CrossSectionalMomentumCandidate().fit(x[:split], y[:split])
    p_csm = csm.predict(x[split:])
    assert p_csm.return_point is not None
    assert np.isfinite(p_csm.return_point).all()


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


def test_missing_direction_labels_raises_error_when_direction_task_enabled() -> None:
    from panel.candidates import CandidateTargets, GlobalRecurrentCandidate

    x, y = make_pooled(rows=20, n_tickers=2)
    model = GlobalRecurrentCandidate(lookback=20, epochs=1, tasks=("returns", "direction"))
    # Targets without direction_classes must fail closed
    targets = CandidateTargets(cumulative_returns=y)
    with pytest.raises(ValueError, match="requires direction_classes in targets"):
        model.fit(x, targets)


def test_direction_probabilities_sum_to_one_and_quantiles_are_monotonic() -> None:
    from panel.candidates import CandidateTargets, GlobalRecurrentCandidate

    x, y = make_pooled(rows=20, n_tickers=2)
    dir_classes = (y > 0.05).astype(int) + (y < -0.05).astype(int) * 0  # 0, 1, 2
    # Ensure all three classes present
    dir_classes[0] = 0
    dir_classes[1] = 1
    dir_classes[2] = 2
    targets = CandidateTargets(cumulative_returns=y, direction_classes=dir_classes)

    model = GlobalRecurrentCandidate(lookback=20, epochs=2, tasks=("returns", "direction"), seed=42)
    model.fit(x, targets)
    pred = model.predict(x[:10])

    assert pred.direction_probabilities is not None
    assert pred.direction_probabilities.shape == (10, 3)
    assert np.isfinite(pred.direction_probabilities).all()
    np.testing.assert_allclose(pred.direction_probabilities.sum(axis=1), np.ones(10), atol=1e-5)

    assert pred.return_quantiles is not None
    q10 = pred.return_quantiles["0.1"]
    q50 = pred.return_quantiles["0.5"]
    q90 = pred.return_quantiles["0.9"]
    # Quantiles must be monotonic: q10 <= q50 <= q90
    assert (q10 <= q50 + 1e-6).all()
    assert (q50 <= q90 + 1e-6).all()


def test_deterministic_seed_produces_identical_predictions() -> None:
    from panel.candidates import CandidateTargets, GlobalRecurrentCandidate

    x, y = make_pooled(rows=20, n_tickers=2, seed=10)
    dir_classes = np.ones(len(y), dtype=int)
    targets = CandidateTargets(cumulative_returns=y, direction_classes=dir_classes)

    m1 = GlobalRecurrentCandidate(lookback=20, epochs=2, seed=77)
    m1.fit(x, targets)
    p1 = m1.predict(x)

    m2 = GlobalRecurrentCandidate(lookback=20, epochs=2, seed=77)
    m2.fit(x, targets)
    p2 = m2.predict(x)

    np.testing.assert_allclose(p1.return_point, p2.return_point, atol=1e-5)
    np.testing.assert_allclose(p1.direction_probabilities, p2.direction_probabilities, atol=1e-5)


def test_tcn_gru_lstm_neural_candidates_train_and_compile_losses() -> None:
    from panel.candidates import (
        CandidateTargets,
        GlobalRecurrentCandidate,
        TemporalConvolutionCandidate,
    )

    x, y = make_pooled(rows=30, n_tickers=2, window=20, features=4, seed=5)
    dir_classes = np.ones(len(y), dtype=int)
    targets = CandidateTargets(cumulative_returns=y, direction_classes=dir_classes)

    # Test TCN
    tcn = TemporalConvolutionCandidate(lookback=20, epochs=2, seed=1)
    tcn.fit(x, targets)
    pred_tcn = tcn.predict(x[:5])
    assert pred_tcn.return_point is not None
    assert pred_tcn.direction_probabilities is not None
    assert pred_tcn.return_quantiles is not None
    assert tcn.diagnostics["completed_epochs"] == 2
    tcn.dispose()
    assert tcn._model is None

    # Test GRU with inner validation
    gru = GlobalRecurrentCandidate(
        architecture="gru", lookback=20, epochs=2, inner_val_split=0.2, seed=2
    )
    gru.fit(x, targets)
    pred_gru = gru.predict(x[:5])
    assert pred_gru.return_point is not None
    gru.dispose()


def test_garch_lstm_global_candidate_trains_and_predicts() -> None:
    from panel.candidates import CandidateTargets, GarchLstmGlobalCandidate

    rng = np.random.default_rng(123)
    rets = rng.normal(0, 0.015, 300)
    candidate = GarchLstmGlobalCandidate(horizon=5, lookback=20, epochs=2, seed=42)
    candidate.fit(rets, CandidateTargets(cumulative_returns=rets))
    pred = candidate.predict(rets)
    assert pred.variance_forecast is not None
    assert (pred.variance_forecast > 0).all()
    candidate.dispose()
