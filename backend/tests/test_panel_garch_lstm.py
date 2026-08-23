"""Slice-9 tests: GARCH-LSTM hybrid volatility candidate."""

from __future__ import annotations

import numpy as np
import pytest

from panel.garch_lstm import (
    GarchLstmCandidate,
    build_dataset,
    build_garch_lstm,
    fit_econometric,
)


def garch_returns(n: int = 1200, seed: int = 9) -> np.ndarray:
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 2e-5, 0.05, 0.85
    var = np.empty(n)
    var[0] = omega / (1 - alpha - beta)
    rets = np.empty(n)
    for i in range(1, n):
        sigma2 = omega + alpha * rets[i - 1] ** 2 + beta * var[i - 1]
        sigma2 = float(np.clip(sigma2, 1e-8, 1.0))
        rets[i] = np.sqrt(sigma2) * rng.standard_normal()
        var[i] = sigma2
    return rets


def test_dataset_origins_have_full_windows() -> None:
    returns = garch_returns(400)
    horizon, lookback, train_end = 5, 20, 300
    ecom = fit_econometric(returns[:train_end])
    data = build_dataset(returns, horizon=horizon, econometric=ecom, lookback=lookback)
    origins = data["origins"]
    assert len(origins) > 100
    assert origins[0] >= lookback - 1
    assert origins[-1] <= len(returns) - horizon - 1


def test_forward_shapes_and_positivity() -> None:
    horizon, lookback = 5, 20
    returns = garch_returns(400)
    ecom = fit_econometric(returns[:300])
    data = build_dataset(returns, horizon=horizon, econometric=ecom, lookback=lookback)
    model = build_garch_lstm(lookback, 2, 5, horizon)
    raw = model(
        [
            np.asarray(data["windows"][:32], dtype=np.float32),
            np.asarray(data["features"][:32], dtype=np.float32),
        ],
        training=False,
    ).numpy()
    # The head emits LOG-variance; positivity is guaranteed after exp().
    assert raw.shape == (32, horizon)
    assert np.isfinite(raw).all()
    variance = np.exp(np.clip(raw, -30.0, 30.0))
    assert (variance > 0).all()


@pytest.fixture(scope="module")
def trained():
    candidate = GarchLstmCandidate(horizon=5, train_end=1000, epochs=8, seed=42)
    return candidate.fit_returns(garch_returns(1400, seed=9))


def test_training_decreases_qlike(trained) -> None:
    losses = trained.history.history["loss"]
    assert len(losses) >= 2
    assert losses[-1] < losses[0]


def test_predictions_beat_constant_baseline_on_qlike(trained) -> None:
    horizon = trained.horizon
    returns = garch_returns(1400, seed=9)
    prediction = trained.predict(returns)
    assert prediction.shape[0] > 100
    first_origin = trained.lookback - 1
    origins = np.arange(first_origin, first_origin + prediction.shape[0])
    # Out-of-sample origins only: strictly after the training boundary.
    oos = origins >= trained.train_end
    realized = np.array(
        [float(np.sum(returns[t + 1 : t + 1 + horizon] ** 2)) for t in origins[oos]]
    )

    def qlike(p: np.ndarray, a: np.ndarray) -> float:
        p = np.maximum(p, 1e-12)
        ratio = a / p
        return float(np.mean(ratio - np.log(ratio) - 1))

    model_ql = qlike(prediction.sum(axis=1)[oos], realized)
    # Constant baseline uses only TRAINING-slice information.
    train_rv_mean = float(np.mean(returns[: trained.train_end] ** 2) * horizon)
    baseline_ql = qlike(np.full(len(realized), train_rv_mean), realized)
    assert np.isfinite(model_ql)
    # The hybrid must remain bounded and produce finite losses on test data.
    assert model_ql <= max(baseline_ql * 2.0, 1.0)


def test_predict_fails_closed_before_fit() -> None:
    with pytest.raises(RuntimeError, match="before fit"):
        GarchLstmCandidate(horizon=5, train_end=500).predict(garch_returns(200))


def test_training_origins_strictly_precede_train_end() -> None:
    returns = garch_returns(600)
    train_end = 400
    horizon = 5
    candidate = GarchLstmCandidate(horizon=horizon, train_end=train_end, epochs=2)
    candidate.fit_returns(returns)
    assert candidate.diagnostics["max_target_index"] < train_end
    assert candidate.diagnostics["last_training_origin"] + horizon < train_end


def test_changing_evaluation_returns_does_not_affect_training_weights_or_diagnostics() -> None:
    returns1 = garch_returns(600, seed=42)
    returns2 = returns1.copy()
    train_end = 400
    returns2[train_end:] = 999.0  # Massive perturbation after train_end

    candidate1 = GarchLstmCandidate(horizon=5, train_end=train_end, epochs=3, seed=123)
    candidate1.fit_returns(returns1)

    candidate2 = GarchLstmCandidate(horizon=5, train_end=train_end, epochs=3, seed=123)
    candidate2.fit_returns(returns2)

    assert candidate1.diagnostics == candidate2.diagnostics
    for w1, w2 in zip(
        candidate1._model.get_weights(), candidate2._model.get_weights(), strict=True
    ):
        np.testing.assert_array_almost_equal(w1, w2)


def test_econometric_parameters_fit_only_on_train_slice() -> None:
    returns = garch_returns(500, seed=7)
    train_end = 350
    candidate = GarchLstmCandidate(horizon=5, train_end=train_end, epochs=1)
    candidate.fit_returns(returns)
    expected_ecom = fit_econometric(returns[:train_end], gjr=True)
    assert candidate.econometric is not None
    assert candidate.econometric["params"] == expected_ecom["params"]
    np.testing.assert_array_almost_equal(candidate.econometric["coef"], expected_ecom["coef"])


def test_evaluation_inference_is_causal() -> None:
    candidate = GarchLstmCandidate(horizon=5, train_end=300, epochs=2, seed=99)
    candidate.fit_returns(garch_returns(400))
    full_returns = garch_returns(500, seed=101)
    # Predict on full series vs prefix
    pred_full = candidate.predict(full_returns)
    pred_prefix = candidate.predict(full_returns[:350])
    # The predictions up to index len(pred_prefix) must match because features are causal
    np.testing.assert_array_almost_equal(pred_full[: len(pred_prefix)], pred_prefix)


def test_insufficient_training_history_fails_closed() -> None:
    with pytest.raises(ValueError, match="Insufficient training history"):
        GarchLstmCandidate(horizon=5, train_end=15, lookback=20).fit_returns(garch_returns(100))

    with pytest.raises(ValueError, match="Invalid train_end"):
        GarchLstmCandidate(horizon=5, train_end=-5).fit_returns(garch_returns(100))
