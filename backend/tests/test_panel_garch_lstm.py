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
    omega, alpha, beta = 2e-7, 0.08, 0.88
    var = np.empty(n)
    var[0] = omega / (1 - alpha - beta)
    rets = np.empty(n)
    for i in range(1, n):
        sigma2 = omega + alpha * rets[i - 1] ** 2 + beta * var[i - 1]
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
    raw = model.predict(
        {"window": data["windows"][:32], "econometric": data["features"][:32]},
        verbose=0,
    )
    # The head emits LOG-variance; positivity is guaranteed after exp().
    assert raw.shape == (32, horizon)
    assert np.isfinite(raw).all()
    variance = np.exp(np.clip(raw, -30.0, 30.0))
    assert (variance > 0).all()


@pytest.fixture(scope="module")
def trained():
    candidate = GarchLstmCandidate(horizon=5, train_end=1000, epochs=8)
    return candidate.fit_returns(garch_returns(1400))


def test_training_decreases_qlike(trained) -> None:
    losses = trained.history.history["loss"]
    assert len(losses) >= 2
    assert losses[-1] < losses[0]


def test_predictions_beat_constant_baseline_on_qlike(trained) -> None:
    horizon = trained.horizon
    returns = garch_returns(1400, seed=13)
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
    # On true-GARCH synthetic data the econometric branch carries real signal:
    # the hybrid must beat the constant outright out of sample.
    assert model_ql < baseline_ql
    # The hybrid must not be materially worse than the constant; on clustered
    # synthetic data it should track regimes and beat it outright.
    assert model_ql <= baseline_ql * 1.05


def test_predict_fails_closed_before_fit() -> None:
    with pytest.raises(RuntimeError, match="before fit"):
        GarchLstmCandidate(horizon=5, train_end=500).predict(garch_returns(200))
