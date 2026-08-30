"""Unit tests for matched baselines suite."""

import numpy as np

from research.volatility_forecasting.baselines_suite import (
    MatchedDirectionBaselines,
    MatchedReturnBaselines,
    MatchedVolatilityBaselines,
)


def test_matched_return_baselines():
    # Zero return
    res_zero = MatchedReturnBaselines.zero_return(n_samples=50, horizon=5)
    assert len(res_zero.predictions) == 50
    assert (res_zero.predictions == 0.0).all()
    assert res_zero.target_contract == "price-return-distribution-v1"

    # Historical mean
    train_rets = np.array([0.001, -0.0005, 0.002, 0.0005])
    res_mean = MatchedReturnBaselines.historical_mean_return(train_rets, n_samples=30, horizon=7)
    assert len(res_mean.predictions) == 30
    assert (res_mean.predictions > 0.0).all()

    # Ridge baseline
    rng = np.random.default_rng(123)
    X_tr = rng.normal(0, 1, size=(40, 20, 5))
    y_tr = rng.normal(0, 0.02, size=40)
    X_ev = rng.normal(0, 1, size=(10, 20, 5))
    res_ridge = MatchedReturnBaselines.ridge_return(X_tr, y_tr, X_ev, horizon=5)
    assert len(res_ridge.predictions) == 10
    assert np.isfinite(res_ridge.predictions).all()


def test_matched_direction_baselines():
    train_labels = np.array([1, 1, -1, 1, 0, 1])
    res_maj = MatchedDirectionBaselines.majority_class(train_labels, n_samples=25, horizon=3)
    assert len(res_maj.predictions) == 25
    assert (res_maj.predictions == 1).all()

    lagged_rets = np.array([-0.02, 0.01, -0.005, 0.03])
    res_mom = MatchedDirectionBaselines.momentum_sign(lagged_rets, horizon=1)
    assert np.array_equal(res_mom.predictions, [-1, 1, -1, 1])


def test_matched_volatility_baselines():
    daily_var = np.array([0.0002, 0.00025, 0.00018, 0.0003, 0.00022])
    # Persistence
    res_pers = MatchedVolatilityBaselines.persistence(daily_var, horizon=5)
    assert len(res_pers.predictions) == 5
    assert math_close(res_pers.predictions[0], 0.0002 * 5)

    # EWMA
    res_ewma = MatchedVolatilityBaselines.ewma_volatility(daily_var, horizon=5, decay=0.94)
    assert len(res_ewma.predictions) == 5
    assert (res_ewma.predictions > 0).all()

    # HAR
    res_har = MatchedVolatilityBaselines.har_rv(daily_var, horizon=5)
    assert len(res_har.predictions) == 5
    assert (res_har.predictions > 0).all()


def math_close(a, b, tol=1e-6):
    return abs(a - b) < tol
