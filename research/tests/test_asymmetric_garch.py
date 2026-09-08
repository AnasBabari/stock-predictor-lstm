import numpy as np
import pandas as pd
import pytest

from research.volatility_structure.asymmetric_garch import (
    egarch_cumulative_variance_path,
    fit_egarch,
    fit_gjr_garch,
    gjr_cumulative_variance_path,
)


def _simulate_gjr(n, omega, alpha, gamma, beta, seed):
    rng = np.random.default_rng(seed)
    shocks = rng.normal(size=n)
    variance = np.empty(n)
    eps = np.empty(n)
    variance[0] = omega / (1.0 - alpha - gamma / 2.0 - beta)
    for t in range(n):
        if t > 0:
            prev = eps[t - 1]
            variance[t] = (
                omega
                + alpha * prev**2
                + gamma * (prev**2 if prev < 0.0 else 0.0)
                + beta * variance[t - 1]
            )
        eps[t] = np.sqrt(variance[t]) * shocks[t]
    return pd.Series(100.0 * np.exp(np.cumsum(eps)))


def _gbm_closes(n=800, seed=3):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.012, n)
    return pd.Series(100.0 * np.exp(np.cumsum(rets)))


def test_gjr_recovers_leverage_on_asymmetric_dgp():
    closes = _simulate_gjr(1500, 1.5e-6, 0.06, 0.09, 0.88, seed=11)
    fit = fit_gjr_garch(closes)
    assert fit["converged"] is True
    assert fit["gamma"] > 0.02
    assert abs(fit["alpha"] + fit["gamma"] / 2.0 + fit["beta"] - 0.985) < 0.08


def test_gjr_gamma_near_zero_on_symmetric_dgp():
    closes = _simulate_gjr(1500, 2.0e-6, 0.08, 0.00, 0.90, seed=12)
    fit = fit_gjr_garch(closes)
    assert fit["gamma"] < 0.08
    assert abs(fit["alpha"] + fit["beta"] - 0.98) < 0.10


def test_gjr_path_is_finite_monotone_cumulative():
    path = gjr_cumulative_variance_path(_gbm_closes(), maximum_horizon=20)
    assert len(path) == 20
    assert np.isfinite(path).all() and (path > 0).all()
    assert (np.diff(path) >= -1e-12).all()


def test_egarch_fit_and_path_are_stable():
    closes = _simulate_gjr(1500, 1.5e-6, 0.06, 0.09, 0.88, seed=13)
    fit = fit_egarch(closes)
    assert abs(fit["beta"]) < 1.0
    assert np.isfinite([fit["omega"], fit["alpha"], fit["gamma"]]).all()
    path = egarch_cumulative_variance_path(closes, maximum_horizon=20)
    assert len(path) == 20
    assert np.isfinite(path).all() and (path > 0).all()
    assert (np.diff(path) >= -1e-12).all()


def test_short_or_invalid_input_raises():
    with pytest.raises(ValueError, match="sixty"):
        fit_gjr_garch(_gbm_closes(n=30))
    with pytest.raises(ValueError, match="sixty"):
        fit_egarch(_gbm_closes(n=30))
    with pytest.raises(ValueError, match="positive"):
        gjr_cumulative_variance_path(_gbm_closes(), maximum_horizon=0)
