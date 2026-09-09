"""GJR-GARCH and EGARCH reference fits with Gaussian likelihood (study only).

Mirrors the production GARCH(1,1) conventions: trailing return window,
penalty-guarded Gaussian negative log-likelihood, L-BFGS-B, and a
variance-targeted fallback when the optimizer fails. Multi-step paths are
cumulative daily variances. The EGARCH multi-step path iterates the
recursion on expectations (Jensen gap documented, not hidden).
Student-t likelihood stays gated behind the preregistered sub-arm decision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

_EPS = 1e-12
_PENALTY = 1e12
_SQRT_2_OVER_PI = float(np.sqrt(2.0 / np.pi))


def _clean_returns(close: pd.Series | np.ndarray, *, minimum: int = 60) -> np.ndarray:
    prices = np.asarray(
        close["Close"].to_numpy(dtype=float)
        if isinstance(close, pd.DataFrame)
        else pd.Series(close, dtype=float).to_numpy(dtype=float)
    ).reshape(-1)
    finite = prices[np.isfinite(prices) & (prices > 0.0)]
    returns = np.diff(np.log(finite))
    returns = returns[np.isfinite(returns)][-252:]
    if len(returns) < minimum:
        raise ValueError("Asymmetric GARCH requires at least sixty valid returns")
    return returns


def _gjr_conditional(
    returns: np.ndarray, omega: float, alpha: float, gamma: float, beta: float
) -> np.ndarray | None:
    conditional = np.empty(len(returns), dtype=np.float64)
    conditional[0] = max(float(np.var(returns, ddof=1)), _EPS)
    for index in range(1, len(returns)):
        previous = returns[index - 1]
        conditional[index] = (
            omega
            + alpha * previous**2
            + gamma * (previous**2 if previous < 0.0 else 0.0)
            + beta * conditional[index - 1]
        )
        if conditional[index] <= 0.0 or not np.isfinite(conditional[index]):
            return None
    return conditional


def fit_gjr_garch(close: pd.Series | np.ndarray) -> dict[str, float]:
    """Fit Gaussian GJR-GARCH(1,1); gamma >= 0 captures the leverage effect."""
    returns = _clean_returns(close)
    sample_var = max(float(np.var(returns, ddof=1)), 1e-8)

    def negative_log_likelihood(params: np.ndarray) -> float:
        omega, alpha, gamma, beta = (float(value) for value in params)
        if omega <= 0.0 or alpha < 0.0 or gamma < 0.0 or beta < 0.0:
            return _PENALTY
        if alpha + gamma / 2.0 + beta >= 1.0:
            return _PENALTY
        conditional = _gjr_conditional(returns, omega, alpha, gamma, beta)
        if conditional is None:
            return _PENALTY
        # Positive NLL: +0.5 * sum(lnc + r^2/c). (A leading minus sign here
        # would minimize the log-likelihood and pin every parameter at a
        # degenerate bound; the test DGPs guard this orientation.)
        return float(0.5 * np.sum(np.log(conditional) + returns**2 / conditional))

    initial = np.array([0.05 * sample_var, 0.06, 0.05, 0.85], dtype=np.float64)
    result = minimize(
        negative_log_likelihood,
        initial,
        method="L-BFGS-B",
        bounds=[(1e-10, 1.0), (1e-4, 0.40), (0.0, 0.40), (0.50, 0.999)],
        options={"maxiter": 150, "ftol": 1e-7},
    )
    if result.success:
        omega, alpha, gamma, beta = (float(value) for value in result.x)
        if alpha + gamma / 2.0 + beta < 1.0:
            return {
                "omega": omega,
                "alpha": alpha,
                "gamma": gamma,
                "beta": beta,
                "loglik": float(-result.fun),
                "converged": True,
            }
    alpha, gamma, beta = 0.06, 0.05, 0.85
    omega = (1.0 - alpha - gamma / 2.0 - beta) * sample_var
    return {
        "omega": omega,
        "alpha": alpha,
        "gamma": gamma,
        "beta": beta,
        "loglik": float("nan"),
        "converged": False,
    }


def gjr_cumulative_variance_path(
    close: pd.Series | np.ndarray, maximum_horizon: int = 20
) -> np.ndarray:
    """Cumulative Gaussian GJR variance path, mean-reverting at persistence
    alpha + gamma/2 + beta (E[I(z<0) z^2] = 1/2 under Gaussian shocks)."""
    if maximum_horizon < 1:
        raise ValueError("maximum_horizon must be positive")
    returns = _clean_returns(close)
    fit = fit_gjr_garch(returns)
    omega, alpha, gamma, beta = fit["omega"], fit["alpha"], fit["gamma"], fit["beta"]
    conditional = _gjr_conditional(returns, omega, alpha, gamma, beta)
    if conditional is None:  # pragma: no cover - guarded by fit bounds
        raise ValueError("GJR-GARCH produced an invalid variance filter")
    persistence = alpha + gamma / 2.0 + beta
    unconditional = omega / max(1.0 - persistence, 1e-5)
    next_variance = (
        omega
        + alpha * returns[-1] ** 2
        + gamma * (returns[-1] ** 2 if returns[-1] < 0.0 else 0.0)
        + beta * conditional[-1]
    )
    if persistence >= 0.9999:
        daily_path = np.full(maximum_horizon, next_variance, dtype=np.float64)
    else:
        daily_path = unconditional + (next_variance - unconditional) * persistence ** np.arange(
            maximum_horizon, dtype=np.float64
        )
    daily_path = np.maximum(daily_path, _EPS)
    cumulative_path = np.cumsum(daily_path)
    if not np.isfinite(cumulative_path).all() or np.any(np.diff(cumulative_path) < -1e-12):
        raise ValueError("GJR-GARCH produced an invalid cumulative variance path")
    return cumulative_path


def fit_egarch(close: pd.Series | np.ndarray) -> dict[str, float]:
    """Fit Gaussian EGARCH(1,1). A negative gamma is the leverage effect."""
    returns = _clean_returns(close)
    sample_var = max(float(np.var(returns, ddof=1)), 1e-8)
    log_sample = float(np.log(sample_var))

    def negative_log_likelihood(params: np.ndarray) -> float:
        omega, alpha, gamma, beta = (float(value) for value in params)
        if not -0.999 < beta < 0.999:
            return _PENALTY
        log_variance = np.empty(len(returns), dtype=np.float64)
        log_variance[0] = log_sample
        for index in range(1, len(returns)):
            shock = returns[index - 1] / max(np.sqrt(np.exp(log_variance[index - 1])), 1e-8)
            if not np.isfinite(shock):
                return _PENALTY
            log_variance[index] = (
                omega
                + alpha * (abs(shock) - _SQRT_2_OVER_PI)
                + gamma * shock
                + beta * log_variance[index - 1]
            )
        if not np.isfinite(log_variance).all():
            return _PENALTY
        variance = np.exp(log_variance)
        return float(0.5 * np.sum(np.log(variance) + returns**2 / variance))

    initial = np.array([(1.0 - 0.9) * log_sample, 0.10, -0.05, 0.90], dtype=np.float64)
    result = minimize(
        negative_log_likelihood,
        initial,
        method="L-BFGS-B",
        bounds=[(-5.0, 5.0), (0.0, 3.0), (-3.0, 3.0), (-0.999, 0.999)],
        options={"maxiter": 150, "ftol": 1e-7},
    )
    if result.success and abs(float(result.x[3])) < 1.0:
        omega, alpha, gamma, beta = (float(value) for value in result.x)
        return {
            "omega": omega,
            "alpha": alpha,
            "gamma": gamma,
            "beta": beta,
            "loglik": float(-result.fun),
            "converged": True,
        }
    return {
        "omega": (1.0 - 0.9) * log_sample,
        "alpha": 0.10,
        "gamma": -0.05,
        "beta": 0.90,
        "loglik": float("nan"),
        "converged": False,
    }


def egarch_cumulative_variance_path(
    close: pd.Series | np.ndarray, maximum_horizon: int = 20
) -> np.ndarray:
    """Cumulative EGARCH variance path.

    Iterates E[ln sigma^2] forward using E|z| = sqrt(2/pi) and E[z] = 0.
    This ignores the Jensen gap between E[ln sigma^2] and ln E[sigma^2];
    acceptable for a reference rung, documented here rather than hidden.
    """
    if maximum_horizon < 1:
        raise ValueError("maximum_horizon must be positive")
    returns = _clean_returns(close)
    fit = fit_egarch(returns)
    omega, alpha, gamma, beta = fit["omega"], fit["alpha"], fit["gamma"], fit["beta"]
    log_sample = float(np.log(max(float(np.var(returns, ddof=1)), 1e-8)))
    log_variance = np.empty(len(returns), dtype=np.float64)
    log_variance[0] = log_sample
    for index in range(1, len(returns)):
        shock = returns[index - 1] / max(np.sqrt(np.exp(log_variance[index - 1])), 1e-8)
        log_variance[index] = (
            omega
            + alpha * (abs(shock) - _SQRT_2_OVER_PI)
            + gamma * shock
            + beta * log_variance[index - 1]
        )
    if not np.isfinite(log_variance).all():
        raise ValueError("EGARCH produced a non-finite variance filter")
    next_log = (
        omega
        + alpha
        * (abs(returns[-1] / max(np.sqrt(np.exp(log_variance[-1])), 1e-8)) - _SQRT_2_OVER_PI)
        + gamma * (returns[-1] / max(np.sqrt(np.exp(log_variance[-1])), 1e-8))
        + beta * log_variance[-1]
    )
    steps = np.arange(1, maximum_horizon + 1, dtype=np.float64)
    if abs(beta) >= 0.9999:
        expected_log = np.full(maximum_horizon, next_log, dtype=np.float64)
    else:
        unconditional = omega / (1.0 - beta)
        expected_log = unconditional + (next_log - unconditional) * beta**steps
    daily_path = np.maximum(np.exp(expected_log), _EPS)
    cumulative_path = np.cumsum(daily_path)
    if not np.isfinite(cumulative_path).all() or np.any(np.diff(cumulative_path) < -1e-12):
        raise ValueError("EGARCH produced an invalid cumulative variance path")
    return cumulative_path
