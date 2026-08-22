"""Volatility proxies, econometric baselines, and QLIKE metrics (slice 6).

All estimators are causal: a forecast for origin t may only use observations
up to and including row t. Fitted parameters come from the caller-supplied
training slice; the filter functions then roll forward through evaluation
data one bar at a time, exactly as a live origin would.

Volatility proxies (daily, from OHLC):
- Rogers–Satchell intraday component
- Overnight squared gap + intraday sum (free-data total variance proxy)
- Close-to-close squared return
- Parkinson range estimator

Baselines: EWMA (RiskMetrics), HAR-RV (OLS over daily/weekly/monthly
components), GARCH(1,1) and GJR-GARCH(1,1,1) via scipy MLE with normal
innovations. h-step cumulative variance uses mean-reverting GARCH recursion;
EWMA/HAR scale by h under a random-walk-in-variance assumption.

QLIKE (Patton 2011): QLIKE(p, a) = a/p − ln(a/p) − 1 for p, a > 0; zero iff
the forecast equals the proxy target. Lower is better.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ── Realized variance proxies (per-session, from OHLC rows) ─────────────


def rogers_satchell_frame(df: pd.DataFrame) -> pd.Series:
    ho = np.log(df["High"] / df["Open"])
    hc = np.log(df["High"] / df["Close"])
    lo = np.log(df["Low"] / df["Open"])
    lc = np.log(df["Low"] / df["Close"])
    return ho * hc + lo * lc


def realized_variance_proxies(df: pd.DataFrame) -> pd.DataFrame:
    """Daily variance proxies aligned to each session's date."""
    out = df.copy()
    log_close = np.log(df["Close"].where(df["Close"] > 0))
    ret = log_close.diff()
    overnight = np.log(df["Open"].where(df["Open"] > 0)) - log_close.shift(1)
    rs = rogers_satchell_frame(df)
    out["RV_C2C"] = ret.pow(2)
    out["RV_Overnight"] = overnight.pow(2)
    out["RV_RS_Intraday"] = rs
    # Free-data total: overnight gap + intraday RS (positive daily sum).
    out["RV_Total"] = out["RV_Overnight"] + out["RV_RS_Intraday"]
    parkinson = np.log(df["High"].where(df["High"] > 0) / df["Low"].where(df["Low"] > 0)).pow(2)
    out["RV_Parkinson"] = parkinson / (4 * np.log(2))
    return out[["RV_C2C", "RV_Overnight", "RV_RS_Intraday", "RV_Total", "RV_Parkinson"]]


def cumulative_variance_target(
    rv_daily: pd.Series,
    horizon: int,
    *,
    origin_index: int | None = None,
) -> pd.Series:
    """Sum of the next `horizon` daily variance proxies per origin t.

    Origin t receives sum(rv[t+1 .. t+h]) — strictly future realizations,
    never the origin's own row. Windows containing non-finite entries are
    returned as NaN rather than silently zero-filled.
    """
    h = max(1, int(horizon))
    values = rv_daily.to_numpy(dtype=float)
    n = len(values)
    finite = np.isfinite(values)
    safe = np.where(finite, values, 0.0)
    prefix = np.concatenate(([0.0], np.cumsum(safe)))
    origins = np.arange(n)
    # Origin t sums rv[t+1 .. t+h]: exclusive start at t+1.
    starts: np.ndarray = origins + 1
    ends: np.ndarray = starts + h
    in_range = ends < len(prefix)
    safe_end = np.minimum(ends, len(prefix) - 1)
    sums = prefix[safe_end] - prefix[starts]
    valid = in_range & (starts < len(prefix))
    if origin_index is not None:
        valid &= ends <= origin_index + 1
    # A window containing NaN cannot be honestly summed.
    nan_counts = pd.Series(~finite).rolling(h).sum().to_numpy()
    sums[~finite] = np.nan
    sums[(nan_counts > 0) & valid] = np.nan
    sums[~valid] = np.nan
    return pd.Series(sums, index=rv_daily.index, name=f"CV_Target_{h}")


# ── EWMA / RiskMetrics ───────────────────────────────────────────────────


def ewma_variance(returns: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """Recursive sigma²_t = λσ²_{t-1} + (1−λ)r²_{t-1}, seeded with r̄²."""
    r = np.asarray(returns, dtype=float)
    var = np.empty_like(r)
    seed = float(np.mean(r[: min(20, len(r))] ** 2)) or 1e-8
    prev = seed
    for i in range(len(r)):
        var[i] = prev
        prev = lam * prev + (1 - lam) * r[i] ** 2
    return var


def ewma_forecast_cumulative(returns: np.ndarray, horizon: int, lam: float = 0.94) -> float:
    """Random-walk-in-variance forecast: h × current filtered σ²."""
    path = ewma_variance(returns, lam)
    return float(horizon * path[-1])


# ── HAR-RV ───────────────────────────────────────────────────────────────


def har_components(rv_daily: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Design matrix [const, RV_d, RV_w(mean5), RV_m(mean22)] → next-day RV."""
    rv = np.asarray(rv_daily, dtype=float)
    weekly = pd.Series(rv).rolling(5).mean().to_numpy()
    monthly = pd.Series(rv).rolling(22).mean().to_numpy()
    X = np.column_stack([np.ones(len(rv)), rv, weekly, monthly])
    y = np.roll(rv, -1)
    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    valid[-1] = False  # last row has no forward target
    return X[valid], y[valid]


def fit_har(rv_train: np.ndarray) -> np.ndarray:
    X, y = har_components(rv_train)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def har_forecast_path(rv_full: np.ndarray, coef: np.ndarray, horizon: int) -> np.ndarray:
    """One-next-day forecast per origin using only information ≤ that origin."""
    rv = np.asarray(rv_full, dtype=float)
    weekly = pd.Series(rv).rolling(5).mean().to_numpy()
    monthly = pd.Series(rv).rolling(22).mean().to_numpy()
    X = np.column_stack([np.ones(len(rv)), rv, weekly, monthly])
    raw = X @ coef
    out = np.full(len(rv), np.nan)
    # Origin t predicts day t+1; its inputs end at t (weekly/monthly means of
    # trailing windows ending at t are backward-looking ✓).
    valid = ~np.isnan(raw[:-1])
    out[:-1][valid] = raw[:-1][valid]
    return np.maximum(out * horizon, 0.0)


# ── GARCH(1,1) / GJR-GARCH(1,1,1) MLE ───────────────────────────────────


@dataclass(frozen=True)
class GarchParams:
    omega: float
    alpha: float
    gamma: float  # 0 for plain GARCH
    beta: float
    optimizer: str = "SLSQP"
    initialization_count: int = 1
    convergence_state: str = "converged"
    objective: float = 0.0
    fallback_reason: str | None = None

    @property
    def persistence(self) -> float:
        return self.alpha + self.gamma / 2 + self.beta


def _garch_filter(returns: np.ndarray, p: GarchParams) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    var = np.empty_like(r)
    var[0] = float(np.var(r[: min(20, len(r))])) or 1e-10
    for i in range(1, len(r)):
        asym = p.gamma * min(r[i - 1], 0.0) ** 2
        var[i] = p.omega + p.alpha * r[i - 1] ** 2 + asym + p.beta * var[i - 1]
    return var


def _garch_nll(theta: np.ndarray, returns: np.ndarray, gjr: bool) -> float:
    omega = float(theta[0])
    alpha = float(theta[1])
    beta = float(theta[2])
    gamma = float(theta[3]) if gjr else 0.0
    if omega <= 0 or alpha < 0 or beta < 0 or gamma < 0:
        return 1e12
    persistence = alpha + beta + (gamma / 2 if gjr else 0.0)
    if persistence >= 0.99999:
        return 1e12
    params = GarchParams(omega, alpha, gamma, beta)
    var = _garch_filter(returns, params)
    if not np.isfinite(var).all() or (var <= 0).any():
        return 1e12
    z2 = returns.astype(float) ** 2 / var
    nll = 0.5 * np.sum(np.log(var) + z2)
    return float(min(nll, 1e18)) if np.isfinite(nll) else 1e12


def fit_garch(returns: np.ndarray, *, gjr: bool = False) -> GarchParams:
    """Deterministic constrained multi-start MLE for GARCH(1,1)/GJR-GARCH(1,1,1)."""
    r = np.asarray(returns, dtype=float)
    if len(r) == 0:
        return GarchParams(
            omega=1e-6,
            alpha=0.06,
            gamma=0.0,
            beta=0.92,
            optimizer="none",
            initialization_count=0,
            convergence_state="failed",
            objective=float("inf"),
            fallback_reason="empty_series",
        )

    seed_var = float(np.var(r))
    if not np.isfinite(seed_var) or seed_var <= 1e-12:
        safe_omega = 1e-8
        return GarchParams(
            omega=safe_omega,
            alpha=0.04 if gjr else 0.06,
            gamma=0.04 if gjr else 0.0,
            beta=0.92,
            optimizer="none",
            initialization_count=0,
            convergence_state="fallback",
            objective=0.0,
            fallback_reason="degenerate_variance",
        )

    if gjr:
        starts = [
            (0.95, 0.05, 0.86, 0.08),
            (0.98, 0.03, 0.93, 0.04),
            (0.90, 0.08, 0.76, 0.12),
            (0.85, 0.06, 0.75, 0.08),
            (0.96, 0.08, 0.88, 0.00),
        ]
        bounds = [(1e-12, 10.0), (0.0, 0.99), (0.0, 0.99), (0.0, 0.99)]
        constraints = [{"type": "ineq", "fun": lambda x: 0.9999 - (x[1] + x[2] + x[3] / 2.0)}]
    else:
        starts = [
            (0.95, 0.08, 0.87, 0.0),
            (0.98, 0.04, 0.94, 0.0),
            (0.90, 0.12, 0.78, 0.0),
            (0.80, 0.15, 0.65, 0.0),
            (0.99, 0.02, 0.97, 0.0),
        ]
        bounds = [(1e-12, 10.0), (0.0, 0.99), (0.0, 0.99)]
        constraints = [{"type": "ineq", "fun": lambda x: 0.9999 - (x[1] + x[2])}]

    candidates: list[tuple[float, GarchParams]] = []
    init_count = 0

    for p_init, a_init, b_init, g_init in starts:
        init_count += 1
        w_init = seed_var * max(1.0 - p_init, 0.01)
        x0 = (
            np.array([w_init, a_init, b_init, g_init], dtype=float)
            if gjr
            else np.array([w_init, a_init, b_init], dtype=float)
        )

        try:
            res = minimize(
                _garch_nll,
                x0,
                args=(r, gjr),
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 400, "ftol": 1e-7},
            )
            if res.success or res.fun < 1e10:
                w, a, b = float(res.x[0]), float(res.x[1]), float(res.x[2])
                g = float(res.x[3]) if gjr else 0.0
                pers = a + b + g / 2.0
                if w > 0 and a >= 0 and b >= 0 and g >= 0 and pers < 0.99999:
                    obj = _garch_nll(res.x, r, gjr)
                    if np.isfinite(obj) and obj < 1e10:
                        params = GarchParams(
                            omega=w,
                            alpha=a,
                            gamma=g,
                            beta=b,
                            optimizer="SLSQP",
                            initialization_count=init_count,
                            convergence_state="converged",
                            objective=float(obj),
                            fallback_reason=None,
                        )
                        candidates.append((obj, params))
        except Exception:
            pass

        try:
            res = minimize(
                _garch_nll,
                x0,
                args=(r, gjr),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 400, "ftol": 1e-7},
            )
            if res.success or res.fun < 1e10:
                w, a, b = float(res.x[0]), float(res.x[1]), float(res.x[2])
                g = float(res.x[3]) if gjr else 0.0
                pers = a + b + g / 2.0
                if w > 0 and a >= 0 and b >= 0 and g >= 0 and pers < 0.99999:
                    obj = _garch_nll(res.x, r, gjr)
                    if np.isfinite(obj) and obj < 1e10:
                        params = GarchParams(
                            omega=w,
                            alpha=a,
                            gamma=g,
                            beta=b,
                            optimizer="L-BFGS-B",
                            initialization_count=init_count,
                            convergence_state="converged",
                            objective=float(obj),
                            fallback_reason=None,
                        )
                        candidates.append((obj, params))
        except Exception:
            pass

    if candidates:
        candidates.sort(key=lambda c: c[0])
        best_params = candidates[0][1]
        return GarchParams(
            omega=best_params.omega,
            alpha=best_params.alpha,
            gamma=best_params.gamma,
            beta=best_params.beta,
            optimizer=best_params.optimizer,
            initialization_count=init_count,
            convergence_state=best_params.convergence_state,
            objective=best_params.objective,
            fallback_reason=None,
        )

    fb_omega = seed_var * 0.02
    return GarchParams(
        omega=max(fb_omega, 1e-12),
        alpha=0.04 if gjr else 0.06,
        gamma=0.04 if gjr else 0.0,
        beta=0.92,
        optimizer="fallback_riskmetrics",
        initialization_count=init_count,
        convergence_state="fallback",
        objective=float("inf"),
        fallback_reason="all_starts_failed_constraints",
    )


def garch_forecast_cumulative(returns: np.ndarray, params: GarchParams, horizon: int) -> float:
    """E[Σ_{i=1..h} σ²_{t+i}] with mean reversion toward ω/(1−α−β−γ/2)."""
    var_path = _garch_filter(returns, params)
    last_return = float(returns[-1])
    sigma2_t1 = (
        params.omega
        + params.alpha * last_return**2
        + params.gamma * min(last_return, 0.0) ** 2
        + params.beta * var_path[-1]
    )
    total = 0.0
    level = sigma2_t1
    for _ in range(max(1, int(horizon))):
        total += level
        level = params.omega + params.persistence * level
    return float(total)


# ── QLIKE and log-variance scoring ───────────────────────────────────────


def qlike_loss(forecast_var: np.ndarray, realized_var: np.ndarray) -> float:
    """Mean QLIKE over origins with both entries positive; NaN otherwise."""
    p = np.asarray(forecast_var, dtype=float)
    a = np.asarray(realized_var, dtype=float)
    mask = np.isfinite(p) & np.isfinite(a) & (p > 0) & (a > 0)
    if not mask.any():
        return float("nan")
    ratio = a[mask] / p[mask]
    return float(np.mean(ratio - np.log(ratio) - 1))


def relative_qlike(model_qlike: float, baseline_qlike: float) -> float | None:
    if not (np.isfinite(baseline_qlike) and baseline_qlike != 0):
        return None
    return float(model_qlike / baseline_qlike)


def log_variance_errors(forecast_var: np.ndarray, realized_var: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(forecast_var) & np.isfinite(realized_var) & (realized_var > 0)
    diff = np.log(realized_var[mask]) - np.log(forecast_var[mask])
    return {
        "logvar_mae": float(np.mean(np.abs(diff))),
        "logvar_rmse": float(np.sqrt(np.mean(diff**2))),
    }
