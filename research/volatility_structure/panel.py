"""Frozen A-F ablation panel for the daily-volatility-structure study.

Arm matrix (frozen before any execution; builders must not redefine arms):
  A        rolling champion: H * trailing-20d variance (production parity).
  B1-*     standalone range forecasts: trailing-20d mean range variance * H
           (parkinson / garman_klass / rogers_satchell / yang_zhang).
  B2       Ridge on the four trailing-20d mean range variances.
  C        production recursive HAR forecasts (har_wrapper).
  D        Ridge on HAR log components + range block.
  E        Ridge on D features + sign-conditioned variation block.
  F-gjr    Gaussian GJR-GARCH cumulative paths (analytic mean reversion).
  F-egarch Gaussian EGARCH cumulative paths (expected-recursion approx).

Learned arms share one frozen class: Ridge(alpha=100), StandardScaler fit
on train origins only, one model per (arm, horizon). GJR/EGARCH refit every
20 trading sessions (frozen cadence; still strictly trailing history).
Targets are forward cumulative realized variance; QLIKE is primary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_HORIZONS: tuple[int, int, int] = (5, 10, 20)
QLIKE_FLOOR = 1e-12
RIDGE_ALPHA = 100.0
GJR_REFIT_EVERY = 20
TRAILING_WINDOW = 20

ARMS: dict[str, dict[str, str]] = {
    "A": {
        "kind": "standalone",
        "multistep_method": "constant_volatility_extrapolation",
        "description": "H * trailing-20d close-to-close variance",
    },
    "B1-parkinson": {
        "kind": "standalone",
        "multistep_method": "constant_volatility_extrapolation",
        "description": "H * trailing-20d mean Parkinson variance",
    },
    "B1-garman_klass": {
        "kind": "standalone",
        "multistep_method": "constant_volatility_extrapolation",
        "description": "H * trailing-20d mean Garman-Klass variance",
    },
    "B1-rogers_satchell": {
        "kind": "standalone",
        "multistep_method": "constant_volatility_extrapolation",
        "description": "H * trailing-20d mean Rogers-Satchell variance",
    },
    "B1-yang_zhang": {
        "kind": "standalone",
        "multistep_method": "constant_volatility_extrapolation",
        "description": "H * trailing-20d mean Yang-Zhang variance",
    },
    "B2": {
        "kind": "ridge",
        "multistep_method": "direct_per_horizon",
        "description": "Ridge on four trailing-20d mean range variances",
    },
    "C": {
        "kind": "standalone",
        "multistep_method": "recursive_har",
        "description": "production recursive HAR cumulative forecasts",
    },
    "D": {
        "kind": "ridge",
        "multistep_method": "direct_per_horizon",
        "description": "Ridge on HAR log components + range block",
    },
    "E": {
        "kind": "ridge",
        "multistep_method": "direct_per_horizon",
        "description": "Ridge on D features + sign-conditioned variation block",
    },
    "F-gjr": {
        "kind": "standalone",
        "multistep_method": "gaussian_closed_form",
        "description": "Gaussian GJR-GARCH cumulative paths, refit every 20 sessions",
    },
    "F-egarch": {
        "kind": "standalone",
        "multistep_method": "expected_recursion_approx",
        "description": "Gaussian EGARCH cumulative paths, refit every 20 sessions",
    },
}

B2_COLUMNS: tuple[str, ...] = (
    "range_park_20",
    "range_gk_20",
    "range_rs_20",
    "range_yz_20",
)
D_EXTRA_COLUMNS: tuple[str, ...] = (
    "har_log_daily",
    "har_log_weekly",
    "har_log_monthly",
)
E_EXTRA_COLUMNS: tuple[str, ...] = (
    "downside_sq_20",
    "upside_sq_20",
    "neg_freq_20",
)
FEATURE_SCHEMA_VERSION = "vol-structure-v1"


def daily_log_returns(close: pd.Series) -> pd.Series:
    prices = pd.Series(close, dtype=float)
    if prices.empty:
        raise ValueError("Close input is empty")
    return pd.Series(
        np.log(prices.to_numpy() / pd.Series(prices.to_numpy()).shift(1).to_numpy()),
        index=prices.index,
    )


def forward_realized_variance(close: pd.Series, horizon: int) -> pd.Series:
    """Forward cumulative sum of squared log returns over t+1..t+h.

    Origin ``o`` covers returns ``o+1..o+h`` (closes ``o..o+h``): strictly
    after the origin, excluding the already-known return into ``o``.
    Origins without a full forward window are NaN.
    """
    if horizon < 1:
        raise ValueError("horizon must be positive")
    returns = daily_log_returns(close)
    squared = returns.to_numpy() ** 2
    forward = np.full(len(close), np.nan)
    for origin in range(len(close) - horizon):
        window = squared[origin + 1 : origin + horizon + 1]
        if np.isfinite(window).all():
            forward[origin] = float(np.sum(window))
    return pd.Series(forward, index=close.index)


def trailing_variance(returns: pd.Series, window: int = TRAILING_WINDOW) -> pd.Series:
    return returns.rolling(window, min_periods=window).var(ddof=1)


def arm_a_forecasts(close: pd.Series) -> pd.DataFrame:
    """Rolling champion: H * trailing-20d close-to-close variance."""
    returns = daily_log_returns(close)
    trail_var = trailing_variance(returns)
    return pd.DataFrame(
        {f"f_h{h}": trail_var.to_numpy() * h for h in TARGET_HORIZONS},
        index=close.index,
    )


def arm_b1_forecasts(range_variances: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Standalone range forecasts from each estimator's trailing-20d mean.

    Computed on RAW estimator values (Garman-Klass negatives preserved);
    the non-negative floor lives only at the scoring boundary
    (apply_forecast_floor), never in feature construction.
    """
    mapping = {
        "B1-parkinson": "parkinson_var",
        "B1-garman_klass": "garman_klass_var",
        "B1-rogers_satchell": "rogers_satchell_var",
        "B1-yang_zhang": "yang_zhang_var",
    }
    out = {}
    for arm, column in mapping.items():
        mean20 = (
            range_variances[column].rolling(TRAILING_WINDOW, min_periods=TRAILING_WINDOW).mean()
        )
        out[arm] = pd.DataFrame(
            {f"f_h{h}": mean20.to_numpy() * h for h in TARGET_HORIZONS},
            index=range_variances.index,
        )
    return out


def apply_forecast_floor(predictions: pd.DataFrame, floor: float = 0.0) -> pd.DataFrame:
    """The explicitly defined forecast boundary: variance forecasts cannot
    be negative. Applied at scoring time only, never inside features."""
    floored = predictions.copy()
    for column in floored.columns:
        if column.startswith("f_h"):
            floored[column] = floored[column].clip(lower=floor)
    return floored


def range_block_features(range_variances: pd.DataFrame) -> pd.DataFrame:
    """Frozen B2/D range block: trailing-20d RAW means (NaN-aware)."""
    mapping = {
        "range_park_20": "parkinson_var",
        "range_gk_20": "garman_klass_var",
        "range_rs_20": "rogers_satchell_var",
        "range_yz_20": "yang_zhang_var",
    }
    return pd.DataFrame(
        {
            name: range_variances[column]
            .rolling(TRAILING_WINDOW, min_periods=TRAILING_WINDOW)
            .mean()
            for name, column in mapping.items()
        },
        index=range_variances.index,
    )


def fit_ridge_direct(x_train: np.ndarray, y_train: np.ndarray):
    """Frozen learned class: StandardScaler fit on train only + Ridge(100)."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(x_train)
    model = Ridge(alpha=RIDGE_ALPHA, solver="cholesky").fit(scaler.transform(x_train), y_train)
    return scaler, model


def qlike(actual: float, forecast: float) -> float:
    """QLIKE loss with a shared floor (Patton-robust to proxy noise)."""
    y = max(float(actual), QLIKE_FLOOR)
    f = max(float(forecast), QLIKE_FLOOR)
    return float(y / f - np.log(y / f) - 1.0)


def partitions(dates: np.ndarray, label_end: np.ndarray) -> tuple[np.ndarray, np.ndarray, str, str]:
    """Global unique-date 70/15/15 split; labels must end before the next block.

    Mirrors the rank-study construction so results stay comparable. Inputs
    are normalized to datetime64 (object-dtype Timestamp arrays do not
    compare against scalars under NumPy 2.x).
    """
    dates = pd.to_datetime(dates).to_numpy(dtype="datetime64[ns]")
    label_end = pd.to_datetime(label_end).to_numpy(dtype="datetime64[ns]")
    unique = np.sort(np.unique(dates))
    validation_start = unique[int(len(unique) * 0.70)]
    reserve_start = unique[int(len(unique) * 0.85)]
    train = (dates < validation_start) & (label_end < validation_start)
    validation = (dates >= validation_start) & (dates < reserve_start) & (label_end < reserve_start)
    return train, validation, str(validation_start), str(reserve_start)
