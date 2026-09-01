"""Deployable, causal inputs for the active volatility baseline service."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from calendars import future_trading_dates
from data_pipeline import _download_ohlcv

VOLATILITY_HORIZONS = (1, 3, 5, 7, 14, 30)
VOLATILITY_WINDOW_SIZE = 60
VOLATILITY_PATH_HORIZONS = tuple(range(1, max(VOLATILITY_HORIZONS) + 1))

DEPLOYABLE_FEATURE_COLUMNS_V5 = (
    "log_return_1d",
    "log_return_5d",
    "log_return_20d",
    "Vol_C2C_5",
    "Vol_C2C_20",
    "Vol_C2C_60",
    "EWMA_Var",
    "Vol_Parkinson_20",
    "Vol_GarmanKlass_20",
    "Vol_RogersSatchell_20",
    "vol_ratio_5_20",
    "vol_ratio_20_60",
    "norm_range_1d",
    "norm_range_5d",
    "norm_range_20d",
    "volume_log_ratio_20d",
    "volume_zscore_20d",
)


def realized_variance_proxies(df: pd.DataFrame) -> pd.DataFrame:
    """Compute close-to-close daily realized variance proxy."""
    close = df["Close"].to_numpy(dtype=np.float64)
    ret = np.zeros_like(close)
    ret[1:] = np.log(close[1:] / close[:-1])
    rv_c2c = ret**2
    return pd.DataFrame({"RV_C2C": rv_c2c}, index=df.index)


def _log_har_row(history: np.ndarray) -> np.ndarray:
    """Log-HAR predictors ending at the final value in ``history``."""
    if len(history) < 22:
        raise ValueError("HAR row requires at least 22 realized observations")
    safe = np.maximum(np.asarray(history, dtype=np.float64), 1e-12)
    return np.array(
        [
            1.0,
            np.log(safe[-1]),
            np.log(np.mean(safe[-5:])),
            np.log(np.mean(safe[-22:])),
        ],
        dtype=np.float64,
    )


def causal_log_har_forecasts(
    rv_daily: pd.Series | np.ndarray,
    horizons: tuple[int, ...] | list[int] = VOLATILITY_HORIZONS,
    *,
    minimum_history: int = 60,
    refit_every: int = 5,
    ridge: float = 1e-4,
) -> np.ndarray:
    """Return causal cumulative variance forecasts for every origin."""
    if not horizons or min(horizons) < 1:
        raise ValueError("horizons must contain positive integers")
    if refit_every < 1:
        raise ValueError("refit_every must be positive")
    rv = np.asarray(rv_daily, dtype=np.float64)
    output = np.full((len(rv), len(horizons)), np.nan, dtype=np.float64)
    coefficients: np.ndarray | None = None
    last_refit = -refit_every
    maximum_horizon = max(horizons)
    horizon_to_column = {horizon: column for column, horizon in enumerate(horizons)}
    xtx = np.zeros((4, 4), dtype=np.float64)
    xty = np.zeros(4, dtype=np.float64)
    fitted_rows = 0
    penalty = np.eye(4, dtype=np.float64) * float(ridge)
    penalty[0, 0] = 0.0

    for origin in range(22, len(rv)):
        training_origin = origin - 1
        if np.isfinite(rv[training_origin - 21 : training_origin + 2]).all():
            design = _log_har_row(rv[training_origin - 21 : training_origin + 1])
            response = float(np.log(max(rv[origin], 1e-12)))
            xtx += np.outer(design, design)
            xty += design * response
            fitted_rows += 1
        if origin < max(minimum_history, 22) or fitted_rows < 20:
            continue
        if not np.isfinite(rv[origin - 21 : origin + 1]).all():
            continue
        if coefficients is None or origin - last_refit >= refit_every:
            coefficients = np.linalg.solve(xtx + penalty, xty)
            last_refit = origin

        history_tail = list(np.maximum(rv[origin - 21 : origin + 1], 1e-12))
        cumulative = 0.0
        for step in range(1, maximum_horizon + 1):
            row = _log_har_row(np.asarray(history_tail, dtype=np.float64))
            next_variance = float(np.exp(np.clip(row @ coefficients, -30.0, 5.0)))
            next_variance = max(next_variance, 1e-12)
            history_tail.append(next_variance)
            del history_tail[0]
            cumulative += next_variance
            if step in horizon_to_column:
                output[origin, horizon_to_column[step]] = cumulative
    return output


def build_features_v5(df: pd.DataFrame) -> pd.DataFrame:
    """Extract causal features for deployable volatility inference."""
    close = df["Close"].to_numpy(dtype=np.float64)
    open_p = df["Open"].to_numpy(dtype=np.float64)
    high = df["High"].to_numpy(dtype=np.float64)
    low = df["Low"].to_numpy(dtype=np.float64)
    vol = df["Volume"].to_numpy(dtype=np.float64)

    ret_1 = np.full_like(close, np.nan)
    ret_1[1:] = np.log(close[1:] / close[:-1])
    s_ret = pd.Series(ret_1, index=df.index)

    ret_5 = np.full_like(close, np.nan)
    ret_5[5:] = np.log(close[5:] / close[:-5])

    ret_20 = np.full_like(close, np.nan)
    ret_20[20:] = np.log(close[20:] / close[:-20])

    vol_c2c_5 = s_ret.rolling(5).std().to_numpy()
    vol_c2c_20 = s_ret.rolling(20).std().to_numpy()
    vol_c2c_60 = s_ret.rolling(60).std().to_numpy()

    # EWMA variance (RiskMetrics lambda = 0.94)
    ewma_var = s_ret.pow(2).ewm(alpha=0.06, adjust=False).mean().to_numpy()

    # Parkinson range volatility
    hl_log = np.log(np.maximum(high, 1e-8) / np.maximum(low, 1e-8))
    s_park = pd.Series(hl_log**2 / (4.0 * np.log(2.0)), index=df.index)
    vol_park_20 = np.sqrt(np.maximum(s_park.rolling(20).mean().to_numpy(), 1e-12))

    # Garman-Klass volatility
    co_log = np.log(np.maximum(close, 1e-8) / np.maximum(open_p, 1e-8))
    s_gk = pd.Series(0.5 * (hl_log**2) - (2.0 * np.log(2.0) - 1.0) * (co_log**2), index=df.index)
    vol_gk_20 = np.sqrt(np.maximum(s_gk.rolling(20).mean().to_numpy(), 1e-12))

    # Rogers-Satchell volatility
    ho_log = np.log(np.maximum(high, 1e-8) / np.maximum(open_p, 1e-8))
    lo_log = np.log(np.maximum(low, 1e-8) / np.maximum(open_p, 1e-8))
    s_rs = pd.Series(ho_log * (ho_log - co_log) + lo_log * (lo_log - co_log), index=df.index)
    vol_rs_20 = np.sqrt(np.maximum(s_rs.rolling(20).mean().to_numpy(), 1e-12))

    vol_ratio_5_20 = vol_c2c_5 / np.maximum(vol_c2c_20, 1e-6)
    vol_ratio_20_60 = vol_c2c_20 / np.maximum(vol_c2c_60, 1e-6)

    norm_range_1d = (high - low) / np.maximum(close, 1e-6)
    s_nr = pd.Series(norm_range_1d, index=df.index)
    norm_range_5d = s_nr.rolling(5).mean().to_numpy()
    norm_range_20d = s_nr.rolling(20).mean().to_numpy()

    s_vol = pd.Series(vol, index=df.index)
    vol_mean_20 = s_vol.rolling(20).mean().to_numpy()
    vol_std_20 = s_vol.rolling(20).std().to_numpy()
    volume_log_ratio_20d = np.log(np.maximum(vol, 1.0) / np.maximum(vol_mean_20, 1.0))
    volume_zscore_20d = (vol - vol_mean_20) / np.maximum(vol_std_20, 1.0)

    out = pd.DataFrame(
        {
            "log_return_1d": ret_1,
            "log_return_5d": ret_5,
            "log_return_20d": ret_20,
            "Vol_C2C_5": vol_c2c_5,
            "Vol_C2C_20": vol_c2c_20,
            "Vol_C2C_60": vol_c2c_60,
            "EWMA_Var": ewma_var,
            "Vol_Parkinson_20": vol_park_20,
            "Vol_GarmanKlass_20": vol_gk_20,
            "Vol_RogersSatchell_20": vol_rs_20,
            "vol_ratio_5_20": vol_ratio_5_20,
            "vol_ratio_20_60": vol_ratio_20_60,
            "norm_range_1d": norm_range_1d,
            "norm_range_5d": norm_range_5d,
            "norm_range_20d": norm_range_20d,
            "volume_log_ratio_20d": volume_log_ratio_20d,
            "volume_zscore_20d": volume_zscore_20d,
        },
        index=df.index,
    )
    return out


@dataclass(frozen=True)
class VolatilityInferenceSnapshot:
    ticker: str
    snapshot_id: str
    origin_date: str
    origin_close: float
    feature_names: tuple[str, ...]
    features: np.ndarray
    causal_har_variance: np.ndarray
    baseline_candidates: dict[str, np.ndarray]
    historical_dates: tuple[str, ...]
    historical_prices: np.ndarray
    future_dates: tuple[str, ...] = ()
    baseline_variance_paths: dict[str, np.ndarray] | None = None

    def __post_init__(self) -> None:
        if self.features.shape != (VOLATILITY_WINDOW_SIZE, len(self.feature_names)):
            raise ValueError("volatility feature window has an incompatible shape")
        if self.causal_har_variance.shape != (len(VOLATILITY_HORIZONS),):
            raise ValueError("volatility HAR baseline has an incompatible shape")
        if not np.isfinite(self.features).all():
            raise ValueError("volatility feature window must be finite")
        if not np.isfinite(self.causal_har_variance).all() or (self.causal_har_variance <= 0).any():
            raise ValueError("volatility HAR baseline must be finite and positive")
        if self.baseline_variance_paths is not None:
            if "causal_log_har" not in self.baseline_variance_paths:
                raise ValueError("baseline variance paths must contain causal_log_har")
            har_path = self.baseline_variance_paths["causal_log_har"]
            if har_path.shape != (max(VOLATILITY_HORIZONS),):
                raise ValueError("HAR variance path has an incompatible shape")
            if not np.isfinite(har_path).all() or (har_path <= 0).any():
                raise ValueError("HAR variance path must be finite and positive")


def _baseline_candidates(feature_row: pd.Series, har_variance: np.ndarray) -> dict[str, np.ndarray]:
    """Map causal trailing features to multi-horizon variance forecasts."""
    candidates: dict[str, np.ndarray] = {
        "causal_log_har": np.asarray(har_variance, dtype=np.float64)
    }
    horizons = np.asarray(VOLATILITY_HORIZONS, dtype=np.float64)
    source = {
        "riskmetrics_ewma_c2c": "EWMA_Var",
        "rolling_c2c_5": "Vol_C2C_5",
        "rolling_c2c_20": "Vol_C2C_20",
        "rolling_c2c_60": "Vol_C2C_60",
    }
    for family, column in source.items():
        val = float(feature_row[column])
        if np.isfinite(val) and val > 0:
            daily_var = val if family == "riskmetrics_ewma_c2c" else val**2
            forecast = np.maximum(daily_var * horizons, 1e-12)
            candidates[family] = forecast
    rolling = [
        candidates[name]
        for name in ("rolling_c2c_5", "rolling_c2c_20", "rolling_c2c_60")
        if name in candidates
    ]
    if len(rolling) == 3:
        candidates["rolling_c2c_multiscale"] = np.exp(
            np.mean(np.log(np.maximum(np.stack(rolling), 1e-12)), axis=0)
        )
    return candidates


def _snapshot_identity(
    ticker: str,
    dates: pd.DatetimeIndex,
    features: np.ndarray,
    baseline: np.ndarray,
    baseline_paths: dict[str, np.ndarray] | None = None,
) -> str:
    hasher = hashlib.sha256()
    hasher.update(ticker.encode("ascii"))
    hasher.update("deployable_v5".encode("ascii"))
    hasher.update("|".join(DEPLOYABLE_FEATURE_COLUMNS_V5).encode("utf-8"))
    hasher.update(
        "|".join(pd.Timestamp(value).date().isoformat() for value in dates).encode("ascii")
    )
    hasher.update(np.asarray(features, dtype=np.float64).tobytes())
    hasher.update(np.asarray(baseline, dtype=np.float64).tobytes())
    if baseline_paths:
        for name in sorted(baseline_paths):
            hasher.update(name.encode("ascii"))
            hasher.update(np.asarray(baseline_paths[name], dtype=np.float64).tobytes())
    return hasher.hexdigest()


def _baseline_variance_paths(feature_row: pd.Series, har_path: np.ndarray) -> dict[str, np.ndarray]:
    """Build cumulative variance paths for the active train-free baselines."""
    days = np.arange(1, max(VOLATILITY_HORIZONS) + 1, dtype=np.float64)
    paths: dict[str, np.ndarray] = {}
    source = {
        "riskmetrics_ewma_c2c": "EWMA_Var",
        "rolling_c2c_5": "Vol_C2C_5",
        "rolling_c2c_20": "Vol_C2C_20",
        "rolling_c2c_60": "Vol_C2C_60",
    }
    for name, column in source.items():
        value = float(feature_row[column])
        if not np.isfinite(value) or value <= 0:
            continue
        daily_variance = value if name == "riskmetrics_ewma_c2c" else value**2
        paths[name] = np.maximum(daily_variance * days, 1e-12)
    har_values = np.asarray(har_path, dtype=np.float64).reshape(-1)
    if (
        har_values.shape == days.shape
        and np.isfinite(har_values).all()
        and (har_values > 0).all()
        and not np.any(np.diff(har_values) < -1e-12)
    ):
        paths["causal_log_har"] = har_values
    rolling = [
        paths[name]
        for name in ("rolling_c2c_5", "rolling_c2c_20", "rolling_c2c_60")
        if name in paths
    ]
    if len(rolling) == 3:
        paths["rolling_c2c_multiscale"] = np.exp(
            np.mean(np.log(np.maximum(np.stack(rolling), 1e-12)), axis=0)
        )
    return paths


def build_volatility_inference_snapshot(ticker: str) -> VolatilityInferenceSnapshot:
    """Build the latest model input using observations available through the origin."""
    symbol = ticker.upper().strip()
    if not symbol:
        raise ValueError("volatility ticker is required")
    raw = _download_ohlcv(symbol)
    features = build_features_v5(raw)
    proxy = realized_variance_proxies(raw)["RV_C2C"].clip(lower=1e-12)
    har = causal_log_har_forecasts(proxy, VOLATILITY_HORIZONS)
    har_path = causal_log_har_forecasts(proxy, VOLATILITY_PATH_HORIZONS)[-1]
    feature_matrix = features[list(DEPLOYABLE_FEATURE_COLUMNS_V5)].to_numpy(dtype=np.float64)
    last = len(raw) - 1
    if last < VOLATILITY_WINDOW_SIZE - 1:
        raise ValueError("market history is too short for volatility inference")
    window = feature_matrix[last - VOLATILITY_WINDOW_SIZE + 1 : last + 1]
    baseline = har[last]
    baseline_paths = _baseline_variance_paths(features.iloc[last], har_path)
    if "causal_log_har" not in baseline_paths:
        raise ValueError("latest market history cannot form a finite HAR variance path")
    if not np.isfinite(window).all() or not np.isfinite(baseline).all() or (baseline <= 0).any():
        raise ValueError("latest market history cannot form a finite volatility snapshot")
    history = raw.iloc[max(0, last - 89) : last + 1]
    dates = raw.index[last - VOLATILITY_WINDOW_SIZE + 1 : last + 1]
    future_dates, _ = future_trading_dates(
        symbol,
        pd.Timestamp(raw.index[last]),
        max(VOLATILITY_HORIZONS),
    )
    return VolatilityInferenceSnapshot(
        ticker=symbol,
        snapshot_id=_snapshot_identity(symbol, dates, window, baseline, baseline_paths),
        origin_date=pd.Timestamp(raw.index[last]).date().isoformat(),
        origin_close=float(raw["Close"].iloc[last]),
        feature_names=DEPLOYABLE_FEATURE_COLUMNS_V5,
        features=window.astype(np.float32),
        causal_har_variance=baseline.astype(np.float32),
        baseline_candidates=_baseline_candidates(features.iloc[last], baseline),
        baseline_variance_paths=baseline_paths,
        historical_dates=tuple(pd.Timestamp(value).date().isoformat() for value in history.index),
        historical_prices=history["Close"].to_numpy(dtype=np.float64),
        future_dates=tuple(future_dates),
    )
