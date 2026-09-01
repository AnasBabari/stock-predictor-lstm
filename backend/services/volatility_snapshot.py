"""Deployable, causal inputs for the active volatility baseline service."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from calendars import future_trading_dates
from data_pipeline import _download_ohlcv
from panel.features import DEPLOYABLE_FEATURE_COLUMNS_V5, build_features_v5
from panel.volatility import causal_log_har_forecasts, realized_variance_proxies

VOLATILITY_HORIZONS = (1, 3, 5, 7, 14, 30)
VOLATILITY_WINDOW_SIZE = 60
VOLATILITY_PATH_HORIZONS = tuple(range(1, max(VOLATILITY_HORIZONS) + 1))


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
            for name, path in self.baseline_variance_paths.items():
                values = np.asarray(path, dtype=np.float64)
                if values.shape != (max(VOLATILITY_HORIZONS),):
                    raise ValueError(f"volatility baseline path has an incompatible shape: {name}")
                if not np.isfinite(values).all() or (values <= 0).any():
                    raise ValueError(
                        f"volatility baseline path must be finite and positive: {name}"
                    )
                if np.any(np.diff(values) < -1e-12):
                    raise ValueError(f"volatility baseline path must be cumulative: {name}")
        if self.origin_close <= 0 or not np.isfinite(self.origin_close):
            raise ValueError("volatility origin close must be finite and positive")
        if len(self.historical_dates) != len(self.historical_prices):
            raise ValueError("volatility chart history is misaligned")
        if any(not isinstance(value, str) for value in self.future_dates):
            raise ValueError("volatility future dates must be strings")
        if any(a >= b for a, b in zip(self.future_dates, self.future_dates[1:], strict=False)):
            raise ValueError("volatility future dates must be chronological")


def _baseline_candidates(feature_row: pd.Series, har: np.ndarray) -> dict[str, np.ndarray]:
    horizon_scale = np.asarray(VOLATILITY_HORIZONS, dtype=np.float64)
    candidates: dict[str, np.ndarray] = {"causal_log_har": har.astype(np.float64)}
    source = {
        "riskmetrics_ewma_c2c": ("EWMA_Var", False),
        "rolling_c2c_5": ("Vol_C2C_5", True),
        "rolling_c2c_20": ("Vol_C2C_20", True),
        "rolling_c2c_60": ("Vol_C2C_60", True),
    }
    for family, (name, square) in source.items():
        value = float(feature_row[name])
        daily_variance = value**2 if square else value
        forecast = np.maximum(daily_variance * horizon_scale, 1e-12)
        if np.isfinite(forecast).all():
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
    """Build cumulative variance paths for the active train-free baselines.

    The previous implementation reconstructed every intermediate price band
    from the terminal variance.  That is only coherent for a linear
    random-walk-in-variance forecast.  Keeping a daily cumulative path makes
    the interpolation rule explicit and lets recursive HAR forecasts retain
    their horizon shape.
    """

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
        # The recursive HAR output is already a cumulative sum of positive
        # one-step variances.  Reject a malformed path rather than silently
        # repairing it, so a provider/model drift cannot change the cone.
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
    # The active product target is close-to-close realised volatility. Keep the
    # serving baseline on that same proxy as the offline benchmark; the OHLC
    # range/overnight proxies remain research-only candidates.
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
