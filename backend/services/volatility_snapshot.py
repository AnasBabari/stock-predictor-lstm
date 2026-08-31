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

    def __post_init__(self) -> None:
        if self.features.shape != (VOLATILITY_WINDOW_SIZE, len(self.feature_names)):
            raise ValueError("volatility feature window has an incompatible shape")
        if self.causal_har_variance.shape != (len(VOLATILITY_HORIZONS),):
            raise ValueError("volatility HAR baseline has an incompatible shape")
        if not np.isfinite(self.features).all():
            raise ValueError("volatility feature window must be finite")
        if not np.isfinite(self.causal_har_variance).all() or (self.causal_har_variance <= 0).any():
            raise ValueError("volatility HAR baseline must be finite and positive")
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
    return hasher.hexdigest()


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
    feature_matrix = features[list(DEPLOYABLE_FEATURE_COLUMNS_V5)].to_numpy(dtype=np.float64)
    last = len(raw) - 1
    if last < VOLATILITY_WINDOW_SIZE - 1:
        raise ValueError("market history is too short for volatility inference")
    window = feature_matrix[last - VOLATILITY_WINDOW_SIZE + 1 : last + 1]
    baseline = har[last]
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
        snapshot_id=_snapshot_identity(symbol, dates, window, baseline),
        origin_date=pd.Timestamp(raw.index[last]).date().isoformat(),
        origin_close=float(raw["Close"].iloc[last]),
        feature_names=DEPLOYABLE_FEATURE_COLUMNS_V5,
        features=window.astype(np.float32),
        causal_har_variance=baseline.astype(np.float32),
        baseline_candidates=_baseline_candidates(features.iloc[last], baseline),
        historical_dates=tuple(pd.Timestamp(value).date().isoformat() for value in history.index),
        historical_prices=history["Close"].to_numpy(dtype=np.float64),
        future_dates=tuple(future_dates),
    )
