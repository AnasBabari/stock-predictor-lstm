"""Promoted G3 GPU-volatility serving (ONNX, no request-time training).

G3 is a global XGBoost correction to rolling volatility, validated on the
held-out test panel (see artifacts/gpu_rolling_origin_v1/test_report.json).
For horizon H with rolling base B: Vhat = B * exp(delta), delta from the
packaged ONNX graph plus log(B) added outside it. Forecasts are always
positive; any failure degrades to the rolling baseline explicitly.

Feature definitions mirror research/volatility_structure exactly; the parity
test pins them bit-for-bit. This module must never import research code on
the production request path.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

G3_MODEL_VERSION = "g3-gpu-xgb-v1"
G3_FEATURE_SET_VERSION = "vol-gpu-panel-v1"
G3_HORIZONS = (5, 10, 20)
G3_FEATURES: tuple[str, ...] = (
    "rv_5",
    "rv_10",
    "rv_20",
    "rv_60",
    "yz_now",
    "yz_chg_5",
    "yz_chg_20",
    "yz_roll_ratio",
    "range_park_20",
    "range_gk_20",
    "range_rs_20",
    "range_yz_20",
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "downside_sq",
    "upside_sq",
    "neg_indicator",
    "har_log_daily",
    "har_log_weekly",
    "har_log_monthly",
    "vol_of_vol_20",
)
_FOUR_LOG_2 = 4.0 * np.log(2.0)
_TWO_LOG_2_MINUS_1 = 2.0 * np.log(2.0) - 1.0
_SQRT_2_OVER_PI = float(np.sqrt(2.0 / np.pi))

# Frozen test-panel evidence for user-facing provenance (do not edit by hand;
# regenerate from artifacts/gpu_rolling_origin_v1/test_report.json).
G3_TEST_EVIDENCE = {
    5: {"delta_qlike": 0.23926, "relative_improvement": 0.2769, "p_two_sided": 1.04e-19},
    10: {"delta_qlike": 0.23752, "relative_improvement": 0.3450, "p_two_sided": 1.19e-15},
    20: {"delta_qlike": 0.23665, "relative_improvement": 0.4220, "p_two_sided": 1.86e-11},
}

_sessions: dict[int, Any] = {}
_sessions_lock = threading.Lock()


def _model_dir() -> Path:
    import os

    override = (os.getenv("VOLATILITY_G3_MODEL_DIR") or "").strip()
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[1] / "volatility_models"


def _range_variances(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray):
    o = np.asarray(open_, dtype=float)
    h = np.asarray(high, dtype=float)
    lo = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    valid = (
        np.isfinite(o)
        & np.isfinite(h)
        & np.isfinite(lo)
        & np.isfinite(c)
        & (o > 0.0)
        & (h > 0.0)
        & (lo > 0.0)
        & (c > 0.0)
        & (h >= lo)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        log_hl = np.log(h / lo)
        log_co = np.log(c / o)
        parkinson = log_hl**2 / _FOUR_LOG_2
        garman_klass = 0.5 * log_hl**2 - _TWO_LOG_2_MINUS_1 * log_co**2
        rogers_satchell = np.log(h / c) * np.log(h / o) + np.log(lo / c) * np.log(lo / o)
    parkinson[~valid] = np.nan
    garman_klass[~valid] = np.nan
    rogers_satchell[~valid] = np.nan
    return parkinson, garman_klass, rogers_satchell


def _yang_zhang_var(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 22
) -> np.ndarray:
    # Mirrors research range_estimators, including its validity mask: only
    # finite, positive prints with high >= low participate; anything else
    # yields NaN rows rather than silently shifting the surface.
    o = np.asarray(open_, dtype=float)
    h = np.asarray(high, dtype=float)
    lo = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    valid = (
        np.isfinite(o)
        & np.isfinite(h)
        & np.isfinite(lo)
        & np.isfinite(c)
        & (o > 0.0)
        & (h > 0.0)
        & (lo > 0.0)
        & (c > 0.0)
        & (h >= lo)
    )
    out = np.full(len(close), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        prev_close = np.roll(close, 1)
        overnight = np.log(open_ / prev_close)
        open_close = np.log(close / open_)
        log_hc = np.log(high / close)
        log_ho = np.log(high / open_)
        log_lc = np.log(low / close)
        log_lo = np.log(low / open_)
        rs_day = log_hc * log_ho + log_lc * log_lo
    overnight[0] = np.nan
    overnight[~valid] = np.nan
    open_close[~valid] = np.nan
    rs_day[~valid] = np.nan
    frame = pd.DataFrame({"o": overnight, "c": open_close, "rs": rs_day})
    var_o = frame["o"].rolling(window, min_periods=window).var(ddof=1).to_numpy()
    var_c = frame["c"].rolling(window, min_periods=window).var(ddof=1)
    mean_rs = frame["rs"].rolling(window, min_periods=window).mean().to_numpy()
    n = float(window)
    k = 0.34 / (1.34 + (n + 1.0) / (n - 1.0))
    with np.errstate(invalid="ignore"):
        out = var_o + k * var_c.to_numpy() + (1.0 - k) * mean_rs
    return out


def _har_log_components(rv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trailing daily/weekly/monthly log components (production row logic).

    Production takes log AFTER averaging (log of mean variance), not the
    mean of logs: weekly(t) = log(mean(RV[t-4..t])), monthly(t) =
    log(mean(RV[t-21..t])). Outputs are NaN before index 21, matching the
    research wrapper exactly.
    """
    safe = np.maximum(np.asarray(rv, dtype=float), 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        logged = np.log(safe)
        weekly = np.log(pd.Series(safe).rolling(5, min_periods=5).mean().to_numpy())
        monthly = np.log(pd.Series(safe).rolling(22, min_periods=22).mean().to_numpy())
    # All-or-nothing validity exactly like the research wrapper.
    valid_window = pd.Series(np.isfinite(safe)).rolling(22, min_periods=22).sum().to_numpy() == 22
    daily = np.where(valid_window, logged, np.nan)
    weekly = np.where(valid_window, weekly, np.nan)
    monthly = np.where(valid_window, monthly, np.nan)
    daily[:21] = np.nan
    weekly[:21] = np.nan
    monthly[:21] = np.nan
    return daily, weekly, monthly


def build_g3_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Frozen 22-column G3 feature block for an OHLCV frame (trailing only)."""
    open_ = frame["Open"].to_numpy(dtype=float)
    high = frame["High"].to_numpy(dtype=float)
    low = frame["Low"].to_numpy(dtype=float)
    close = frame["Close"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        logret = np.log(close / np.roll(close, 1))
    logret[0] = np.nan
    squared = logret**2
    downside = np.where(logret < 0.0, squared, 0.0)
    downside[~np.isfinite(logret)] = np.nan
    upside = np.where(logret > 0.0, squared, 0.0)
    upside[~np.isfinite(logret)] = np.nan
    neg_indicator = np.where(np.isfinite(logret), np.where(logret < 0.0, 1.0, 0.0), np.nan)
    park, gk, rs = _range_variances(open_, high, low, close)
    yz = _yang_zhang_var(open_, high, low, close)
    # Production HAR convention: ret[0] = 0, so RV[0] = 0.0 (not NaN).
    har_rv = squared.copy()
    if len(har_rv):
        har_rv[0] = 0.0
    har_d, har_w, har_m = _har_log_components(har_rv)
    out = pd.DataFrame(index=frame.index)
    out["rv_5"] = pd.Series(squared).rolling(5, min_periods=5).mean().to_numpy()
    out["rv_10"] = pd.Series(squared).rolling(10, min_periods=10).mean().to_numpy()
    out["rv_20"] = pd.Series(squared).rolling(20, min_periods=20).mean().to_numpy()
    out["rv_60"] = pd.Series(squared).rolling(60, min_periods=60).mean().to_numpy()
    out["yz_now"] = yz
    out["yz_chg_5"] = yz - pd.Series(yz).rolling(5, min_periods=5).mean().to_numpy()
    out["yz_chg_20"] = yz - pd.Series(yz).rolling(20, min_periods=20).mean().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        out["yz_roll_ratio"] = yz / np.where(
            out["rv_20"].to_numpy() == 0.0, np.nan, out["rv_20"].to_numpy()
        )
    for name, series in (
        ("range_park_20", park),
        ("range_gk_20", gk),
        ("range_rs_20", rs),
        ("range_yz_20", yz),
    ):
        out[name] = pd.Series(series).rolling(20, min_periods=20).mean().to_numpy()
    out["ret_1d"] = logret
    close_series = pd.Series(close, index=frame.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["ret_5d"] = np.log(close_series / close_series.shift(5)).to_numpy()
        out["ret_20d"] = np.log(close_series / close_series.shift(20)).to_numpy()
    out["downside_sq"] = downside
    out["upside_sq"] = upside
    out["neg_indicator"] = neg_indicator
    out["har_log_daily"] = har_d
    out["har_log_weekly"] = har_w
    out["har_log_monthly"] = har_m
    out["vol_of_vol_20"] = (
        pd.Series(np.sqrt(np.maximum(squared, 0.0)))
        .rolling(20, min_periods=20)
        .std(ddof=1)
        .to_numpy()
    )
    return out[list(G3_FEATURES)]


class G3UnavailableError(RuntimeError):
    """Packaged G3 artifacts cannot serve this request; use rolling fallback."""


def _load_session(horizon: int):
    import onnxruntime as ort

    with _sessions_lock:
        session = _sessions.get(horizon)
        if session is None:
            path = _model_dir() / f"g3_h{horizon}.onnx"
            meta_path = _model_dir() / f"g3_h{horizon}.meta.json"
            if not path.is_file() or not meta_path.is_file():
                raise G3UnavailableError(f"packaged G3 model missing for horizon {horizon}")
            session = ort.InferenceSession(str(path))
            _sessions[horizon] = session
        return session


def g3_cumulative_variance(frame: pd.DataFrame, horizon: int) -> float:
    """G3 cumulative variance forecast from the latest session (always positive)."""
    if horizon not in G3_HORIZONS:
        raise ValueError(f"G3 horizon must be one of {list(G3_HORIZONS)}")
    features = build_g3_features(frame)
    row = features.iloc[[-1]].to_numpy(dtype=np.float32)
    if not np.isfinite(row).all():
        raise G3UnavailableError("latest session lacks complete G3 features")
    base = panel_rolling_base(frame, horizon)
    session = _load_session(horizon)
    raw = np.asarray(session.run(None, {"input": row})[0]).reshape(-1)[0]
    # z = delta + log(B); Vhat = exp(z) = B * exp(delta). The base enters
    # exactly once, inside the log-margin.
    log_variance = min(max(float(raw), -30.0), 10.0) + np.log(max(base, 1e-12))
    variance = float(np.exp(log_variance))
    if not np.isfinite(variance) or variance <= 0:
        raise G3UnavailableError("G3 produced a non-positive variance")
    return float(variance)


def panel_rolling_base(frame: pd.DataFrame, horizon: int) -> float:
    """Trailing-20d variance times horizon (the frozen G0 definition)."""
    close = frame["Close"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        logret = np.log(close / np.roll(close, 1))
    window = logret[-20:]
    if len(window) < 20 or not np.isfinite(window).all():
        raise G3UnavailableError("insufficient history for the rolling base")
    return float(np.var(window, ddof=1) * horizon)
