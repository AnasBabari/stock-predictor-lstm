"""Causal GPU-panel dataset for the volatility study (GPU branch only).

One row per (ticker, session) with frozen features, forward targets, and
the rolling-champion base forecasts. Every column at origin ``t`` depends
only on observations through ``t``. No sector column exists: the OHLCV
cache carries no sector metadata, and inventing one is out of scope
(recorded here instead of silently omitted).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.volatility_structure import panel
from research.volatility_structure.asymmetry_features import asymmetry_features
from research.volatility_structure.har_wrapper import har_log_components
from research.volatility_structure.range_estimators import range_variances

FEATURE_COLUMNS: tuple[str, ...] = (
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
    "market_US",
    "market_UK",
    "spy_rv_20",
    "spy_ret_20",
)
FEATURE_SCHEMA_VERSION = "vol-gpu-panel-v1"


def build_ticker_features(
    frame: pd.DataFrame, market: str, spy: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Frozen feature set for one ticker's OHLCV frame (Close/High/Low/Open)."""
    close = frame["Close"].astype(float)
    logret = np.log(close / close.shift(1))
    sq = logret**2
    out = pd.DataFrame(index=frame.index)
    out["rv_5"] = sq.rolling(5, min_periods=5).mean()
    out["rv_10"] = sq.rolling(10, min_periods=10).mean()
    out["rv_20"] = sq.rolling(20, min_periods=20).mean()
    out["rv_60"] = sq.rolling(60, min_periods=60).mean()

    estimated = range_variances(frame["Open"], frame["High"], frame["Low"], close)
    out["yz_now"] = estimated["yang_zhang_var"]
    out["yz_chg_5"] = (
        estimated["yang_zhang_var"] - estimated["yang_zhang_var"].rolling(5, min_periods=5).mean()
    )
    out["yz_chg_20"] = (
        estimated["yang_zhang_var"] - estimated["yang_zhang_var"].rolling(20, min_periods=20).mean()
    )
    out["yz_roll_ratio"] = estimated["yang_zhang_var"] / out["rv_20"].replace(0.0, np.nan)
    block = panel.range_block_features(estimated)
    out["range_park_20"] = block["range_park_20"]
    out["range_gk_20"] = block["range_gk_20"]
    out["range_rs_20"] = block["range_rs_20"]
    out["range_yz_20"] = block["range_yz_20"]

    out["ret_1d"] = logret
    out["ret_5d"] = np.log(close / close.shift(5))
    out["ret_20d"] = np.log(close / close.shift(20))
    asym = asymmetry_features(close)
    out["downside_sq"] = asym["downside_sq"]
    out["upside_sq"] = asym["upside_sq"]
    out["neg_indicator"] = asym["neg_indicator"]

    har = har_log_components(close)
    out["har_log_daily"] = har["har_log_daily"]
    out["har_log_weekly"] = har["har_log_weekly"]
    out["har_log_monthly"] = har["har_log_monthly"]

    out["vol_of_vol_20"] = np.sqrt(sq).rolling(20, min_periods=20).std(ddof=1)
    out["market_US"] = 1.0 if market == "US" else 0.0
    out["market_UK"] = 1.0 if market == "UK" else 0.0
    if spy is not None:
        aligned = spy.reindex(frame.index)
        spy_logret = np.log(aligned["SPY"] / aligned["SPY"].shift(1))
        out["spy_rv_20"] = (spy_logret**2).rolling(20, min_periods=20).mean()
        out["spy_ret_20"] = np.log(aligned["SPY"] / aligned["SPY"].shift(20))
    else:
        out["spy_rv_20"] = np.nan
        out["spy_ret_20"] = np.nan
    return out[list(FEATURE_COLUMNS)]


def qlike_objective():
    """Custom XGBoost objective for L(z) = y*exp(-z) + z (QLIKE up to constants).

    Gradient g = 1 - y*exp(-z), Hessian h = y*exp(-z) > 0 for y > 0:
    strictly convex in the log-forecast z. Raw margin z is clipped to
    [-30, 10] for numerical safety; labels are floored by the caller.
    """

    def objective(preds: np.ndarray, dtrain) -> tuple[np.ndarray, np.ndarray]:
        labels = dtrain.get_label()
        z = np.clip(np.asarray(preds, dtype=np.float64), -30.0, 10.0)
        scaled = np.maximum(labels, panel.QLIKE_FLOOR) * np.exp(-z)
        grad = 1.0 - scaled
        hess = np.maximum(scaled, 1e-6)
        return grad, hess

    return objective
