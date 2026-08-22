"""Stationary schema v5 feature groups for the global panel (slice 5).

Every feature is causal: row t may only use information from rows <= t.
Groups follow overhaul spec §5.2: return structure, volatility structure,
liquidity, and history-only regime labels. Cross-sectional ranks operate at
panel level (same-date across tickers) and never use future membership.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

DEPLOYABLE_SCHEMA_VERSION = "deployable_v5"
RESEARCH_SCHEMA_VERSION = "research_v5"
EWMA_LAMBDA = 0.94
REGIME_LOOKBACK = 126

RETURN_STRUCTURE_COLUMNS = [
    "Return_1D",
    "Return_5D",
    "Return_10D",
    "Return_20D",
    "Overnight_Return",
    "OpenToClose_Return",
    "HL_Range_Log",
    "Downside_Semivar_20",
    "Realized_Skew_20",
    "Realized_Kurt_20",
    "Drawdown_From_Peak",
    "Up_Streak",
    "Down_Streak",
]
VOLATILITY_COLUMNS = [
    "Vol_C2C_5",
    "Vol_C2C_10",
    "Vol_C2C_20",
    "Vol_C2C_60",
    "EWMA_Var",
    "Vol_Of_Vol_20",
    "Vol_Percentile_252",
]
LIQUIDITY_COLUMNS = [
    "Log_Dollar_Volume",
    "Dollar_Volume_Median_20",
    "Volume_Surprise",
    "Amihud_Illiquidity_20",
    "Zero_Return_Fraction_20",
    "Stale_Price_Flag",
]
REGIME_COLUMNS = [
    "Regime_Trend",
    "Regime_Volatility",
    "Regime_Liquidity",
]

DEPLOYABLE_FEATURE_COLUMNS_V5: tuple[str, ...] = tuple(
    RETURN_STRUCTURE_COLUMNS + VOLATILITY_COLUMNS + LIQUIDITY_COLUMNS
)

RESEARCH_FEATURE_COLUMNS_V5: tuple[str, ...] = tuple(
    RETURN_STRUCTURE_COLUMNS + VOLATILITY_COLUMNS + LIQUIDITY_COLUMNS + REGIME_COLUMNS
)

FEATURE_COLUMNS_V5 = list(RESEARCH_FEATURE_COLUMNS_V5)


@dataclass(frozen=True)
class DeployableFeatureContract:
    schema_version: str = DEPLOYABLE_SCHEMA_VERSION
    feature_names: tuple[str, ...] = DEPLOYABLE_FEATURE_COLUMNS_V5
    window_size: int = 60
    transformation_version: str = "v5_robust"
    target_version: str = "cumulative_three_way_v2"

    def validate(
        self,
        names: list[str] | tuple[str, ...],
        schema_ver: str,
        transformation_ver: str,
    ) -> None:
        if schema_ver != self.schema_version:
            raise ValueError(
                f"Schema version mismatch: expected {self.schema_version}, got {schema_ver}"
            )
        if transformation_ver != self.transformation_version:
            raise ValueError(
                f"Transformation version mismatch: expected {self.transformation_version}, "
                f"got {transformation_ver}"
            )
        if tuple(names) != self.feature_names:
            raise ValueError(
                f"Feature contract mismatch: expected {len(self.feature_names)} features in order, "
                f"got {len(names)} features."
            )


@dataclass(frozen=True)
class DeployableRobustScaler:
    median: list[float]
    iqr: list[float]
    feature_names: tuple[str, ...] = DEPLOYABLE_FEATURE_COLUMNS_V5
    schema_version: str = DEPLOYABLE_SCHEMA_VERSION

    @classmethod
    def fit(
        cls,
        rows: np.ndarray | pd.DataFrame,
        feature_names: tuple[str, ...] = DEPLOYABLE_FEATURE_COLUMNS_V5,
    ) -> DeployableRobustScaler:
        arr = np.asarray(rows, dtype=float)
        if arr.ndim != 2 or arr.shape[-1] != len(feature_names):
            raise ValueError(
                f"Scaler input shape mismatch: expected (*, {len(feature_names)}), got {arr.shape}"
            )
        # Robust statistics per column: median and IQR
        med = [float(np.nanmedian(arr[:, c])) for c in range(arr.shape[-1])]
        q75 = [float(np.nanpercentile(arr[:, c], 75)) for c in range(arr.shape[-1])]
        q25 = [float(np.nanpercentile(arr[:, c], 25)) for c in range(arr.shape[-1])]
        iqr = [float(max(q75[c] - q25[c], 1e-12)) for c in range(arr.shape[-1])]
        return cls(median=med, iqr=iqr, feature_names=feature_names)

    def transform(self, rows: np.ndarray) -> np.ndarray:
        arr = np.asarray(rows, dtype=float)
        if arr.shape[-1] != len(self.median):
            raise ValueError(
                f"Feature dimension mismatch: expected {len(self.median)}, got {arr.shape[-1]}"
            )
        med = np.asarray(self.median, dtype=float)
        spread = np.asarray(self.iqr, dtype=float)
        return (arr - med) / spread

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "feature_names": list(self.feature_names),
            "median": list(self.median),
            "iqr": list(self.iqr),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeployableRobustScaler:
        return cls(
            median=[float(x) for x in data["median"]],
            iqr=[float(x) for x in data["iqr"]],
            feature_names=tuple(data["feature_names"]),
            schema_version=str(data.get("schema_version", DEPLOYABLE_SCHEMA_VERSION)),
        )


def _log(x: pd.Series) -> pd.Series:
    return np.log(x.where(x > 0))


def add_return_structure(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    log_close = _log(close)
    df["Return_1D"] = log_close.diff()
    for n in (5, 10, 20):
        df[f"Return_{n}D"] = log_close.diff(n)
    df["Overnight_Return"] = _log(df["Open"]) - log_close.shift(1)
    df["OpenToClose_Return"] = log_close - _log(df["Open"])
    df["HL_Range_Log"] = _log(df["High"]) - _log(df["Low"])

    ret = df["Return_1D"]
    downside = ret.where(ret < 0, 0.0) ** 2
    df["Downside_Semivar_20"] = np.sqrt(downside.rolling(20).mean())
    df["Realized_Skew_20"] = ret.rolling(20).skew()
    df["Realized_Kurt_20"] = ret.rolling(20).kurt()

    running_peak = close.cummax()
    df["Drawdown_From_Peak"] = _log(close) - _log(running_peak)

    up = (ret > 0).astype(int)
    down = (ret < 0).astype(int)
    streak_up = up.groupby((~up.astype(bool)).cumsum()).cumsum()
    streak_down = down.groupby((~down.astype(bool)).cumsum()).cumsum()
    df["Up_Streak"] = streak_up.where(up == 1, 0)
    df["Down_Streak"] = streak_down.where(down == 1, 0)
    return df


def add_volatility_structure(df: pd.DataFrame) -> pd.DataFrame:
    ret = df["Return_1D"]
    for n in (5, 10, 20, 60):
        df[f"Vol_C2C_{n}"] = ret.rolling(n).std()
    ewma_var = ret.pow(2).ewm(alpha=1 - EWMA_LAMBDA, adjust=False).mean()
    df["EWMA_Var"] = ewma_var
    df["Vol_Of_Vol_20"] = df["Vol_C2C_5"].rolling(20).std()
    vol_c2c_20 = df["Vol_C2C_20"]
    df["Vol_Percentile_252"] = vol_c2c_20.rolling(252, min_periods=60).rank(pct=True)
    return df


def add_liquidity(df: pd.DataFrame) -> pd.DataFrame:
    dollar_volume = df["Close"] * df["Volume"]
    df["Log_Dollar_Volume"] = _log(dollar_volume)
    median_dv = dollar_volume.rolling(20).median()
    df["Dollar_Volume_Median_20"] = median_dv
    df["Volume_Surprise"] = dollar_volume / median_dv - 1
    ret_abs = df["Return_1D"].abs()
    df["Amihud_Illiquidity_20"] = (ret_abs / dollar_volume.replace(0, np.nan)).rolling(
        20
    ).mean() * 1e9
    flat = (df["Return_1D"].abs() < 1e-9).astype(int)
    df["Zero_Return_Fraction_20"] = flat.rolling(20).mean()
    df["Stale_Price_Flag"] = flat
    return df


def _terciles(series: pd.Series, lookback: int, min_periods: int) -> tuple[pd.Series, pd.Series]:
    rolled = series.rolling(lookback, min_periods=min_periods)
    return rolled.quantile(1 / 3), rolled.quantile(2 / 3)


def _tercile_label(
    value: pd.Series, lo: pd.Series, hi: pd.Series, low: str, high: str, mid: str
) -> pd.Series:
    label = pd.Series(mid, index=value.index, dtype=object)
    label[value <= lo] = low
    label[value >= hi] = high
    label[value.isna() | lo.isna()] = np.nan
    return label


def add_regime_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Trend/vol/liquidity labels from trailing-126-session terciles only."""
    ret_20 = df["Return_1D"].rolling(20).sum()
    lo_t, hi_t = _terciles(ret_20, REGIME_LOOKBACK, 60)
    df["Regime_Trend"] = _tercile_label(ret_20, lo_t, hi_t, "bear", "bull", "range")

    vol = df["Vol_C2C_20"]
    lo_v, hi_v = _terciles(vol, REGIME_LOOKBACK, 60)
    df["Regime_Volatility"] = _tercile_label(vol, lo_v, hi_v, "calm", "stressed", "normal")

    dv_med = df["Log_Dollar_Volume"]
    lo_l, hi_l = _terciles(dv_med, REGIME_LOOKBACK, 60)
    df["Regime_Liquidity"] = _tercile_label(dv_med, lo_l, hi_l, "illiquid", "liquid", "normal")
    return df


def build_features_v5(df: pd.DataFrame) -> pd.DataFrame:
    """Causal v5 features aligned to the input index; NaNs retained upstream.

    Warm-up rows (rolling windows not yet satisfied) stay NaN — the fold
    builder drops them per-feature-group so ablations can compare fairly.
    """
    out = df.copy()
    out = add_return_structure(out)
    out = add_volatility_structure(out)
    out = add_liquidity(out)
    out = add_regime_labels(out)
    missing = [c for c in FEATURE_COLUMNS_V5 if c not in out.columns]
    if missing:
        raise RuntimeError(f"feature builder produced missing columns: {missing}")
    return out


def add_cross_sectional_ranks(
    panels: dict[str, pd.DataFrame],
    columns: list[str],
) -> dict[str, pd.DataFrame]:
    """Same-date cross-sectional percentile ranks across tickers (causal).

    Rank uses only values observed on the SAME date for OTHER tickers — no
    future information, no future universe membership. Output is pct-rank in
    (0, 1] appended as `<col>_XSRank`.
    """
    if not panels:
        return {}
    stacked = {ticker: frame[columns].copy() for ticker, frame in panels.items()}
    combined = pd.concat(stacked, names=["Ticker", "Date"])
    ranked = combined.groupby(level="Date").rank(pct=True)
    result: dict[str, pd.DataFrame] = {}
    for ticker in panels:
        ranks = ranked.xs(ticker, level="Ticker")
        ranks.columns = [f"{col}_XSRank" for col in ranks.columns]
        result[ticker] = panels[ticker].join(ranks)
    return result
