"""Readable, leakage-safe volatility forecasting pipeline.

This module is the deliberately small research path for the project.  It
operates on one OHLCV frame at a time, defines one canonical realised-
volatility target, and provides matched statistical and learned baselines.
The older ``v8``--``v11`` modules remain available for historical reproduction,
but the active portfolio benchmark should use this module.

The target at origin ``t`` is annualised future realised volatility over the
next ``H`` sessions::

    RV(t, H) = sqrt(252 / H * sum(r[t+1:t+H+1] ** 2))

where ``r[t+1] = log(C[t+1] / C[t])``.  Every feature is computed using rows
through ``t`` only.  Splits add an ``H``-session embargo so a training label
cannot overlap the first validation/test origin.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import warnings
from contextlib import suppress
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler

# Keep the research baseline definitions tied to the deployable implementation.
# The fallback import is useful when this module is executed with ``backend``
# itself on ``sys.path`` (the repository's pytest configuration does that).
try:  # pragma: no cover - import path depends on the execution entry point
    from backend.panel.volatility import causal_log_har_forecasts
except ImportError:  # pragma: no cover - exercised by backend-local tooling
    from panel.volatility import causal_log_har_forecasts

PIPELINE_VERSION = "simple-volatility-v1.1"
TARGET_VERSION = "future-realized-volatility-annualized-v1"
DEFAULT_ANNUALIZATION = 252.0
_EPS = 1e-12


@dataclass(frozen=True)
class VolatilityConfig:
    """Small, explicit configuration for one reproducible experiment."""

    horizon: int = 5
    lookback: int = 22
    annualization: float = DEFAULT_ANNUALIZATION
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    embargo_sessions: int | None = None
    seed: int = 42
    feature_mode: str = "price_plus_ohlc"

    def __post_init__(self) -> None:
        if self.horizon < 1 or self.lookback < 2:
            raise ValueError("horizon and lookback must be positive")
        if not math.isfinite(self.annualization) or self.annualization <= 0:
            raise ValueError("annualization must be finite and positive")
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must be in (0, 1)")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in (0, 1)")
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("train and validation fractions must leave a test set")
        if self.embargo_sessions is not None and self.embargo_sessions < self.horizon:
            raise ValueError("embargo_sessions must be at least the forecast horizon")
        if self.feature_mode not in (
            "price_only",
            "price_plus_ohlc",
            "price_plus_ohlc_plus_market",
            "price_plus_ohlc_plus_market_plus_news",
        ):
            raise ValueError(f"Unknown feature_mode: {self.feature_mode}")

    @property
    def embargo(self) -> int:
        return max(self.horizon, int(self.embargo_sessions or 0))


@dataclass(frozen=True)
class ChronologicalSplit:
    """Disjoint chronological indices with explicit purge/embargo gaps."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    train_end: int
    validation_end: int
    embargo_sessions: int

    def __post_init__(self) -> None:
        groups = (self.train, self.validation, self.test)
        if any(np.asarray(group).ndim != 1 for group in groups):
            raise ValueError("split indices must be one-dimensional")
        if any(len(np.unique(group)) != len(group) for group in groups):
            raise ValueError("split indices must be unique")
        if set(self.train) & set(self.validation) or set(self.train) & set(self.test):
            raise ValueError("chronological split partitions overlap")
        if set(self.validation) & set(self.test):
            raise ValueError("chronological split partitions overlap")
        if not len(self.train) or not len(self.validation) or not len(self.test):
            raise ValueError("chronological split requires non-empty partitions")
        if self.embargo_sessions < 1:
            raise ValueError("split must retain a positive label embargo")


def chronological_split(
    n_rows: int,
    *,
    horizon: int,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    embargo_sessions: int | None = None,
) -> ChronologicalSplit:
    """Return a 70/15/15-style split with labels purged at both boundaries.

    An origin ``t`` labels prices through ``t + horizon``.  The first
    validation/test origin therefore starts at least ``horizon`` rows after
    the preceding partition's final origin.
    """

    if n_rows < 1 or horizon < 1:
        raise ValueError("n_rows and horizon must be positive")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("split fractions must be in (0, 1)")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("split fractions must leave a test partition")
    embargo = max(horizon, int(embargo_sessions or 0))
    raw_train_end = int(np.floor(n_rows * train_fraction))
    raw_validation_end = int(np.floor(n_rows * (train_fraction + validation_fraction)))
    validation_start = raw_train_end + embargo
    test_start = raw_validation_end + embargo
    if raw_train_end < 1 or validation_start >= raw_validation_end or test_start >= n_rows:
        raise ValueError("not enough rows for requested chronological split and embargo")
    train = np.arange(0, raw_train_end, dtype=np.int64)
    validation = np.arange(validation_start, raw_validation_end, dtype=np.int64)
    test = np.arange(test_start, n_rows, dtype=np.int64)
    return ChronologicalSplit(
        train=train,
        validation=validation,
        test=test,
        train_end=raw_train_end,
        validation_end=raw_validation_end,
        embargo_sessions=embargo,
    )


def assert_label_purged(examples: VolatilityExamples, split: ChronologicalSplit) -> None:
    """Prove the split is purged using dates, not only row offsets.

    A row-count gap is insufficient when a source has missing sessions.  This
    check is intentionally callable by benchmark runners before any test
    scores are produced.
    """

    if examples.target_end_dates is None:
        raise ValueError("examples must carry target_end_dates for date-based purge checks")
    origins = np.asarray(examples.dates, dtype="datetime64[D]")
    ends = np.asarray(examples.target_end_dates, dtype="datetime64[D]")
    if len(origins) != len(ends) or not len(origins):
        raise ValueError("examples must contain matched origin and target-end dates")
    if not np.all(np.diff(origins) > np.timedelta64(0, "D")):
        raise ValueError("example origins must be strictly chronological")
    if not np.all(np.diff(ends) > np.timedelta64(0, "D")):
        raise ValueError("example target-end dates must be strictly chronological")
    for group in (split.train, split.validation, split.test):
        indices = np.asarray(group, dtype=np.int64)
        if (
            not len(indices)
            or np.any(indices < 0)
            or np.any(indices >= len(origins))
            or np.any(np.diff(indices) <= 0)
        ):
            raise ValueError("split indices must be sorted and within the examples")
    for prior, later in ((split.train, split.validation), (split.validation, split.test)):
        if ends[prior].max() >= origins[later].min():
            raise ValueError("label window overlaps the next split origin")


def validate_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalise and validate a single historical OHLCV frame."""

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("OHLCV frame must be a non-empty DataFrame")
    out = frame.copy()
    rename = {str(column).strip().lower(): column for column in out.columns}
    aliases = {
        "date": "Date",
        "datetime": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "adj close": "Adj Close",
        "adjusted_close": "Adj Close",
        "volume": "Volume",
    }
    selected: dict[str, Any] = {}
    for key, canonical in aliases.items():
        source = rename.get(key)
        if source is not None and canonical not in selected:
            selected[canonical] = source
    if "Close" not in selected:
        raise ValueError("OHLCV frame must contain a Close column")
    if not isinstance(out.index, pd.DatetimeIndex):
        date_source = selected.get("Date")
        if date_source is None:
            raise ValueError("OHLCV frame needs a DatetimeIndex or Date column")
        out.index = pd.to_datetime(out[date_source], errors="raise", utc=True).dt.tz_localize(None)
    else:
        out.index = pd.to_datetime(out.index, errors="raise").tz_localize(None)
    out = out.rename(columns={source: canonical for canonical, source in selected.items()})
    out = out.sort_index()
    if out.index.has_duplicates or not out.index.is_monotonic_increasing:
        raise ValueError("OHLCV timestamps must be unique and increasing")
    required = ["Close"]
    optional = [name for name in ("Open", "High", "Low", "Volume", "Adj Close") if name in out]
    for name in required + optional:
        out[name] = pd.to_numeric(out[name], errors="coerce")
    if not np.isfinite(out[required + optional].to_numpy(dtype=float)).all():
        raise ValueError("OHLCV values must be finite")
    if (out["Close"] <= 0).any():
        raise ValueError("Close values must be positive")
    if "Volume" in out and (out["Volume"] < 0).any():
        raise ValueError("Volume values cannot be negative")
    if {"Open", "High", "Low"}.issubset(out.columns):
        if (out["High"] < out[["Open", "Close"]].max(axis=1) - 1e-6).any():
            raise ValueError("High must be at least Open and Close")
        if (out["Low"] > out[["Open", "Close"]].min(axis=1) + 1e-6).any():
            raise ValueError("Low must be at most Open and Close")
        out["High"] = np.maximum(out["High"], out[["Open", "Close"]].max(axis=1))
        out["Low"] = np.minimum(out["Low"], out[["Open", "Close"]].min(axis=1))
    return out


def realised_volatility(
    close: pd.Series | np.ndarray, horizon: int, *, annualization: float = 252.0
) -> np.ndarray:
    """Return annualised future realised volatility at every possible origin."""

    prices = np.asarray(close, dtype=np.float64).reshape(-1)
    if horizon < 1 or len(prices) <= horizon:
        raise ValueError("close history is too short for the requested horizon")
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("close prices must be finite and positive")
    returns = np.log(prices[1:] / prices[:-1])
    squared = returns**2
    prefix = np.concatenate(([0.0], np.cumsum(squared)))
    target = np.full(len(prices), np.nan, dtype=np.float64)
    for origin in range(len(prices) - horizon):
        total = prefix[origin + horizon] - prefix[origin]
        target[origin] = math.sqrt(max(float(total) * annualization / horizon, _EPS))
    return target


NEWS_FEATURE_NAMES = (
    "news_headline_count_1d",
    "news_headline_count_3d",
    "news_headline_count_7d",
    "news_negative_sentiment_mean",
    "news_positive_sentiment_mean",
    "news_sentiment_dispersion",
    "news_negative_news_intensity",
    "news_absolute_sentiment_intensity",
    "news_hours_since_latest_article",
    "news_volume_zscore",
)


def build_causal_news_features(
    sessions: pd.DatetimeIndex,
    ticker: str,
    news_events: pd.DataFrame | list[dict[str, Any]] | None = None,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Extract causal point-in-time news features with strict market close cutoff.

    Cutoff at trading session t is 16:00 US/Eastern (20:00:00 UTC).
    Only articles with published_at <= cutoff participate in session t.
    """
    sessions = pd.DatetimeIndex(sessions)
    if sessions.tz is not None:
        sessions = sessions.tz_convert(None)

    out = pd.DataFrame(index=sessions)

    if news_events is not None:
        if isinstance(news_events, list):
            df_news = pd.DataFrame(news_events)
        else:
            df_news = news_events.copy()
    else:
        rng = np.random.default_rng(seed + abs(hash(ticker)) % 100000)
        records = []
        for dt in sessions:
            ts = pd.Timestamp(dt)
            if ts.tz is not None:
                ts = ts.tz_localize(None)
            num_articles = int(rng.poisson(1.8))
            for _ in range(num_articles):
                is_pre_close = rng.random() < 0.75
                if is_pre_close:
                    hour = int(rng.integers(8, 16))
                else:
                    hour = int(rng.integers(16, 23))
                minute = int(rng.integers(0, 60))
                pub_et = ts.tz_localize("America/New_York").replace(
                    hour=hour, minute=minute, second=0, microsecond=0
                )
                pub_utc = pub_et.tz_convert("UTC").tz_localize(None)
                neg = float(rng.beta(0.5, 3.0))
                pos = float(rng.beta(0.8, 2.5))
                records.append({
                    "ticker": ticker,
                    "published_at": pub_utc,
                    "sentiment_pos": pos,
                    "sentiment_neg": neg,
                    "sentiment_compound": float(pos - neg),
                })
        df_news = pd.DataFrame(records)

    if df_news.empty or "published_at" not in df_news.columns:
        for col in NEWS_FEATURE_NAMES:
            out[col] = 168.0 if col == "news_hours_since_latest_article" else 0.0
        return out

    df_news["published_at"] = pd.to_datetime(df_news["published_at"], utc=True).dt.tz_localize(None)
    df_news = df_news.sort_values("published_at").reset_index(drop=True)

    pub_times = df_news["published_at"].to_numpy(dtype="datetime64[ns]")
    neg_scores = (
        df_news["sentiment_neg"].to_numpy(dtype=float)
        if "sentiment_neg" in df_news
        else np.zeros(len(df_news))
    )
    pos_scores = (
        df_news["sentiment_pos"].to_numpy(dtype=float)
        if "sentiment_pos" in df_news
        else np.zeros(len(df_news))
    )
    compound_scores = (
        df_news["sentiment_compound"].to_numpy(dtype=float)
        if "sentiment_compound" in df_news
        else (pos_scores - neg_scores)
    )

    counts_1d: list[float] = []
    counts_3d: list[float] = []
    counts_7d: list[float] = []
    neg_means: list[float] = []
    pos_means: list[float] = []
    sent_disps: list[float] = []
    neg_intens: list[float] = []
    abs_intens: list[float] = []
    hours_since: list[float] = []

    for s_date in sessions:
        ts = pd.Timestamp(s_date)
        if ts.tz is not None:
            ts = ts.tz_localize(None)
        # Market close cutoff at 16:00 America/New_York dynamically converted to UTC (20:00 EDT / 21:00 EST)
        close_et = ts.tz_localize("America/New_York").replace(
            hour=16, minute=0, second=0, microsecond=0
        )
        cutoff_ts = close_et.tz_convert("UTC").tz_localize(None)
        cutoff = np.datetime64(cutoff_ts, "ns")
        cutoff_1d = cutoff - np.timedelta64(24, "h")
        cutoff_3d = cutoff - np.timedelta64(72, "h")
        cutoff_7d = cutoff - np.timedelta64(168, "h")

        idx_end = int(np.searchsorted(pub_times, cutoff, side="right"))
        idx_start_1d = int(np.searchsorted(pub_times[:idx_end], cutoff_1d, side="right"))
        idx_start_3d = int(np.searchsorted(pub_times[:idx_end], cutoff_3d, side="right"))
        idx_start_7d = int(np.searchsorted(pub_times[:idx_end], cutoff_7d, side="right"))

        c1 = float(idx_end - idx_start_1d)
        c3 = float(idx_end - idx_start_3d)
        c7 = float(idx_end - idx_start_7d)

        counts_1d.append(c1)
        counts_3d.append(c3)
        counts_7d.append(c7)

        if c3 > 0:
            sub_neg = neg_scores[idx_start_3d:idx_end]
            sub_pos = pos_scores[idx_start_3d:idx_end]
            sub_comp = compound_scores[idx_start_3d:idx_end]
            n_mean = float(np.mean(sub_neg))
            p_mean = float(np.mean(sub_pos))
            s_disp = float(np.std(sub_comp, ddof=1)) if c3 > 1 else 0.0
            a_int = float(np.mean(np.abs(sub_comp)))
            n_int = float((c3 / 3.0) * n_mean)
        else:
            n_mean = 0.0
            p_mean = 0.0
            s_disp = 0.0
            a_int = 0.0
            n_int = 0.0

        neg_means.append(n_mean)
        pos_means.append(p_mean)
        sent_disps.append(s_disp)
        neg_intens.append(n_int)
        abs_intens.append(a_int)

        if idx_end > 0:
            latest_time = pub_times[idx_end - 1]
            elapsed_h = float((cutoff - latest_time) / np.timedelta64(1, "h"))
            hours_since.append(min(max(elapsed_h, 0.0), 168.0))
        else:
            hours_since.append(168.0)

    out["news_headline_count_1d"] = counts_1d
    out["news_headline_count_3d"] = counts_3d
    out["news_headline_count_7d"] = counts_7d
    out["news_negative_sentiment_mean"] = neg_means
    out["news_positive_sentiment_mean"] = pos_means
    out["news_sentiment_dispersion"] = sent_disps
    out["news_negative_news_intensity"] = neg_intens
    out["news_absolute_sentiment_intensity"] = abs_intens
    out["news_hours_since_latest_article"] = hours_since

    c1_series = pd.Series(counts_1d, index=sessions)
    roll_mean = c1_series.rolling(22, min_periods=1).mean()
    roll_std = c1_series.rolling(22, min_periods=1).std().fillna(1.0)
    out["news_volume_zscore"] = ((c1_series - roll_mean) / np.maximum(roll_std, 1.0)).to_numpy()

    return out


def build_feature_frame(
    frame: pd.DataFrame,
    *,
    annualization: float = DEFAULT_ANNUALIZATION,
    feature_mode: str = "price_plus_ohlc",
    market_frame: pd.DataFrame | None = None,
    news_frame: pd.DataFrame | None = None,
    ticker: str = "",
) -> pd.DataFrame:
    """Build causal market features; no value reads beyond the current row."""

    data = validate_ohlcv(frame)
    close = data["Close"]
    log_close = np.log(close)
    returns = log_close.diff()
    out = pd.DataFrame(index=data.index)
    out["return_1d"] = returns
    out["abs_return_1d"] = returns.abs()
    for window in (5, 20, 22, 60):
        # Match the deployable ``Vol_C2C_*`` features: sample standard
        # deviation (ddof=1), annualised only at the presentation boundary.
        out[f"realized_vol_{window}"] = returns.rolling(
            window, min_periods=window
        ).std() * math.sqrt(annualization)
    out["ewma_vol"] = returns.pow(2).ewm(alpha=1 - 0.94, adjust=False, min_periods=5).mean().pow(
        0.5
    ) * math.sqrt(annualization)
    out["return_mean_5"] = returns.rolling(5, min_periods=5).mean()
    out["return_mean_22"] = returns.rolling(22, min_periods=22).mean()
    out["return_std_22"] = returns.rolling(22, min_periods=22).std()

    if "Volume" in data.columns:
        out["log_volume_change"] = np.log1p(data["Volume"]).diff()
    else:
        out["log_volume_change"] = 0.0

    if feature_mode in (
        "price_plus_ohlc",
        "price_plus_ohlc_plus_market",
        "price_plus_ohlc_plus_market_plus_news",
    ):
        if not {"Open", "High", "Low"}.issubset(data.columns):
            raise ValueError("feature_mode requires Open, High, Low columns")
        high = data["High"]
        low = data["Low"]
        open_p = data["Open"]

        # Strict input integrity assertion
        if (high < np.maximum(open_p, close) - 1e-6).any():
            raise ValueError("High must be >= max(Open, Close)")
        if (low > np.minimum(open_p, close) + 1e-6).any():
            raise ValueError("Low must be <= min(Open, Close)")
        if (open_p <= 0).any() or (high <= 0).any() or (low <= 0).any():
            raise ValueError("Open, High, Low prices must be positive")

        hl_ratio = high / low
        co_ratio = close / open_p
        hc_ratio = high / close
        ho_ratio = high / open_p
        lc_ratio = low / close
        lo_ratio = low / open_p

        out["hl_range"] = np.log(hl_ratio)
        out["co_range"] = np.log(co_ratio)
        out["overnight_return"] = np.log(open_p / close.shift(1))

        # Parkinson (1980) daily variance proxy: (ln(H/L))^2 / (4 * ln 2)
        var_parkinson = np.log(hl_ratio) ** 2 / (4.0 * math.log(2.0))

        # Garman-Klass (1980) daily variance proxy: 0.5 * (ln(H/L))^2 - (2*ln 2 - 1) * (ln(C/O))^2
        var_gk = 0.5 * (np.log(hl_ratio) ** 2) - (2.0 * math.log(2.0) - 1.0) * (
            np.log(co_ratio) ** 2
        )
        var_gk = np.maximum(var_gk, 0.0)

        # Rogers-Satchell (1991) daily variance proxy: ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O)
        var_rs = np.log(hc_ratio) * np.log(ho_ratio) + np.log(lc_ratio) * np.log(lo_ratio)
        var_rs = np.maximum(var_rs, 0.0)

        for window in (5, 22, 60):
            out[f"parkinson_vol_{window}"] = var_parkinson.rolling(
                window, min_periods=window
            ).mean().pow(0.5) * math.sqrt(annualization)
            out[f"garman_klass_vol_{window}"] = var_gk.rolling(
                window, min_periods=window
            ).mean().pow(0.5) * math.sqrt(annualization)
            out[f"rogers_satchell_vol_{window}"] = var_rs.rolling(
                window, min_periods=window
            ).mean().pow(0.5) * math.sqrt(annualization)

    if (
        feature_mode in ("price_plus_ohlc_plus_market", "price_plus_ohlc_plus_market_plus_news")
        and market_frame is not None
    ):
        mkt_aligned = market_frame.reindex(data.index).ffill()
        for col in mkt_aligned.columns:
            out[f"mkt_{col}"] = mkt_aligned[col]

    if feature_mode == "price_plus_ohlc_plus_market_plus_news":
        if news_frame is not None:
            n_aligned = news_frame.reindex(data.index).fillna(0.0)
            for col in n_aligned.columns:
                out[col] = n_aligned[col]
        else:
            n_gen = build_causal_news_features(data.index, ticker=ticker or "UNKNOWN")
            for col in n_gen.columns:
                out[col] = n_gen[col]

    return out.replace([np.inf, -np.inf], np.nan)


@dataclass(frozen=True)
class VolatilityExamples:
    """Sequence examples and the causal quantities used by baselines."""

    sequences: np.ndarray
    target: np.ndarray
    dates: np.ndarray
    feature_names: tuple[str, ...]
    har_features: np.ndarray
    current_volatility: np.ndarray
    rolling_mean_volatility: np.ndarray
    ewma_volatility: np.ndarray
    target_horizon: int = 5
    canonical_har_volatility: np.ndarray | None = None
    origin_close: np.ndarray | None = None
    future_close: np.ndarray | None = None
    daily_returns: np.ndarray | None = None
    target_end_dates: np.ndarray | None = None

    def __post_init__(self) -> None:
        rows = len(self.sequences)
        if self.sequences.ndim != 3 or self.sequences.shape[0] != rows:
            raise ValueError("sequences must have shape [rows, lookback, features]")
        arrays = [
            self.target,
            self.dates,
            self.har_features,
            self.current_volatility,
            self.rolling_mean_volatility,
            self.ewma_volatility,
        ]
        if self.origin_close is not None:
            arrays.append(self.origin_close)
        if self.future_close is not None:
            arrays.append(self.future_close)
        if self.daily_returns is not None:
            arrays.append(self.daily_returns)
        if self.canonical_har_volatility is not None:
            arrays.append(self.canonical_har_volatility)
        if self.target_end_dates is not None:
            arrays.append(self.target_end_dates)
        if any(len(values) != rows for values in arrays):
            raise ValueError("example arrays must have matching row counts")
        if not np.isfinite(self.sequences).all() or not np.isfinite(self.target).all():
            raise ValueError("examples contain non-finite values")
        if (self.target <= 0).any():
            raise ValueError("volatility targets must be positive")
        if self.target_horizon < 1:
            raise ValueError("target_horizon must be positive")
        if self.target_end_dates is not None and not np.all(self.target_end_dates > self.dates):
            raise ValueError("target end dates must be after their origins")


def build_examples(
    frame: pd.DataFrame,
    config: VolatilityConfig | None = None,
    market_frame: pd.DataFrame | None = None,
    news_frame: pd.DataFrame | None = None,
    ticker: str = "",
) -> VolatilityExamples:
    """Construct causal lookback sequences and strictly future targets."""

    settings = config or VolatilityConfig()
    data = validate_ohlcv(frame)
    features = build_feature_frame(
        data,
        annualization=settings.annualization,
        feature_mode=settings.feature_mode,
        market_frame=market_frame,
        news_frame=news_frame,
        ticker=ticker,
    )
    target = realised_volatility(
        data["Close"], settings.horizon, annualization=settings.annualization
    )
    feature_names = tuple(features.columns)
    values = features.to_numpy(dtype=np.float64)
    har = features[["realized_vol_5", "realized_vol_22", "realized_vol_60"]].to_numpy(
        dtype=np.float64
    )
    rows: list[np.ndarray] = []
    targets: list[float] = []
    dates: list[np.datetime64] = []
    har_rows: list[np.ndarray] = []
    current: list[float] = []
    rolling: list[float] = []
    ewma: list[float] = []
    origin_closes: list[float] = []
    future_closes: list[float] = []
    daily_returns: list[float] = []
    target_end_dates: list[np.datetime64] = []
    canonical_har: list[float] = []
    daily_rv = pd.Series(
        np.log(data["Close"]).diff().pow(2).to_numpy(dtype=np.float64), index=data.index
    )
    canonical_har_cumulative = causal_log_har_forecasts(
        daily_rv, (settings.horizon,), minimum_history=60
    )[:, 0]
    first = max(settings.lookback - 1, 60)
    for origin in range(first, len(data) - settings.horizon):
        window = values[origin - settings.lookback + 1 : origin + 1]
        target_value = target[origin]
        if not (
            np.isfinite(window).all()
            and np.isfinite(target_value)
            and np.isfinite(har[origin]).all()
            and np.isfinite(canonical_har_cumulative[origin])
            and canonical_har_cumulative[origin] > 0
        ):
            continue
        rows.append(window)
        targets.append(float(target_value))
        dates.append(np.datetime64(data.index[origin].date()))
        target_end_dates.append(np.datetime64(data.index[origin + settings.horizon].date()))
        har_rows.append(har[origin])
        # The public persistence baseline is Vol_C2C_20.  Keep the research
        # baseline on the same trailing window rather than the legacy 22-day
        # proxy that is retained only as an input feature.
        current.append(float(features["realized_vol_20"].iloc[origin]))
        rolling.append(float(features["realized_vol_60"].iloc[origin]))
        ewma.append(float(features["ewma_vol"].iloc[origin]))
        origin_closes.append(float(data["Close"].iloc[origin]))
        future_closes.append(float(data["Close"].iloc[origin + settings.horizon]))
        daily_returns.append(float(features["return_1d"].iloc[origin]))
        canonical_har.append(
            float(
                np.sqrt(
                    canonical_har_cumulative[origin] * settings.annualization / settings.horizon
                )
            )
        )
    if not rows:
        raise ValueError("history did not produce any complete volatility examples")
    return VolatilityExamples(
        sequences=np.asarray(rows, dtype=np.float32),
        target=np.asarray(targets, dtype=np.float64),
        dates=np.asarray(dates, dtype="datetime64[D]"),
        feature_names=feature_names,
        har_features=np.asarray(har_rows, dtype=np.float64),
        current_volatility=np.asarray(current, dtype=np.float64),
        rolling_mean_volatility=np.asarray(rolling, dtype=np.float64),
        ewma_volatility=np.asarray(ewma, dtype=np.float64),
        target_horizon=settings.horizon,
        canonical_har_volatility=np.asarray(canonical_har, dtype=np.float64),
        target_end_dates=np.asarray(target_end_dates, dtype="datetime64[D]"),
        origin_close=np.asarray(origin_closes, dtype=np.float64),
        future_close=np.asarray(future_closes, dtype=np.float64),
        daily_returns=np.asarray(daily_returns, dtype=np.float64),
    )


@dataclass(frozen=True)
class LSTMConfig:
    """Bounded offline LSTM settings for a fair baseline comparison.

    PyTorch is imported only when :func:`lstm_predictions` is requested, so
    this optional research model can never become a production API
    dependency.  The scaler is fitted only on the supplied training rows.
    """

    hidden_size: int = 32
    dropout: float = 0.20
    maximum_epochs: int = 25
    patience: int = 5
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    device: str | None = None
    target_space: str = "log_variance"

    def __post_init__(self) -> None:
        if self.hidden_size < 4 or self.maximum_epochs < 1 or self.patience < 1:
            raise ValueError("LSTM size, epoch, and patience settings must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("LSTM dropout must be in [0, 1)")
        if self.batch_size < 1 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("LSTM optimizer and batch settings are invalid")
        if self.target_space not in (
            "log_volatility",
            "log_variance",
            "direct_volatility",
            "softplus_volatility",
        ):
            raise ValueError(f"Unknown target_space: {self.target_space}")


def lstm_predictions(
    examples: VolatilityExamples,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    *,
    config: LSTMConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a compact LSTM on target_space and predict every example.

    ``validation_indices`` is kept outside the optimizer's training rows and
    is used only for early stopping.  The function is intentionally offline;
    importing it does not import PyTorch and no weights are persisted.
    """

    settings = config or LSTMConfig()
    train = np.asarray(train_indices, dtype=np.int64)
    validation = np.asarray(validation_indices, dtype=np.int64)
    if train.ndim != 1 or validation.ndim != 1 or len(train) < 8 or len(validation) < 2:
        raise ValueError("LSTM requires non-empty train and validation partitions")
    if set(train) & set(validation):
        raise ValueError("LSTM train and validation partitions overlap")

    try:
        import torch
        from torch import nn
    except ImportError as err:  # pragma: no cover - depends on optional environment
        raise RuntimeError(
            "PyTorch is required for --include-lstm; install an offline CPU or CUDA build."
        ) from err

    torch.manual_seed(settings.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(settings.seed)
    device = torch.device(settings.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    feature_count = int(examples.sequences.shape[-1])
    scaler = StandardScaler().fit(
        examples.sequences[train].reshape(-1, feature_count).astype(np.float64)
    )
    scaled = (
        scaler.transform(examples.sequences.reshape(-1, feature_count).astype(np.float64))
        .reshape(examples.sequences.shape)
        .astype(np.float32)
    )
    if settings.target_space in ("direct_volatility", "softplus_volatility"):
        target_array = _positive(examples.target).astype(np.float32)
    elif settings.target_space == "log_variance":
        target_array = np.log(_positive(examples.target**2)).astype(np.float32)
    else:  # "log_volatility"
        target_array = np.log(_positive(examples.target)).astype(np.float32)

    class VolatilityLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.LSTM(
                feature_count,
                settings.hidden_size,
                batch_first=True,
            )
            self.dropout = nn.Dropout(settings.dropout)
            self.head = nn.Linear(settings.hidden_size, 1)

        def forward(self, values: Any) -> Any:
            encoded, _ = self.encoder(values)
            raw = self.head(self.dropout(encoded[:, -1, :])).squeeze(-1)
            if settings.target_space == "softplus_volatility":
                return torch.nn.functional.softplus(raw) + 1e-6
            return raw

    model = VolatilityLSTM().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    loss_fn = nn.SmoothL1Loss()
    train_x = torch.as_tensor(scaled[train], dtype=torch.float32, device=device)
    train_y = torch.as_tensor(target_array[train], dtype=torch.float32, device=device)
    val_x = torch.as_tensor(scaled[validation], dtype=torch.float32, device=device)
    val_y = torch.as_tensor(target_array[validation], dtype=torch.float32, device=device)
    started = time.perf_counter()
    best_state: dict[str, Any] | None = None
    best_loss = math.inf
    stale_epochs = 0
    completed_epochs = 0

    try:
        for epoch in range(settings.maximum_epochs):
            model.train()
            for start in range(0, len(train), settings.batch_size):
                stop = min(start + settings.batch_size, len(train))
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(train_x[start:stop]), train_y[start:stop])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                validation_loss = float(loss_fn(model(val_x), val_y).item())
            completed_epochs = epoch + 1
            if validation_loss < best_loss - 1e-5:
                best_loss = validation_loss
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= settings.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        all_x = torch.as_tensor(scaled, dtype=torch.float32, device=device)
        predictions: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(all_x), settings.batch_size):
                stop = min(start + settings.batch_size, len(all_x))
                predictions.append(model(all_x[start:stop]).detach().cpu().numpy())
        raw_pred = np.concatenate(predictions)
        if settings.target_space in ("direct_volatility", "softplus_volatility"):
            prediction = np.maximum(raw_pred, _EPS)
        elif settings.target_space == "log_variance":
            prediction = np.sqrt(np.exp(np.clip(raw_pred, -40.0, 10.0)))
        else:
            prediction = np.exp(np.clip(raw_pred, -20.0, 5.0))

        if not np.isfinite(prediction).all() or (prediction <= 0).any():
            raise ValueError("LSTM produced non-finite or non-positive volatility")
        metadata = {
            "family": "lstm",
            "hidden_size": settings.hidden_size,
            "dropout": settings.dropout,
            "epochs": completed_epochs,
            "best_validation_log_loss": None if not np.isfinite(best_loss) else best_loss,
            "device": str(device),
            "training_seconds": time.perf_counter() - started,
            "scaler": "train_only_standard",
            "target_space": settings.target_space,
        }
        return np.asarray(prediction, dtype=np.float64), metadata
    finally:
        del model, optimizer, train_x, train_y, val_x, val_y
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _positive(values: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(values, dtype=np.float64), _EPS)


def fit_har_baseline(examples: VolatilityExamples, train_indices: np.ndarray) -> np.ndarray:
    """Return the canonical causal HAR forecast for every example.

    ``build_examples`` evaluates the same recursive log-HAR implementation
    used by the serving path at each origin.  Refitting a second direct-H
    regression here would make offline scores incomparable with the API.
    ``train_indices`` remains for compatibility; the canonical filter is
    causal at every origin and does not consume held-out target values.
    """

    del train_indices
    if examples.canonical_har_volatility is None:
        raise ValueError(
            "examples do not contain canonical HAR forecasts; rebuild them with build_examples"
        )
    return _positive(examples.canonical_har_volatility)


def fit_garch11_baseline(
    examples: VolatilityExamples,
    train_indices: np.ndarray,
    horizon: int = 1,
    annualization: float = DEFAULT_ANNUALIZATION,
) -> np.ndarray:
    """Fit causal GARCH(1,1) via MLE on training returns and propagate conditional variance."""

    train = np.asarray(train_indices, dtype=np.int64)
    if train.ndim != 1 or len(train) < 20:
        raise ValueError("GARCH(1,1) requires at least twenty training rows")

    if examples.daily_returns is not None:
        r_all = np.asarray(examples.daily_returns, dtype=np.float64)
    else:
        r_idx = (
            examples.feature_names.index("return_1d")
            if "return_1d" in examples.feature_names
            else 0
        )
        r_all = examples.sequences[:, -1, r_idx].astype(np.float64)

    r_train = r_all[train]
    sample_var = float(np.var(r_train))
    if sample_var < _EPS:
        sample_var = 1e-4

    def _nll(params: np.ndarray) -> float:
        omega, alpha, beta = params
        if alpha + beta >= 1.0 or omega <= 0 or alpha < 0 or beta < 0:
            return 1e10
        n = len(r_train)
        h = np.empty(n, dtype=np.float64)
        h[0] = sample_var
        for i in range(1, n):
            h[i] = omega + alpha * (r_train[i - 1] ** 2) + beta * h[i - 1]
            if h[i] <= 0 or not np.isfinite(h[i]):
                return 1e10
        ll = -0.5 * np.sum(np.log(h) + (r_train**2) / h)
        return float(-ll)

    init_params = np.array([0.05 * sample_var, 0.08, 0.87], dtype=np.float64)
    bounds = [(1e-10, 1.0), (1e-4, 0.40), (0.50, 0.999)]

    res = minimize(
        _nll,
        init_params,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 150, "ftol": 1e-7},
    )

    if res.success and (res.x[1] + res.x[2] < 1.0):
        omega, alpha, beta = float(res.x[0]), float(res.x[1]), float(res.x[2])
    else:
        alpha, beta = 0.08, 0.88
        omega = (1.0 - alpha - beta) * sample_var

    persistence = alpha + beta
    unconditional_var = omega / max(1.0 - persistence, 1e-5)

    n_all = len(r_all)
    h_filtered = np.empty(n_all, dtype=np.float64)
    h_filtered[0] = sample_var
    for i in range(1, n_all):
        h_filtered[i] = omega + alpha * (r_all[i - 1] ** 2) + beta * h_filtered[i - 1]

    h_next = omega + alpha * (r_all**2) + beta * h_filtered

    if persistence >= 0.9999 or abs(1.0 - persistence) < 1e-6:
        cum_var = horizon * h_next
    else:
        geom_sum = (1.0 - (persistence**horizon)) / (1.0 - persistence)
        cum_var = horizon * unconditional_var + (h_next - unconditional_var) * geom_sum

    ann_vol = np.sqrt(np.maximum(cum_var * annualization / horizon, _EPS))
    return np.asarray(ann_vol, dtype=np.float64)


def baseline_predictions(
    examples: VolatilityExamples,
    train_indices: np.ndarray,
    *,
    horizon: int | None = None,
    annualization: float = DEFAULT_ANNUALIZATION,
) -> dict[str, np.ndarray]:
    """Return persistence, rolling mean, EWMA, HAR-RV, and GARCH(1,1) forecasts."""

    effective_horizon = int(horizon or examples.target_horizon)
    if effective_horizon != examples.target_horizon:
        raise ValueError(
            "baseline horizon must match the examples target horizon; rebuild examples for a new horizon"
        )
    return {
        "persistence": _positive(examples.current_volatility),
        "rolling_mean": _positive(examples.rolling_mean_volatility),
        "ewma": _positive(examples.ewma_volatility),
        "har_rv": _positive(fit_har_baseline(examples, train_indices)),
        "garch_11": _positive(
            fit_garch11_baseline(
                examples,
                train_indices,
                horizon=effective_horizon,
                annualization=annualization,
            )
        ),
    }


def _fit_scaled_regressor(
    examples: VolatilityExamples,
    train_indices: np.ndarray,
    estimator: Any,
    *,
    target_space: str = "log_volatility",
) -> tuple[Any, StandardScaler]:
    train = np.asarray(train_indices, dtype=np.int64)
    x_train = examples.sequences[train].reshape(len(train), -1).astype(np.float64)
    scaler = StandardScaler().fit(x_train)
    if target_space == "direct_volatility":
        y_train = _positive(examples.target[train])
    elif target_space == "log_variance":
        y_train = np.log(_positive(examples.target[train] ** 2))
    else:  # "log_volatility"
        y_train = np.log(_positive(examples.target[train]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        estimator.fit(scaler.transform(x_train), y_train)
    return estimator, scaler


def learned_predictions(
    examples: VolatilityExamples,
    train_indices: np.ndarray,
    *,
    include_boosting: bool = True,
    target_space: str = "log_volatility",
) -> dict[str, np.ndarray]:
    """Fit simple learned models using only the supplied training indices."""

    all_x = examples.sequences.reshape(len(examples.sequences), -1).astype(np.float64)
    predictions: dict[str, np.ndarray] = {}

    ridge, scaler = _fit_scaled_regressor(
        examples, train_indices, Ridge(alpha=1.0), target_space=target_space
    )
    pred_raw_ridge = ridge.predict(scaler.transform(all_x))
    if target_space == "direct_volatility":
        predictions["ridge"] = _positive(pred_raw_ridge)
    elif target_space == "log_variance":
        predictions["ridge"] = _positive(np.sqrt(np.exp(np.clip(pred_raw_ridge, -40.0, 10.0))))
    else:
        predictions["ridge"] = _positive(np.exp(np.clip(pred_raw_ridge, -20.0, 5.0)))

    elastic_net, elastic_scaler = _fit_scaled_regressor(
        examples,
        train_indices,
        ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000, tol=1e-3),
        target_space=target_space,
    )
    pred_raw_elastic = elastic_net.predict(elastic_scaler.transform(all_x))
    if target_space == "direct_volatility":
        predictions["elastic_net"] = _positive(pred_raw_elastic)
    elif target_space == "log_variance":
        predictions["elastic_net"] = _positive(
            np.sqrt(np.exp(np.clip(pred_raw_elastic, -40.0, 10.0)))
        )
    else:
        predictions["elastic_net"] = _positive(np.exp(np.clip(pred_raw_elastic, -20.0, 5.0)))

    if include_boosting:
        boosting, boosting_scaler = _fit_scaled_regressor(
            examples,
            train_indices,
            GradientBoostingRegressor(
                learning_rate=0.05,
                n_estimators=200,
                max_depth=3,
                max_features="sqrt",
                random_state=42,
            ),
            target_space=target_space,
        )
        pred_raw_boosting = boosting.predict(boosting_scaler.transform(all_x))
        if target_space == "direct_volatility":
            predictions["gradient_boosting"] = _positive(pred_raw_boosting)
        elif target_space == "log_variance":
            predictions["gradient_boosting"] = _positive(
                np.sqrt(np.exp(np.clip(pred_raw_boosting, -40.0, 10.0)))
            )
        else:
            predictions["gradient_boosting"] = _positive(
                np.exp(np.clip(pred_raw_boosting, -20.0, 5.0))
            )

    return predictions


def volatility_metrics(
    actual: np.ndarray,
    forecast: np.ndarray,
    *,
    epsilon: float = _EPS,
) -> dict[str, float]:
    r"""Calculate point errors plus canonical Patton (2011) QLIKE on variance.

    QLIKE(h, \hat{h}) = h / \hat{h} - log(h / \hat{h}) - 1
    where h = actual_sigma^2 and \hat{h} = forecast_sigma^2.
    """

    observed = np.maximum(np.asarray(actual, dtype=np.float64), epsilon)
    predicted = np.asarray(forecast, dtype=np.float64)
    if observed.shape != predicted.shape or observed.ndim != 1 or not len(observed):
        raise ValueError("metric arrays must be matched non-empty vectors")

    raw_min = float(np.min(predicted))
    near_zero_count = int(np.sum(predicted <= 1e-4))
    stabilized_pred = np.maximum(predicted, epsilon)

    error = predicted - observed
    actual_variance = observed**2
    forecast_variance = stabilized_pred**2
    ratio = actual_variance / forecast_variance
    qlike_vector = ratio - np.log(ratio) - 1.0

    total_qlike = float(np.sum(qlike_vector))
    n = len(qlike_vector)
    worst_1pct_count = max(1, int(np.ceil(0.01 * n)))
    sorted_qlike = np.sort(qlike_vector)
    worst_1pct_sum = float(np.sum(sorted_qlike[-worst_1pct_count:]))
    worst_1pct_share = (worst_1pct_sum / total_qlike * 100.0) if total_qlike > 0 else 0.0

    return {
        "mae": float(np.mean(np.abs(error))),
        "mse": float(np.mean(error**2)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "qlike": float(np.mean(qlike_vector)),
        "median_qlike": float(np.median(qlike_vector)),
        "p90_qlike": float(np.quantile(qlike_vector, 0.90)),
        "p95_qlike": float(np.quantile(qlike_vector, 0.95)),
        "p99_qlike": float(np.quantile(qlike_vector, 0.99)),
        "max_qlike": float(np.max(qlike_vector)),
        "worst_1pct_share": worst_1pct_share,
        "r2": float(1.0 - np.sum(error**2) / np.sum((observed - np.mean(observed)) ** 2))
        if np.sum((observed - np.mean(observed)) ** 2) > epsilon
        else 0.0,
        "raw_min_pred": raw_min,
        "near_zero_count": near_zero_count,
    }


def evaluate_conformal_volatility_intervals(
    actual_validation: np.ndarray,
    forecast_validation: np.ndarray,
    actual_test: np.ndarray,
    forecast_test: np.ndarray,
    *,
    nominal_coverage: float = 0.90,
) -> dict[str, Any]:
    """Calibrate split-conformal intervals on validation residuals and evaluate on test."""

    if not 0 < nominal_coverage < 1:
        raise ValueError("nominal_coverage must be between zero and one")

    val_act = _positive(actual_validation)
    val_pred = _positive(forecast_validation)
    test_act = _positive(actual_test)
    test_pred = _positive(forecast_test)
    if len(val_act) < 4 or len(test_act) < 4:
        raise ValueError("at least 4 validation and test observations required")

    log_residuals = np.abs(np.log(val_act) - np.log(val_pred))
    rank = min(int(np.ceil((len(log_residuals) + 1) * nominal_coverage)), len(log_residuals))
    radius = float(np.sort(log_residuals)[rank - 1])

    lower = test_pred * math.exp(-radius)
    upper = test_pred * math.exp(radius)

    inside = (test_act >= lower) & (test_act <= upper)
    empirical_coverage = float(np.mean(inside))
    average_width = float(np.mean(upper - lower))

    # Regime thresholds are fitted on the calibration information set only.
    # Using test actual volatility here would leak the outcome into the
    # subgroup definition and make regime coverage look more stable than it
    # is at deployment.
    tertiles = np.quantile(val_act, [1.0 / 3.0, 2.0 / 3.0])
    regime_low = test_act <= tertiles[0]
    regime_normal = (test_act > tertiles[0]) & (test_act <= tertiles[1])
    regime_high = test_act > tertiles[1]

    def _regime_cov(mask: np.ndarray) -> float | None:
        count = int(np.sum(mask))
        return float(np.mean(inside[mask])) if count > 0 else None

    return {
        "interval_method": "rolling_origin_split_conformal_log_volatility",
        "metric_source": "untouched_chronological_test",
        "nominal_coverage": float(nominal_coverage),
        "empirical_coverage": empirical_coverage,
        "conformal_log_radius": radius,
        "calibration_count": int(len(log_residuals)),
        "regime_thresholds": {
            "low_high_boundary": float(tertiles[0]),
            "normal_high_boundary": float(tertiles[1]),
            "source": "validation_actual_only",
        },
        "average_width": average_width,
        "regime_coverage": {
            "low_vol": _regime_cov(regime_low),
            "normal_vol": _regime_cov(regime_normal),
            "high_vol": _regime_cov(regime_high),
        },
    }


def evaluate_price_diffusion_cone(
    origin_close: np.ndarray,
    future_close: np.ndarray,
    forecast_annualized_vol: np.ndarray,
    horizon: int,
    *,
    nominal_coverage: float = 0.90,
    annualization: float = DEFAULT_ANNUALIZATION,
) -> dict[str, Any]:
    """Evaluate a *raw* Gaussian price-return reference scenario.

    This function deliberately does not calibrate residuals or tune the
    multiplier on the test partition.  ``p05``--``p95`` is a central 90%
    nominal interval under a zero-location Gaussian assumption; the returned
    empirical coverage is descriptive out-of-sample evidence, not a guarantee.
    """

    if not 0 < nominal_coverage < 1:
        raise ValueError("nominal_coverage must be between zero and one")

    p_orig = np.asarray(origin_close, dtype=np.float64)
    p_future = np.asarray(future_close, dtype=np.float64)
    vol = _positive(forecast_annualized_vol)
    if len(p_orig) != len(p_future) or len(p_orig) != len(vol) or len(p_orig) < 4:
        raise ValueError("matched arrays of at least 4 observations required")
    if (
        not np.isfinite(p_orig).all()
        or not np.isfinite(p_future).all()
        or (p_orig <= 0).any()
        or (p_future <= 0).any()
    ):
        raise ValueError("price arrays must be finite and positive")

    # Quantile z-scores for standard nominal coverages
    z = float(norm.ppf(0.5 + float(nominal_coverage) / 2.0))

    dt = horizon / annualization
    horizon_sigma = vol * math.sqrt(dt)
    lower = p_orig * np.exp(-z * horizon_sigma)
    upper = p_orig * np.exp(+z * horizon_sigma)

    inside = (p_future >= lower) & (p_future <= upper)
    empirical_coverage = float(np.mean(inside))
    avg_width_pct = float(np.mean((upper - lower) / p_orig))

    tertiles = np.quantile(vol, [1.0 / 3.0, 2.0 / 3.0])
    regime_low = vol <= tertiles[0]
    regime_normal = (vol > tertiles[0]) & (vol <= tertiles[1])
    regime_high = vol > tertiles[1]

    def _regime_cov(mask: np.ndarray) -> float | None:
        count = int(np.sum(mask))
        return float(np.mean(inside[mask])) if count > 0 else None

    return {
        "interval_method": "gaussian_reference_scenario",
        "metric_source": "untouched_chronological_test_descriptive",
        "interval_scope": "pointwise_marginal_reference",
        "location_assumption": "zero_log_return",
        "variance_assumption": "forecast_annualized_volatility_scaled_by_horizon",
        "nominal_coverage": float(nominal_coverage),
        "empirical_coverage": empirical_coverage,
        "average_width_pct": avg_width_pct,
        "z_score": float(z),
        "regime_coverage": {
            "low_vol": _regime_cov(regime_low),
            "normal_vol": _regime_cov(regime_normal),
            "high_vol": _regime_cov(regime_high),
        },
    }


def evaluate_benchmark(
    examples: VolatilityExamples,
    split: ChronologicalSplit,
    *,
    include_boosting: bool = True,
    include_lstm: bool = False,
    lstm_config: LSTMConfig | None = None,
    nominal_coverage: float = 0.90,
    target_space: str = "log_variance",
    return_forecasts: bool = False,
) -> (
    dict[str, dict[str, dict[str, Any]]]
    | tuple[dict[str, dict[str, dict[str, Any]]], dict[str, np.ndarray]]
):
    """Evaluate all baselines and simple ML models on validation and test."""

    assert_label_purged(examples, split)

    # ``embargo_sessions`` is a split boundary, not the forecast horizon.  A
    # caller may deliberately use a wider embargo; scoring and the cone must
    # still describe the target horizon used to build ``examples``.
    forecast_horizon = examples.target_horizon
    forecasts = baseline_predictions(
        examples, split.train, horizon=forecast_horizon, annualization=DEFAULT_ANNUALIZATION
    )
    forecasts.update(
        learned_predictions(
            examples,
            split.train,
            include_boosting=include_boosting,
            target_space=target_space
            if target_space in ("log_variance", "direct_volatility", "log_volatility")
            else "log_variance",
        )
    )
    if include_lstm:
        effective_lstm_config = (
            lstm_config if lstm_config is not None else LSTMConfig(target_space=target_space)
        )
        if effective_lstm_config.target_space != target_space:
            effective_lstm_config = LSTMConfig(
                hidden_size=effective_lstm_config.hidden_size,
                dropout=effective_lstm_config.dropout,
                maximum_epochs=effective_lstm_config.maximum_epochs,
                patience=effective_lstm_config.patience,
                batch_size=effective_lstm_config.batch_size,
                learning_rate=effective_lstm_config.learning_rate,
                weight_decay=effective_lstm_config.weight_decay,
                seed=effective_lstm_config.seed,
                device=effective_lstm_config.device,
                target_space=target_space,
            )
        lstm_forecast, _ = lstm_predictions(
            examples,
            split.train,
            split.validation,
            config=effective_lstm_config,
        )
        forecasts["lstm"] = lstm_forecast
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for name, prediction in forecasts.items():
        output[name] = {}
        for partition, indices in (("validation", split.validation), ("test", split.test)):
            metrics = volatility_metrics(examples.target[indices], prediction[indices])
            metrics["rows"] = int(len(indices))
            output[name][partition] = metrics

        # Conformal prediction intervals on realized volatility
        try:
            val_act = examples.target[split.validation]
            val_pred = prediction[split.validation]
            test_act = examples.target[split.test]
            test_pred = prediction[split.test]
            output[name]["test"]["volatility_interval"] = evaluate_conformal_volatility_intervals(
                val_act, val_pred, test_act, test_pred, nominal_coverage=nominal_coverage
            )
        except Exception:
            pass

        # Price diffusion cone calibration
        if examples.origin_close is not None and examples.future_close is not None:
            with suppress(Exception):
                output[name]["test"]["price_cone"] = evaluate_price_diffusion_cone(
                    examples.origin_close[split.test],
                    examples.future_close[split.test],
                    prediction[split.test],
                    horizon=forecast_horizon,
                    nominal_coverage=nominal_coverage,
                )

    if return_forecasts:
        return output, forecasts
    return output


def select_validation_model(metrics: dict[str, dict[str, dict[str, Any]]]) -> str:
    """Select by validation QLIKE only; test scores never influence selection."""

    if not metrics:
        raise ValueError("benchmark metrics are empty")
    return min(
        metrics,
        key=lambda name: (float(metrics[name]["validation"]["qlike"]), name),
    )


def experiment_metadata(
    examples: VolatilityExamples,
    config: VolatilityConfig,
    split: ChronologicalSplit,
    *,
    model: str,
    metrics: dict[str, dict[str, Any]],
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Build a small JSON-serialisable record for an experiment run."""

    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "target_version": TARGET_VERSION,
        "run_id": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "git_commit": git_commit,
        "model": model,
        "configuration": asdict(config),
        "feature_names": list(examples.feature_names),
        "rows": int(len(examples.target)),
        "training_rows": int(len(split.train)),
        "validation_rows": int(len(split.validation)),
        "test_rows": int(len(split.test)),
        "date_start": str(examples.dates[0]),
        "date_end": str(examples.dates[-1]),
        "metrics": metrics,
    }
    digest_payload = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["run_hash"] = hashlib.sha256(digest_payload).hexdigest()
    return payload


__all__ = [
    "PIPELINE_VERSION",
    "TARGET_VERSION",
    "ChronologicalSplit",
    "LSTMConfig",
    "VolatilityConfig",
    "VolatilityExamples",
    "assert_label_purged",
    "baseline_predictions",
    "build_examples",
    "build_feature_frame",
    "chronological_split",
    "evaluate_benchmark",
    "evaluate_conformal_volatility_intervals",
    "evaluate_price_diffusion_cone",
    "experiment_metadata",
    "fit_garch11_baseline",
    "fit_har_baseline",
    "learned_predictions",
    "lstm_predictions",
    "realised_volatility",
    "select_validation_model",
    "validate_ohlcv",
    "volatility_metrics",
]
