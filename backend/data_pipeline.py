# Fetches and preprocesses multi-feature stock data
#
# Features included: OHLCV, Technical Indicators, Market Returns, Cyclical Calendar
# Phase 3: Decoupled sequence generation for walk-forward validation support.
# Schema v4: stationary price-relative features for browser-trained models.

import hashlib

import numpy as np
import pandas as pd
import yfinance as yf  # type: ignore[import-untyped]
from sklearn.preprocessing import RobustScaler  # type: ignore[import-untyped]

from config import (
    FEATURES,
    FEATURES_V4,
    HISTORICAL_YEARS,
    MAX_FORECAST_DAYS,
    SNAPSHOT_SCHEMA_VERSION,
    TRAIN_SPLIT,
    WINDOW_SIZE,
)
from features.pipeline import build_browser_features, build_features

ROBUST_SCALER_QUANTILE_RANGE = (25.0, 75.0)


def fetch_data(ticker: str):
    """
    Download historical OHLCV prices, build enriched features, and validate schema.
    Returns (feature_df, closing_prices, dates, feature_metadata).
    """
    data = yf.download(
        ticker,
        period=f"{HISTORICAL_YEARS}y",
        progress=False,
        auto_adjust=True,
        timeout=30,
    )

    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError(f"No market data is available for {ticker}.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    required_ohlcv = {"Open", "High", "Low", "Close", "Volume"}
    if not required_ohlcv.issubset(data.columns):
        raise ValueError(f"Market data for {ticker} is missing required OHLCV columns.")
    data = data.loc[~data.index.duplicated(keep="last")].sort_index().dropna(subset=required_ohlcv)
    if not np.isfinite(data[list(required_ohlcv)].to_numpy(dtype=float)).all():
        raise ValueError(f"Market data for {ticker} contains non-finite OHLCV values.")
    if (data[["Open", "High", "Low", "Close"]] <= 0).any().any() or (data["Volume"] < 0).any():
        raise ValueError(f"Market data for {ticker} contains invalid OHLCV values.")
    min_rows = WINDOW_SIZE + MAX_FORECAST_DAYS + 30  # account for 20-day rolling window NaNs
    if len(data) < min_rows:
        raise ValueError(
            f"Not enough historical data for {ticker}. Need at least {min_rows} trading days."
        )

    # Build features & metadata
    feature_df, feature_metadata = build_features(data, FEATURES)
    snapshot_hasher = hashlib.sha256()
    snapshot_hasher.update("|".join(feature_df.columns).encode("utf-8"))
    snapshot_hasher.update(pd.util.hash_pandas_object(feature_df, index=True).values.tobytes())
    feature_metadata["snapshot_id"] = snapshot_hasher.hexdigest()
    feature_metadata["ticker"] = ticker
    feature_metadata["feature_schema_version"] = 3
    closing_prices = feature_df["Close"].to_numpy()

    return feature_df, closing_prices, feature_df.index, feature_metadata


def fetch_browser_data(ticker: str):
    """
    Download adjusted OHLCV history and build the stationary schema-v4 matrix
    served to browser training. Returns (feature_df, closing_prices, dates,
    feature_metadata) where the feature matrix contains only v4 features and
    closing prices are aligned row-for-row to it.
    """
    data = yf.download(
        ticker,
        period=f"{HISTORICAL_YEARS}y",
        progress=False,
        auto_adjust=True,
        timeout=30,
    )

    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError(f"No market data is available for {ticker}.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    required_ohlcv = {"Open", "High", "Low", "Close", "Volume"}
    if not required_ohlcv.issubset(data.columns):
        raise ValueError(f"Market data for {ticker} is missing required OHLCV columns.")
    data = data.loc[~data.index.duplicated(keep="last")].sort_index().dropna(subset=required_ohlcv)
    if not np.isfinite(data[list(required_ohlcv)].to_numpy(dtype=float)).all():
        raise ValueError(f"Market data for {ticker} contains non-finite OHLCV values.")
    if (data[["Open", "High", "Low", "Close"]] <= 0).any().any() or (data["Volume"] < 0).any():
        raise ValueError(f"Market data for {ticker} contains invalid OHLCV values.")
    min_rows = WINDOW_SIZE + MAX_FORECAST_DAYS + 30  # 20-day rolling windows plus horizon buffer
    if len(data) < min_rows:
        raise ValueError(
            f"Not enough historical data for {ticker}. Need at least {min_rows} trading days."
        )

    feature_df, feature_metadata = build_browser_features(data, FEATURES_V4)
    closing_prices = data.loc[feature_df.index, "Close"].to_numpy(dtype=float)
    if len(closing_prices) != len(feature_df) or not np.isfinite(closing_prices).all():
        raise ValueError(f"Adjusted close history for {ticker} is incompatible with the features.")
    if (closing_prices <= 0).any():
        raise ValueError(f"Adjusted close history for {ticker} contains invalid values.")

    snapshot_hasher = hashlib.sha256()
    snapshot_hasher.update("|".join(feature_df.columns).encode("utf-8"))
    snapshot_hasher.update(pd.util.hash_pandas_object(feature_df, index=True).values.tobytes())
    feature_metadata["snapshot_id"] = snapshot_hasher.hexdigest()
    feature_metadata["ticker"] = ticker
    feature_metadata["feature_schema_version"] = SNAPSHOT_SCHEMA_VERSION
    feature_metadata["adjusted_prices"] = True
    feature_metadata["quality"] = snapshot_quality_diagnostics(
        feature_df,
        closing_prices,
        feature_df.index,
        (feature_metadata.get("market_context") or {}).get("sources"),
    )

    return feature_df, closing_prices, feature_df.index, feature_metadata


def snapshot_quality_diagnostics(
    feature_df: pd.DataFrame,
    closing_prices: np.ndarray,
    dates: pd.DatetimeIndex,
    market_sources: dict | None = None,
) -> dict:
    """
    Data-quality checks for browser snapshots.

    Hard failures reject the snapshot; soft findings are annotated in the
    snapshot metadata so users can see exactly what the model saw.
    """
    issues: list[dict] = []
    checks: dict[str, object] = {}
    prices = np.asarray(closing_prices, dtype=float).reshape(-1)

    duplicate_count = int(feature_df.index.duplicated().sum())
    checks["duplicate_dates"] = duplicate_count
    if duplicate_count:
        issues.append(
            {
                "code": "duplicate_dates",
                "severity": "error",
                "detail": f"{duplicate_count} duplicate rows.",
            }
        )

    non_positive = bool((prices <= 0).any())
    checks["non_positive_prices"] = non_positive
    if non_positive:
        issues.append(
            {
                "code": "non_positive_values",
                "severity": "error",
                "detail": "Non-positive prices found.",
            }
        )

    if len(prices) > 1:
        returns = np.diff(np.log(prices))
        large_move = int(np.abs(returns).max() > 0.20)
        checks["large_single_day_move"] = bool(large_move)
        if large_move:
            issues.append(
                {
                    "code": "large_single_day_move",
                    "severity": "warning",
                    "detail": "A single-day move exceeds 20%, which is inconsistent with adjusted data.",
                }
            )

    if len(dates) > 1:
        calendar_gaps = int((dates.to_series().diff().dt.days > 7).sum())
        checks["missing_sessions_gap_gt_7_days"] = calendar_gaps
        if calendar_gaps:
            issues.append(
                {
                    "code": "missing_sessions",
                    "severity": "warning",
                    "detail": f"{calendar_gaps} session gaps longer than 7 calendar days.",
                }
            )

    latest_date = dates[-1]
    staleness_days = (pd.Timestamp.now() - latest_date).days if len(dates) else 0
    checks["stale_latest_observation_days"] = int(staleness_days)
    if staleness_days > 15:
        issues.append(
            {
                "code": "stale_latest_observation",
                "severity": "warning",
                "detail": f"Latest observation is {staleness_days} days old.",
            }
        )

    benchmark_short = False
    if market_sources:
        stock_last = dates[-1].date()
        for name, source in market_sources.items():
            last = source.get("last_date")
            if last:
                try:
                    benchmark_last = pd.Timestamp(last).date()
                    if benchmark_last < stock_last:
                        benchmark_short = True
                        checks[f"benchmark_{name}_end"] = last
                except (ValueError, TypeError):
                    continue
    checks["benchmark_series_end_earlier"] = benchmark_short
    if benchmark_short:
        issues.append(
            {
                "code": "benchmark_series_end_earlier",
                "severity": "warning",
                "detail": "One or more market benchmark series end before the stock series.",
            }
        )

    return {"checks": checks, "issues": issues, "status": "clean" if not issues else "annotated"}


def create_sequences(
    scaled: np.ndarray,
    dates: pd.DatetimeIndex,
    close_idx: int,
    forecast_days: int = MAX_FORECAST_DAYS,
    window_size: int = WINDOW_SIZE,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Create sliding-window 3D input sequences and multi-step targets from scaled feature data.

    Generic: can be used by any model type. The caller controls the scaler.
    Returns (X, y, sequence_end_dates) where sequence_end_dates[i] is the last date in window i.
    """
    n_samples = len(scaled) - window_size - forecast_days + 1
    if n_samples <= 0:
        raise ValueError("Not enough data for sequence creation after windowing.")

    X_list, y_list, seq_dates = [], [], []
    for i in range(window_size, window_size + n_samples):
        X_list.append(scaled[i - window_size : i, :])
        y_list.append(scaled[i : i + forecast_days, close_idx])
        # Date of the last timestep in the window (the "as-of" date)
        seq_dates.append(str(dates[i - 1].date()))

    return np.array(X_list), np.array(y_list), seq_dates


def create_direction_sequences(
    scaled_features: np.ndarray,
    log_returns: np.ndarray,
    dates: pd.DatetimeIndex,
    forecast_days: int = MAX_FORECAST_DAYS,
    window_size: int = WINDOW_SIZE,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Create sliding-window sequences with binary direction targets (1 = up, 0 = down).

    The log_returns array must be aligned to the scaled_features array (length == len(scaled_features)).
    Returns (X, y, sequence_end_dates).
    """
    n_samples = len(scaled_features) - window_size - forecast_days + 1
    if n_samples <= 0:
        raise ValueError("Not enough data for direction sequence creation after windowing.")

    X_list, y_list, seq_dates = [], [], []
    for i in range(window_size, window_size + n_samples):
        X_list.append(scaled_features[i - window_size : i, :])
        future_returns = log_returns[i : i + forecast_days]
        y_list.append((future_returns > 0.0).astype(int))
        seq_dates.append(str(dates[i - 1].date()))

    return np.array(X_list), np.array(y_list), seq_dates


def preprocess(feature_df: pd.DataFrame, forecast_days=MAX_FORECAST_DAYS, scaler=None):
    """
    Build multi-feature windowed train/test data with strict zero-leakage scaler discipline.

    Returns (X_train, X_test, y_train, y_test, scaler, train_dates, test_dates).
    """
    feature_values = feature_df[FEATURES].values
    dates = feature_df.index
    close_idx = FEATURES.index("Close")

    n_samples = len(feature_values) - WINDOW_SIZE - forecast_days + 1
    if n_samples <= 0:
        raise ValueError("Not enough data for training after windowing.")

    split = int(n_samples * TRAIN_SPLIT)
    # A multi-step target beginning at sample ``i`` covers raw rows
    # ``WINDOW_SIZE + i : WINDOW_SIZE + i + forecast_days``.  Purge the
    # boundary samples so a target date used by the training partition can
    # never also appear in the diagnostic test partition.
    train_count = split - forecast_days + 1
    if train_count < 1 or split >= n_samples:
        raise ValueError("Not enough data for a leakage-free train/test split.")
    split_raw_idx = split + WINDOW_SIZE

    # ── Fit scaler on training data only to prevent look-ahead bias ────
    # Robust scaling (median/IQR) is used because min/max scaling is overly
    # sensitive to market extremes and regime changes.
    if scaler is None:
        scaler = RobustScaler(quantile_range=ROBUST_SCALER_QUANTILE_RANGE)
        scaler.fit(feature_values[:split_raw_idx])
    scaled = scaler.transform(feature_values)

    X, y, seq_dates = create_sequences(scaled, dates, close_idx, forecast_days, WINDOW_SIZE)

    X_train, X_test = X[:train_count], X[split:]
    y_train, y_test = y[:train_count], y[split:]
    train_dates = seq_dates[:train_count]
    test_dates = seq_dates[split:]

    return X_train, X_test, y_train, y_test, scaler, train_dates, test_dates


def prepare_return_data(feature_df: pd.DataFrame, forecast_days=MAX_FORECAST_DAYS, scaler=None):
    """
    Prepare multi-feature data using direction binary targets.

    Returns (X_train, X_test, y_train, y_test, scaler, train_dates, test_dates).
    """
    feature_values = feature_df[FEATURES].values
    close_values = feature_df["Close"].values
    dates = feature_df.index

    # Calculate daily log returns for the target; align to drop first row
    close_safe = np.maximum(close_values, 1e-8)
    log_returns = np.log(close_safe[1:] / close_safe[:-1])
    aligned_features = feature_values[1:]
    aligned_dates = dates[1:]

    n_samples = len(aligned_features) - WINDOW_SIZE - forecast_days + 1
    if n_samples <= 0:
        raise ValueError("Not enough data for training after windowing.")

    split = int(n_samples * TRAIN_SPLIT)
    train_count = split - forecast_days + 1
    if train_count < 1 or split >= n_samples:
        raise ValueError("Not enough data for a leakage-free train/test split.")
    split_raw_idx = split + WINDOW_SIZE

    # ── Fit scaler on training features only ────
    if scaler is None:
        scaler = RobustScaler(quantile_range=ROBUST_SCALER_QUANTILE_RANGE)
        scaler.fit(aligned_features[:split_raw_idx])

    scaled_features = scaler.transform(aligned_features)

    X, y, seq_dates = create_direction_sequences(
        scaled_features, log_returns, aligned_dates, forecast_days, WINDOW_SIZE
    )

    X_train, X_test = X[:train_count], X[split:]
    y_train, y_test = y[:train_count], y[split:]
    train_dates = seq_dates[:train_count]
    test_dates = seq_dates[split:]

    return X_train, X_test, y_train, y_test, scaler, train_dates, test_dates


def get_raw_feature_arrays(
    feature_df: pd.DataFrame, forecast_days: int = MAX_FORECAST_DAYS
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Return unscaled raw feature values and close-price targets suitable for walk-forward splits.

    Returns (feature_values, y_close_prices, log_returns, dates).
    The caller is responsible for scaler fitting within each fold.
    """
    feature_values = feature_df[FEATURES].values
    close_values = feature_df["Close"].values
    close_safe = np.maximum(close_values, 1e-8)
    log_returns = np.log(close_safe[1:] / close_safe[:-1])
    dates = feature_df.index
    return feature_values, close_values, log_returns, dates


def get_pipeline(ticker: str):
    """Full pipeline: fetch → preprocess → (pipeline_data, raw_prices, dates, metadata)."""
    feature_df, closing_prices, dates, metadata = fetch_data(ticker)
    return preprocess(feature_df), closing_prices, dates, metadata
