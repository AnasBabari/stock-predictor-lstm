# Fetches and preprocesses multi-feature stock data
#
# Features included: OHLCV, Technical Indicators, Market Returns, Cyclical Calendar
# Phase 3: Decoupled sequence generation for walk-forward validation support.

import hashlib

import numpy as np
import pandas as pd
import yfinance as yf  # type: ignore[import-untyped]
from sklearn.preprocessing import MinMaxScaler  # type: ignore[import-untyped]

from config import (
    FEATURES,
    HISTORICAL_YEARS,
    MAX_FORECAST_DAYS,
    SCHEMA_VERSION,
    TRAIN_SPLIT,
    WINDOW_SIZE,
)
from features.pipeline import build_features


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
    feature_metadata["feature_schema_version"] = SCHEMA_VERSION
    closing_prices = feature_df["Close"].to_numpy()

    return feature_df, closing_prices, feature_df.index, feature_metadata


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
    if scaler is None:
        scaler = MinMaxScaler()
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
        scaler = MinMaxScaler()
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
