# Fetches and preprocesses stock data
#
# Fixes applied:
#   3.1  Scaler fit on training data only (no look-ahead bias)
#   2.2  Return date index alongside prices (eliminates double download)
#   3.2  Multi-step targets for direct forecasting

import numpy as np
import yfinance as yf  # type: ignore[import-untyped]
from sklearn.preprocessing import MinMaxScaler  # type: ignore[import-untyped]

from config import HISTORICAL_YEARS, MAX_FORECAST_DAYS, TRAIN_SPLIT, WINDOW_SIZE


def fetch_data(ticker: str):
    """Download historical prices. Returns (closing_prices, date_index)."""
    data = yf.download(ticker, period=f"{HISTORICAL_YEARS}y", progress=False)
    data = data.dropna()
    min_rows = WINDOW_SIZE + MAX_FORECAST_DAYS + 10
    if len(data) < min_rows:
        raise ValueError(
            f"Not enough historical data for {ticker}. " f"Need at least {min_rows} trading days."
        )
    closing_prices = data["Close"].values.reshape(-1, 1)
    return closing_prices, data.index


def preprocess(closing_prices, forecast_days=MAX_FORECAST_DAYS):
    """
    Build windowed train/test data with proper scaler discipline.

    Key fix (3.1): the MinMaxScaler is fit ONLY on the training portion
    of raw prices, preventing future-price leakage into the scaler's
    data_min_ / data_max_.

    Key fix (3.2): targets are multi-step — each sample's y is the next
    `forecast_days` scaled values, enabling direct (non-recursive) output.
    """
    n_samples = len(closing_prices) - WINDOW_SIZE - forecast_days + 1
    if n_samples <= 0:
        raise ValueError("Not enough data for training after windowing.")

    split = int(n_samples * TRAIN_SPLIT)
    split_raw_idx = split + WINDOW_SIZE  # boundary in the raw price array

    # ── Fit scaler on training data only (3.1) ───────────────────────
    scaler = MinMaxScaler()
    scaler.fit(closing_prices[:split_raw_idx])
    scaled = scaler.transform(closing_prices)

    # ── Create multi-step windows (3.2) ──────────────────────────────
    X, y = [], []
    for i in range(WINDOW_SIZE, WINDOW_SIZE + n_samples):
        X.append(scaled[i - WINDOW_SIZE : i, 0])
        y.append(scaled[i : i + forecast_days, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    return X_train, X_test, y_train, y_test, scaler


def get_pipeline(ticker: str):
    """Full pipeline: fetch → preprocess → (pipeline_data, raw_prices, dates)."""
    closing_prices, dates = fetch_data(ticker)
    return preprocess(closing_prices), closing_prices, dates
