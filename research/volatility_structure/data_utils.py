"""Data normalization and universe constants for volatility research."""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_TICKERS = ("AAPL", "GOOGL", "MSFT", "NVDA", "TSLA")

TRI_EXCHANGE_TICKERS = (
    # NASDAQ (10)
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA",
    "AMD",
    "COST",
    "QCOM",
    # NYSE (10)
    "JPM",
    "XOM",
    "WMT",
    "JNJ",
    "CAT",
    "KO",
    "NEE",
    "DIS",
    "BAC",
    "GE",
    # LSE (10)
    "SHEL.L",
    "AZN.L",
    "HSBA.L",
    "BP.L",
    "ULVR.L",
    "GSK.L",
    "RIO.L",
    "BATS.L",
    "BARC.L",
    "DGE.L",
)


def _normalise_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Standardise and validate daily OHLCV bars for volatility feature extraction."""
    data = frame.copy()
    data.columns = [str(column).title() for column in data.columns]
    required = ("Open", "High", "Low", "Close", "Volume")
    if not set(required).issubset(data.columns):
        raise ValueError(f"OHLCV frame is missing {sorted(set(required) - set(data.columns))}")
    data = data.loc[:, required].apply(pd.to_numeric, errors="coerce")
    data.index = pd.to_datetime(data.index, errors="coerce").tz_localize(None)
    data = data.loc[~data.index.isna()]
    data = data.loc[~data.index.duplicated(keep="last")].sort_index()
    if len(data) < 500 or not np.isfinite(data.to_numpy(dtype=np.float64)).all():
        raise ValueError("OHLCV frame is too short or contains non-finite values")
    if (data[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("OHLC prices must be positive")
    data["High"] = np.maximum(data["High"], data[["Open", "Close", "Low"]].max(axis=1))
    data["Low"] = np.minimum(data["Low"], data[["Open", "Close", "High"]].min(axis=1))
    return data
