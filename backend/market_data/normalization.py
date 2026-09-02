"""Strict provider-independent OHLCV normalization."""

from __future__ import annotations

import numpy as np
import pandas as pd

from market_data.base import MarketDataProviderError

REQUIRED_OHLCV = ("Open", "High", "Low", "Close", "Volume")


def normalize_daily_bars(frame: pd.DataFrame, *, provider: str, symbol: str) -> pd.DataFrame:
    """Return a finite, ordered, timezone-naive daily OHLCV frame."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise MarketDataProviderError(f"{provider} returned no bars for {symbol}")
    data = frame.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    missing = set(REQUIRED_OHLCV).difference(data.columns)
    if missing:
        raise MarketDataProviderError(
            f"{provider} returned an incomplete OHLCV schema for {symbol}"
        )
    index = pd.to_datetime(data.index, errors="coerce", utc=True)
    if index.isna().any():
        raise MarketDataProviderError(f"{provider} returned invalid timestamps for {symbol}")
    data.index = index.tz_convert(None).normalize()
    data = data.loc[~data.index.duplicated(keep="last")].sort_index()
    data = data.loc[:, list(REQUIRED_OHLCV)].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=list(REQUIRED_OHLCV))
    if data.empty or not np.isfinite(data.to_numpy(dtype=float)).all():
        raise MarketDataProviderError(f"{provider} returned non-finite OHLCV data for {symbol}")
    if (data[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise MarketDataProviderError(f"{provider} returned non-positive prices for {symbol}")
    if (data["Volume"] < 0).any():
        raise MarketDataProviderError(f"{provider} returned negative volume for {symbol}")
    return data
