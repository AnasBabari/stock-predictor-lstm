"""Exchange-close alignment between point-in-time news and market origins."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from .news import NewsEvent, NewsFeatureMatrix, NewsOrigin, aggregate_news_features


def market_close_news_origins(
    tickers: np.ndarray,
    origin_dates: np.ndarray,
    *,
    calendar_name: str = "NYSE",
) -> tuple[NewsOrigin, ...]:
    """Map every session date to its actual UTC exchange close, including DST."""
    symbols = np.asarray(tickers, dtype=str)
    dates = np.asarray(origin_dates, dtype="datetime64[D]")
    if symbols.ndim != 1 or dates.shape != symbols.shape or len(symbols) == 0:
        raise ValueError("news alignment requires matched non-empty ticker and date vectors")
    unique_dates = np.unique(dates)
    calendar = mcal.get_calendar(calendar_name)
    schedule = calendar.schedule(
        start_date=str(unique_dates[0]),
        end_date=str(unique_dates[-1]),
    )
    close_by_date = {
        np.datetime64(pd.Timestamp(session).date()): pd.Timestamp(close).tz_convert("UTC")
        for session, close in schedule["market_close"].items()
    }
    missing = sorted({str(date) for date in unique_dates if date not in close_by_date})
    if missing:
        preview = ", ".join(missing[:3])
        raise ValueError(f"origin dates are absent from {calendar_name} schedule: {preview}")
    return tuple(
        NewsOrigin(ticker, close_by_date[date]) for ticker, date in zip(symbols, dates, strict=True)
    )


def aggregate_news_for_market_rows(
    events: Sequence[NewsEvent],
    tickers: np.ndarray,
    origin_dates: np.ndarray,
    *,
    exposure_map: Mapping[str, Mapping[str, float]] | None = None,
    calendar_name: str = "NYSE",
) -> NewsFeatureMatrix:
    origins = market_close_news_origins(
        tickers,
        origin_dates,
        calendar_name=calendar_name,
    )
    return aggregate_news_features(events, origins, exposure_map=exposure_map)
