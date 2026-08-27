"""Exchange-close alignment between point-in-time news and market origins."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from .news import NewsEvent, NewsFeatureMatrix, NewsOrigin, aggregate_news_features

V8_MIC_CALENDAR_NAMES = {
    "XNAS": "NASDAQ",
    "XNYS": "NYSE",
    "XLON": "LSE",
}


def validate_news_coverage(
    manifest: Mapping[str, object],
    cutoffs: np.ndarray,
    *,
    lookback_days: int = 7,
) -> None:
    """Prove that zero-valued news features mean no events, not missing data."""
    if lookback_days < 1:
        raise ValueError("news lookback must be positive")
    values = np.asarray(cutoffs, dtype="datetime64[ns]")
    if values.ndim != 1 or len(values) == 0 or np.isnat(values).any():
        raise ValueError("news coverage validation requires finite cutoff timestamps")
    try:
        coverage_start = pd.Timestamp(manifest["coverage_start"])
        coverage_end = pd.Timestamp(manifest["coverage_end_exclusive"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("news manifest coverage timestamps are missing or invalid") from error
    if coverage_start.tzinfo is None or coverage_end.tzinfo is None:
        raise ValueError("news manifest coverage timestamps must be timezone-aware")
    coverage_start = coverage_start.tz_convert("UTC")
    coverage_end = coverage_end.tz_convert("UTC")
    earliest_cutoff = pd.Timestamp(values.min()).tz_localize("UTC")
    latest_cutoff = pd.Timestamp(values.max()).tz_localize("UTC")
    required_start = earliest_cutoff - pd.Timedelta(days=lookback_days)
    if coverage_start > required_start:
        raise ValueError("news provider coverage does not include the full initial lookback")
    if coverage_end <= latest_cutoff:
        raise ValueError("news provider coverage ends before the final forecast cutoff")


def market_close_news_origins(
    tickers: np.ndarray,
    origin_dates: np.ndarray,
    *,
    calendar_name: str = "NYSE",
    calendar_by_ticker: Mapping[str, str] | None = None,
) -> tuple[NewsOrigin, ...]:
    """Map every session date to its actual UTC exchange close, including DST."""
    symbols = np.asarray(tickers, dtype=str)
    dates = np.asarray(origin_dates, dtype="datetime64[D]")
    if symbols.ndim != 1 or dates.shape != symbols.shape or len(symbols) == 0:
        raise ValueError("news alignment requires matched non-empty ticker and date vectors")
    normalized_calendars = (
        {
            str(ticker).strip().upper(): str(name).strip()
            for ticker, name in calendar_by_ticker.items()
        }
        if calendar_by_ticker is not None
        else None
    )
    unique_tickers = {str(ticker).strip().upper() for ticker in symbols}
    if normalized_calendars is not None and set(normalized_calendars) != unique_tickers:
        raise ValueError("calendar mapping must exactly cover news-origin tickers")
    row_calendars = np.asarray(
        [
            normalized_calendars[str(ticker).strip().upper()]
            if normalized_calendars is not None
            else calendar_name
            for ticker in symbols
        ],
        dtype=str,
    )
    close_by_calendar_date: dict[tuple[str, np.datetime64], pd.Timestamp] = {}
    for selected_calendar in sorted(set(row_calendars)):
        mask = row_calendars == selected_calendar
        calendar_dates = np.unique(dates[mask])
        try:
            calendar = mcal.get_calendar(selected_calendar)
        except RuntimeError as error:
            raise ValueError(f"unknown market calendar {selected_calendar!r}") from error
        schedule = calendar.schedule(
            start_date=str(calendar_dates[0]),
            end_date=str(calendar_dates[-1]),
        )
        closes = {
            np.datetime64(pd.Timestamp(session).date()): pd.Timestamp(close).tz_convert("UTC")
            for session, close in schedule["market_close"].items()
        }
        missing = sorted({str(date) for date in calendar_dates if date not in closes})
        if missing:
            preview = ", ".join(missing[:3])
            raise ValueError(
                f"origin dates are absent from {selected_calendar} schedule: {preview}"
            )
        close_by_calendar_date.update(
            {(selected_calendar, date): close for date, close in closes.items()}
        )
    return tuple(
        NewsOrigin(ticker, close_by_calendar_date[(selected_calendar, date)])
        for ticker, date, selected_calendar in zip(
            symbols,
            dates,
            row_calendars,
            strict=True,
        )
    )


def aggregate_news_for_market_rows(
    events: Sequence[NewsEvent],
    tickers: np.ndarray,
    origin_dates: np.ndarray,
    *,
    exposure_map: Mapping[str, Mapping[str, float]] | None = None,
    calendar_name: str = "NYSE",
    calendar_by_ticker: Mapping[str, str] | None = None,
) -> NewsFeatureMatrix:
    origins = market_close_news_origins(
        tickers,
        origin_dates,
        calendar_name=calendar_name,
        calendar_by_ticker=calendar_by_ticker,
    )
    return aggregate_news_features(events, origins, exposure_map=exposure_map)


def v8_calendar_map(ticker_exchange_map: Mapping[str, str]) -> dict[str, str]:
    """Translate verified v8 MIC identities into exchange calendar names."""

    result: dict[str, str] = {}
    for ticker, raw_mic in ticker_exchange_map.items():
        normalized_ticker = str(ticker).strip().upper()
        mic = str(raw_mic).strip().upper()
        try:
            result[normalized_ticker] = V8_MIC_CALENDAR_NAMES[mic]
        except KeyError as error:
            raise ValueError(f"unsupported v8 exchange MIC {mic!r}") from error
    return result
