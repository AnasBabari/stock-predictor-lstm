"""Market data caching and ingestion for point-in-time Nasdaq-100 constituents."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "ndx100" / "cache"

TICKER_ALIASES: dict[str, str] = {
    # Providers expose the current symbol across the full adjusted history.
    # Membership identities remain point-in-time (FB/FISV before their changes).
    "FB": "META",
    "FISV": "FI",
}


def normalize_market_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw yfinance download into standardized OHLCV frame."""
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    col_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    present_cols = [c for c in col_map if c in raw.columns]
    if "Close" not in present_cols or "Volume" not in present_cols:
        return pd.DataFrame()

    frame = raw[present_cols].rename(columns=col_map).dropna(subset=["close", "volume"]).copy()
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
    frame = frame.sort_index()
    return frame[~frame.index.duplicated(keep="last")]


def download_ticker_history(
    ticker: str,
    start_date: str = "2015-01-01",
    end_date: str = "2026-08-28",
) -> pd.DataFrame:
    """Download single ticker daily adjusted history from provider."""
    symbol = TICKER_ALIASES.get(ticker.upper(), ticker.upper())
    try:
        raw = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
        )
        if raw is None or raw.empty:
            return pd.DataFrame()
        return normalize_market_frame(raw)
    except Exception as exc:
        logger.warning("Failed to download history for %s: %s", ticker, exc)
        return pd.DataFrame()


def cache_ticker_history(
    ticker: str,
    frame: pd.DataFrame,
    cache_dir: Path | None = None,
) -> Path | None:
    """Save normalized market history to local parquet cache."""
    if frame.empty:
        return None
    target_dir = cache_dir or DEFAULT_CACHE_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / f"{ticker.upper()}.parquet"
    frame.to_parquet(target_file)
    return target_file


def load_ticker_history(
    ticker: str,
    cache_dir: Path | None = None,
) -> pd.DataFrame | None:
    """Load cached history for ticker if present, otherwise None."""
    target_dir = cache_dir or DEFAULT_CACHE_DIR
    target_file = target_dir / f"{ticker.upper()}.parquet"
    if not target_file.is_file():
        return None
    try:
        df = pd.read_parquet(target_file)
        df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
        return df.sort_index()
    except Exception as exc:
        logger.warning("Failed to load cached parquet for %s: %s", ticker, exc)
        return None


def download_and_cache_universe(
    tickers: Sequence[str],
    cache_dir: Path | None = None,
    start_date: str = "2015-01-01",
    end_date: str = "2026-08-28",
    force_redownload: bool = False,
) -> dict[str, pd.DataFrame]:
    """Download and locally cache history for all universe tickers."""
    target_dir = cache_dir or DEFAULT_CACHE_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, pd.DataFrame] = {}
    to_download: list[str] = []

    for t in tickers:
        t_upper = t.upper()
        if not force_redownload:
            cached = load_ticker_history(t_upper, target_dir)
            if cached is not None and len(cached) > 200:
                results[t_upper] = cached
                continue
        to_download.append(t_upper)

    if to_download:
        logger.info("Downloading %d tickers via batch download...", len(to_download))
        # Batch download for speed
        download_symbols = [TICKER_ALIASES.get(t, t) for t in to_download]
        try:
            raw_batch = yf.download(
                download_symbols,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False,
                group_by="ticker",
            )
            for orig_t, sym in zip(to_download, download_symbols, strict=True):
                try:
                    if len(to_download) == 1:
                        frame = normalize_market_frame(raw_batch)
                    else:
                        sub = (
                            raw_batch[sym] if sym in raw_batch.columns.levels[0] else pd.DataFrame()
                        )
                        frame = normalize_market_frame(sub)
                    if not frame.empty and len(frame) >= 60:
                        cache_ticker_history(orig_t, frame, target_dir)
                        results[orig_t] = frame
                    else:
                        # Try individual fallback
                        individual = download_ticker_history(orig_t, start_date, end_date)
                        if not individual.empty and len(individual) >= 60:
                            cache_ticker_history(orig_t, individual, target_dir)
                            results[orig_t] = individual
                except Exception as e:
                    logger.warning("Error processing %s (%s): %s", orig_t, sym, e)
        except Exception as exc:
            logger.warning("Batch download failed, falling back to sequential: %s", exc)
            for orig_t in to_download:
                individual = download_ticker_history(orig_t, start_date, end_date)
                if not individual.empty and len(individual) >= 60:
                    cache_ticker_history(orig_t, individual, target_dir)
                    results[orig_t] = individual

    return results
