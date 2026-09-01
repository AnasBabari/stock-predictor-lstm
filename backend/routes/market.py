"""Market data, ticker search, and company info endpoints."""

from __future__ import annotations

import logging
import re
import threading

import yfinance as default_yf  # type: ignore[import-untyped]
from cachetools import TTLCache
from fastapi import APIRouter, HTTPException, Query, Request

from config import settings
from routes.common import limiter, validate_ticker

logger = logging.getLogger(__name__)
router = APIRouter(tags=["market"])

_info_cache_lock = threading.Lock()
_info_cache: TTLCache = TTLCache(
    maxsize=settings.cache_max_size,
    ttl=settings.info_cache_ttl,
)


@router.get("/api/v1/search")
@limiter.limit("30/minute")
def search(
    request: Request,
    query: str = Query(..., min_length=1, max_length=100),
):
    import api

    yf = getattr(api, "yf", default_yf)

    exact_symbol = query.strip().upper()
    fallback = []
    if re.fullmatch(r"[A-Z0-9.\-]{1,12}", exact_symbol):
        fallback.append({"ticker": exact_symbol, "name": exact_symbol, "type": "SYMBOL"})
    if fallback and query.strip() == exact_symbol:
        return {"results": fallback}
    try:
        results = yf.Search(query, max_results=8)
        suggestions = []
        for r in results.quotes:
            if r.get("quoteType") in ("EQUITY", "ETF"):
                suggestions.append(
                    {
                        "ticker": r.get("symbol", ""),
                        "name": r.get("longname") or r.get("shortname", ""),
                        "type": r.get("quoteType", ""),
                    }
                )
        seen = {item["ticker"] for item in suggestions}
        return {"results": suggestions + [item for item in fallback if item["ticker"] not in seen]}
    except Exception as err:
        if fallback:
            logger.warning("Autocomplete upstream unavailable; returning exact symbol fallback")
            return {"results": fallback, "degraded": True}
        logger.exception("Error in /api/v1/search")
        raise HTTPException(
            status_code=500,
            detail="Search failed. Please try again later.",
        ) from err


@router.get("/api/v1/info")
@limiter.limit("20/minute")
def stock_info(request: Request, ticker: str = "AAPL"):
    """Return rich metadata for a ticker."""
    import api

    yf = getattr(api, "yf", default_yf)
    cache = getattr(api, "_info_cache", _info_cache)
    lock = getattr(api, "_info_cache_lock", _info_cache_lock)

    ticker = validate_ticker(ticker)

    with lock:
        cached = cache.get(ticker)
    if cached:
        return cached

    try:
        info = yf.Ticker(ticker).info
        data = {
            "ticker": ticker,
            "name": info.get("longName") or info.get("shortName", ticker),
            "exchange": info.get("exchange", "—"),
            "currency": info.get("currency", "USD"),
            "marketCap": info.get("marketCap"),
            "peRatio": info.get("trailingPE"),
            "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
            "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
            "avgVolume": info.get("averageVolume"),
            "dayHigh": info.get("dayHigh"),
            "dayLow": info.get("dayLow"),
            "previousClose": info.get("previousClose"),
            "sector": info.get("sector", "—"),
            "industry": info.get("industry", "—"),
        }
        with lock:
            cache[ticker] = data
        return data
    except Exception as err:
        logger.exception("Error fetching info for %s", ticker)
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch stock info. Please try again later.",
        ) from err
