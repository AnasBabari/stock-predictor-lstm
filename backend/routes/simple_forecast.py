"""Public endpoints for the simplified five-ticker forecasting product."""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from config import settings
from data_pipeline import (
    MarketDataUnavailable,
    MarketTransportError,
    UnknownTickerError,
    _download_ohlcv,
)
from routes.common import limiter, validate_ticker
from services.market_news import fetch_recent_news
from services.simple_forecast import FORECAST_DAYS, SUPPORTED_TICKERS, train_and_forecast

router = APIRouter(tags=["simple-forecast"])


def _supported_ticker(value: str) -> str:
    symbol = validate_ticker(value)
    if symbol not in SUPPORTED_TICKERS:
        raise HTTPException(
            status_code=400,
            detail=f"This first benchmark supports: {', '.join(SUPPORTED_TICKERS)}.",
        )
    return symbol


@router.get("/api/v1/forecast")
@limiter.limit("12/minute")
def forecast(
    request: Request,
    ticker: str = Query(default="MSFT", min_length=1, max_length=12),
    days: int = Query(default=FORECAST_DAYS, ge=1, le=FORECAST_DAYS),
    model: str = Query(default="auto", min_length=1, max_length=30),
) -> Any:
    """Train/select from historical bars and return a learned seven-day path."""
    symbol = _supported_ticker(ticker)
    if days != FORECAST_DAYS:
        raise HTTPException(
            status_code=400, detail="This first benchmark uses a fixed 7-day horizon."
        )
    try:
        route_started = time.perf_counter()
        t_data = time.perf_counter()
        frame = _download_ohlcv(symbol)
        data_ms = (time.perf_counter() - t_data) * 1000.0
        t_train = time.perf_counter()
        if model == "auto":
            result = train_and_forecast(symbol, frame)
        else:
            result = train_and_forecast(symbol, frame, model_name=model)
        train_ms = (time.perf_counter() - t_train) * 1000.0
        total_ms = (time.perf_counter() - route_started) * 1000.0
        response = JSONResponse(content=result)
        # Machine-readable totals; per-stage breakdown (features/select/
        # infer, cache hit vs train) is logged server-side by
        # train_and_forecast. No body fields change.
        response.headers["Server-Timing"] = (
            f"data;dur={data_ms:.0f}, train_or_cache;dur={train_ms:.0f}, total;dur={total_ms:.0f}"
        )
        return response
    except UnknownTickerError as err:
        raise HTTPException(status_code=404, detail="No market data is available.") from err
    except MarketTransportError as err:
        raise HTTPException(
            status_code=503, detail="Market data is temporarily unavailable."
        ) from err
    except MarketDataUnavailable as err:
        raise HTTPException(
            status_code=422, detail="Not enough valid market history is available."
        ) from err
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err)) from err


@router.get("/api/v1/news")
@limiter.limit("30/minute")
def news(
    request: Request,
    ticker: str = Query(default="MSFT", min_length=1, max_length=12),
) -> dict[str, Any]:
    """Return recent ticker headlines as explicitly non-model context."""
    symbol = _supported_ticker(ticker)
    return {
        "ticker": symbol,
        **fetch_recent_news(
            symbol,
            key_id=settings.alpaca_api_key_id,
            secret_key=settings.alpaca_api_secret_key,
            base_url=settings.alpaca_data_base_url,
            timeout_seconds=settings.market_data_timeout_seconds,
        ),
    }
