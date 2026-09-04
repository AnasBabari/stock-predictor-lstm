"""Public endpoints for the simplified five-ticker forecasting product."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

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
        frame = _download_ohlcv(symbol)
        if model == "auto":
            return train_and_forecast(symbol, frame)
        return train_and_forecast(symbol, frame, model_name=model)
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
