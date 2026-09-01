"""Stock Volatility Forecasting API — FastAPI backend application.

Serves causal equity volatility forecasts, scenario cones, company metadata,
and immutable forecast ledger track records.
"""

from __future__ import annotations

import logging

import yfinance as yf  # type: ignore[import-untyped]
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from config import APP_VERSION, settings
from data_pipeline import (
    MarketDataUnavailable,
    MarketTransportError,
    market_circuit_breaker,
)
from routes.common import limiter, validate_ticker
from routes.health import deployment_identity as _deployment_identity
from routes.health import router as health_router
from routes.market import router as market_router
from routes.volatility import router as volatility_router

__all__ = [
    "APP_VERSION",
    "MarketDataUnavailable",
    "MarketTransportError",
    "app",
    "limiter",
    "validate_ticker",
    "yf",
]

# ── Logging ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-20s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Application Setup ────────────────────────────────────────────────
app = FastAPI(
    title="Stock Volatility Forecasting API",
    version=APP_VERSION,
    description="Causal forward volatility forecasting platform with immutable track record ledger.",
)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please wait before trying again."},
    )


# ── CORS Middleware ──────────────────────────────────────────────────
_preview_cors_regex = (
    settings.preview_cors_origin_regex
    if _deployment_identity()["preview"] and settings.preview_cors_origin_regex
    else None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_origin_regex=_preview_cors_regex,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["Content-Type", "X-Prediction-Request-ID", "X-Client-Class"],
)


# ── Security Headers ─────────────────────────────────────────────────
@app.middleware("http")
async def _security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    return response


# ── Router Registration ──────────────────────────────────────────────
app.include_router(health_router)
app.include_router(market_router)
app.include_router(volatility_router)


def _record_upstream(status: str, error: str | None = None) -> None:
    """Reset or update upstream circuit breaker state for tests/diagnostics."""
    if status == "available":
        market_circuit_breaker.record_success("__global__")
    elif status == "unavailable":
        market_circuit_breaker.record_failure(
            "__global__", MarketTransportError(error or "Unavailable")
        )
