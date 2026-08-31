"""StockLSTM API — FastAPI backend application and router orchestrator.

This application acts as a high-reliability data and prediction coordination service:
- Generates stationary, leakage-safe snapshots for diagnostics and offline research.
- Serves a causal volatility baseline and optional signed compatibility models.
- Enforces strict rate limiting via trusted proxy inspection and process-bounded concurrency.
"""

from __future__ import annotations

import logging

import yfinance as yf  # type: ignore[import-untyped]
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from calendars import future_trading_dates
from config import (
    APP_VERSION,
    DEFAULT_FORECAST_DAYS,
    FEATURES,
    MAX_FORECAST_DAYS,
    SCHEMA_VERSION,
    WINDOW_SIZE,
    settings,
)
from data_pipeline import (
    MarketDataUnavailable,
    MarketTransportError,
    fetch_data,
    market_circuit_breaker,
)
from features.market import MarketContextUnavailable
from news_features import get_live_financial_sentiment as get_financial_sentiment
from routes.common import (
    VALID_MODEL_TYPES,
    _is_trusted_ip,
    _normalise_ip,
    _trusted_proxy_ips,
    limiter,
    rate_limit_identity,
    validate_model_type,
    validate_ticker,
)
from routes.forecasts import (
    FORECAST_UNAVAILABLE_RESPONSE,
    ArtifactValidationError,
    PredictionJob,
    PredictionStatusRegistry,
    ServiceBusyError,
    TrainingCapacityError,
    WorkCoordinator,
    _await_prediction,
    _direction_prediction_pipeline,
    _fetch_snapshot,
    _get_fresh_cached_response,
    _measure,
    _parse_request_id,
    _predict_cache,
    _predict_cache_lock,
    _price_prediction_pipeline,
    _resolve_runtime_commit,
    _status_registry,
    _validated_future_dates,
    _with_execution_metadata,
    _work_coordinator,
    get_runtime_metadata,
)
from routes.forecasts import (
    router as forecasts_router,
)
from routes.health import (
    deployment_commit as _deployment_commit,
)
from routes.health import (
    deployment_environment as _deployment_environment,
)
from routes.health import (
    deployment_identity as _deployment_identity,
)
from routes.health import (
    router as health_router,
)
from routes.market import (
    _info_cache,
    _info_cache_lock,
)
from routes.market import (
    router as market_router,
)
from routes.models import (
    ModelPerformanceResponse,
    load_cross_validation,
    load_metadata,
    load_metrics,
    load_validation_results,
)
from routes.models import (
    router as models_router,
)
from routes.training_data import (
    _SNAPSHOT_CACHE_MAX,
    _SNAPSHOT_CACHE_TTL,
    _execute_snapshot_build,
    _in_flight_tasks,
    _snapshot_cache,
    _snapshot_lock,
    _training_semaphore,
)
from routes.training_data import (
    router as training_data_router,
)
from routes.volatility import router as volatility_router
from routes.volatility_v2 import router as volatility_v2_router
from server_models.api import router as server_forecasts_router
from server_models.response_models import (
    DirectionForecastResponse,
    PredictionStatusResponse,
    PriceForecastResponse,
)
from services.baselines import base_rate_direction_forecast, persistence_price_forecast
from services.training_data import build_training_snapshot

__all__ = [
    "APP_VERSION",
    "ArtifactValidationError",
    "DEFAULT_FORECAST_DAYS",
    "DirectionForecastResponse",
    "FEATURES",
    "FORECAST_UNAVAILABLE_RESPONSE",
    "MAX_FORECAST_DAYS",
    "MarketContextUnavailable",
    "MarketDataUnavailable",
    "MarketTransportError",
    "ModelPerformanceResponse",
    "PredictionJob",
    "PredictionStatusResponse",
    "PredictionStatusRegistry",
    "PriceForecastResponse",
    "Request",
    "SCHEMA_VERSION",
    "ServiceBusyError",
    "TrainingCapacityError",
    "VALID_MODEL_TYPES",
    "WINDOW_SIZE",
    "WorkCoordinator",
    "_SNAPSHOT_CACHE_MAX",
    "_SNAPSHOT_CACHE_TTL",
    "_await_prediction",
    "_deployment_commit",
    "_deployment_environment",
    "_deployment_identity",
    "_direction_prediction_pipeline",
    "_execute_snapshot_build",
    "_fetch_snapshot",
    "_get_fresh_cached_response",
    "_in_flight_tasks",
    "_info_cache",
    "_info_cache_lock",
    "_is_trusted_ip",
    "_measure",
    "_normalise_ip",
    "_parse_request_id",
    "_predict_cache",
    "_predict_cache_lock",
    "_price_prediction_pipeline",
    "_record_upstream",
    "_resolve_runtime_commit",
    "_snapshot_cache",
    "_snapshot_lock",
    "_status_registry",
    "_training_semaphore",
    "_trusted_proxy_ips",
    "_validated_future_dates",
    "_with_execution_metadata",
    "_work_coordinator",
    "app",
    "base_rate_direction_forecast",
    "build_training_snapshot",
    "fetch_data",
    "future_trading_dates",
    "get_financial_sentiment",
    "get_manifest",
    "get_runtime_metadata",
    "limiter",
    "load_cross_validation",
    "load_fresh_artifact",
    "load_metadata",
    "load_metrics",
    "load_validation_results",
    "market_circuit_breaker",
    "persistence_price_forecast",
    "predict_direction",
    "predict_future",
    "rate_limit_identity",
    "settings",
    "validate_model_type",
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
app = FastAPI(title="StockLSTM API", version=APP_VERSION)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    headers = (
        {"Cache-Control": "no-store"}
        if request.url.path.startswith("/api/v1/prediction-status/")
        else None
    )
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please wait before trying again."},
        headers=headers,
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
app.include_router(training_data_router)
app.include_router(market_router)
app.include_router(models_router)
app.include_router(forecasts_router)
app.include_router(server_forecasts_router)
app.include_router(volatility_v2_router)
app.include_router(volatility_router)


# ── Compatibility Helpers for Legacy Callers & Tests ────────────────
def load_fresh_artifact(*_args, **_kwargs):
    """Server-side artifact loading is intentionally disabled in production."""
    raise ArtifactValidationError("Server-side model artifacts are disabled.")


def get_manifest() -> dict:
    return {}


def predict_future(*_args, **_kwargs):
    raise ArtifactValidationError("Server-side model artifacts are disabled.")


def predict_direction(*_args, **_kwargs):
    raise ArtifactValidationError("Server-side model artifacts are disabled.")


def _record_upstream(status: str, error: str | None = None) -> None:
    """Reset or update upstream circuit breaker state for tests/diagnostics."""
    if status == "available":
        market_circuit_breaker.record_success("__global__")
    elif status == "unavailable":
        market_circuit_breaker.record_failure(
            "__global__", MarketTransportError(error or "Unavailable")
        )
