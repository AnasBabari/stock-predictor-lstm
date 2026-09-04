"""Health, readiness, and root service routes."""

from __future__ import annotations

import os
import re
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from config import APP_VERSION, settings
from data_pipeline import market_circuit_breaker, market_data_service
from services.forecast_ledger import LedgerUnavailableError, get_forecast_ledger
from services.live_volatility import SUPPORTED_BASELINES
from services.volatility_contract import AUTO_MODEL_POLICY, VOLATILITY_MODEL_POLICY_VERSION
from services.volatility_snapshot import VOLATILITY_HORIZONS

router = APIRouter(tags=["health"])


def deployment_environment() -> str:
    if os.getenv("IS_PULL_REQUEST", "").strip().lower() in {"1", "true", "yes"}:
        return "preview"
    return (
        settings.deployment_environment
        or os.getenv("RENDER_SERVICE_TYPE")
        or os.getenv("VERCEL_ENV")
        or os.getenv("ENVIRONMENT")
        or "local"
    )


def deployment_commit() -> str | None:
    value = (
        settings.deployment_commit
        or os.getenv("RENDER_GIT_COMMIT")
        or os.getenv("VERCEL_GIT_COMMIT_SHA")
        or os.getenv("GITHUB_SHA")
    )
    if not value or not re.fullmatch(r"[0-9a-fA-F]{7,40}", value):
        return None
    return value[:12].lower()


def deployment_identity() -> dict[str, Any]:
    environment = deployment_environment()
    provider = settings.deployment_provider or ("render" if os.getenv("RENDER") else "unknown")
    return {
        "provider": provider,
        "environment": environment,
        "commit": deployment_commit(),
        "preview": environment.lower() in {"preview", "pr", "pull_request"},
    }


@router.get("/")
def root():
    return {
        "name": "Signal Seven Forecast API",
        "status": "online",
        "version": APP_VERSION,
        "docs": "/docs",
        "redoc": "/redoc",
        "openapi": "/openapi.json",
        "health": "/health",
        "readiness": "/ready",
    }


@router.get("/health")
def health():
    """O(1) Liveness probe."""
    return {"status": "ok", "version": APP_VERSION, "deployment": deployment_identity()}


@router.get("/ready")
def ready():
    """Readiness checks market data and required forecast-ledger connectivity.

    Liveness remains deliberately independent of external services.  A
    configured PostgreSQL ledger is a hard production dependency; an
    inaccessible or missing durable store therefore keeps this endpoint at
    503 instead of allowing unpersisted live observations.
    """
    provider_ready, upstream = market_data_service.readiness()
    circuit_ready, circuit = market_circuit_breaker.is_ready()
    upstream["circuit"] = circuit

    configured_database_url = (
        settings.forecast_ledger_database_url
        or os.getenv("DATABASE_URL")
        or os.getenv("FORECAST_LEDGER_DATABASE_URL")
    )
    ledger_required = bool(configured_database_url or settings.forecast_ledger_database_required)
    try:
        ledger = get_forecast_ledger()
        ledger_status = ledger.check_connection()
    except LedgerUnavailableError:
        ledger_status = {
            "status": "unavailable",
            "backend": "postgresql"
            if configured_database_url or settings.forecast_ledger_database_required
            else "sqlite",
            "durable": bool(configured_database_url),
            "required": ledger_required,
        }
    ledger_status["required"] = ledger_required
    ledger_ready = ledger_status.get("status") == "available"
    is_ready = provider_ready and circuit_ready and (ledger_ready or not ledger_required)
    dependencies = {
        "market_data": upstream,
        "forecast_ledger": ledger_status,
    }

    content = {
        "status": "ready" if is_ready else "degraded",
        "version": APP_VERSION,
        "deployment": deployment_identity(),
        "dependencies": dependencies,
    }
    return JSONResponse(status_code=200 if is_ready else 503, content=content)


@router.get("/models")
def models_discovery():
    """Discover active volatility forecasting models and capabilities."""
    return {
        "status": "online",
        "simple_price_forecast": {
            "status": "available",
            "endpoint": "/api/v1/forecast",
            "horizon": 7,
            "supported_tickers": ["AAPL", "GOOGL", "MSFT", "NVDA", "TSLA"],
            "learned_candidates": ["ridge", "random_forest"],
            "evaluation": "chronological_70_15_15_with_7_session_purge",
            "news_endpoint": "/api/v1/news",
            "news_role": "context_only",
        },
        "volatility_forecasting": {
            "status": "available",
            "endpoint": "/api/v1/volatility/forecast",
            "public_forecast_mode": "read_only_preview",
            "live_collection_endpoint": "/api/v1/volatility/collect",
            "live_collection_authentication": "bearer_token_required",
            "metric_source": "baseline_definition",
            "supported_horizons": list(VOLATILITY_HORIZONS),
            "supported_models": list(SUPPORTED_BASELINES),
            "model_policy_version": VOLATILITY_MODEL_POLICY_VERSION,
            "auto_model_policy": {
                str(horizon): model for horizon, model in AUTO_MODEL_POLICY.items()
            },
        },
        "model_storage": {
            "required": False,
            "status": "not_required",
        },
        "forecast_ledger": {
            "configured_backend": "postgresql"
            if settings.forecast_ledger_database_url
            or os.getenv("DATABASE_URL")
            or os.getenv("FORECAST_LEDGER_DATABASE_URL")
            or settings.forecast_ledger_database_required
            else "sqlite",
            "durable_required": bool(
                settings.forecast_ledger_database_url
                or os.getenv("DATABASE_URL")
                or os.getenv("FORECAST_LEDGER_DATABASE_URL")
                or settings.forecast_ledger_database_required
            ),
            "note": "SQLite is for local/test use; genuine live records are written only by the authenticated collector.",
        },
    }
