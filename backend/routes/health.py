"""Health, readiness, and root service routes."""

from __future__ import annotations

import os
import re
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from config import APP_VERSION, settings
from data_pipeline import market_circuit_breaker, market_data_service

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
        "name": "Stock Volatility Forecasting API",
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
    """Readiness checks market data ingestion connectivity."""
    provider_ready, upstream = market_data_service.readiness()
    circuit_ready, circuit = market_circuit_breaker.is_ready()
    upstream["circuit"] = circuit
    is_ready = provider_ready and circuit_ready
    dependencies = {
        "market_data": upstream,
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
        "volatility_forecasting": {
            "status": "available",
            "endpoint": "/api/v1/volatility/forecast",
            "metric_source": "baseline_definition",
            "supported_horizons": [1, 3, 5, 7, 14, 30],
            "supported_models": [
                "auto",
                "rolling_mean",
                "har_rv",
                "ewma",
                "persistence",
                "garch_11",
            ],
        },
        "model_storage": {
            "required": False,
            "status": "not_required",
        },
    }
