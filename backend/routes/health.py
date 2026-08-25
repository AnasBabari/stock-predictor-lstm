"""Health, readiness, and root service routes."""

from __future__ import annotations

import os
import re
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from config import APP_VERSION, settings
from data_pipeline import market_circuit_breaker

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
        "name": "StockLSTM API",
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
    """Readiness checks market data and any configured learned-serving contract."""
    market_ready, upstream = market_circuit_breaker.is_ready()
    is_ready = market_ready
    dependencies = {
        "market_data": upstream,
        "model_storage": {
            "required": False,
            "writable": None,
            "detail": "Browser-trained models are trained and cached in each user's browser.",
        },
        "global_volatility": {
            "configured": bool(
                settings.volatility_release_dir and settings.volatility_public_key_path
            ),
            "required": settings.volatility_serving_required,
            "status": "not_required",
        },
    }

    if settings.volatility_serving_required or (
        settings.volatility_release_dir and settings.volatility_public_key_path
    ):
        from routes.volatility_v2 import volatility_release_readiness

        volatility_ready = volatility_release_readiness()
        dependencies["global_volatility"] = {
            **volatility_ready,
            "required": settings.volatility_serving_required,
        }
        if settings.volatility_serving_required:
            is_ready = is_ready and volatility_ready["status"] == "ready"

    if settings.server_forecast_serving_enabled:
        from server_models.api import server_forecast_readiness

        readiness = server_forecast_readiness()
        server_ready = readiness.configured
        dependencies["server_forecasts"] = {
            "configured": readiness.configured,
            "status": "ready" if readiness.configured else readiness.reason,
            "required": settings.training_mode == "server_pretrained",
            "bundle_retention_days": settings.server_bundle_retention_days,
        }
        if settings.training_mode == "server_pretrained":
            dependencies["model_storage"] = {
                "required": True,
                "writable": readiness.configured,
                "detail": "Server artifacts are stored in the registry and object store.",
            }
            is_ready = is_ready and server_ready
    content = {
        "status": "ready" if is_ready else "degraded",
        "version": APP_VERSION,
        "deployment": deployment_identity(),
        "dependencies": dependencies,
    }
    return JSONResponse(status_code=200 if is_ready else 503, content=content)
