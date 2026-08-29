"""Model discovery, performance disclosure, and legacy diagnostics routes."""

from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from config import APP_VERSION, settings
from routes.common import limiter, validate_model_type, validate_ticker

logger = logging.getLogger(__name__)
router = APIRouter(tags=["models"])


def load_metadata(*_args, **_kwargs) -> dict:
    return {}


def load_metrics(*_args, **_kwargs) -> dict:
    return {}


def load_cross_validation(*_args, **_kwargs) -> dict:
    return {}


def load_validation_results(*_args, **_kwargs) -> list:
    return []


class ModelPerformanceResponse(BaseModel):
    ticker: str
    forecast_type: Literal["price", "direction"]
    engine: dict
    metrics: dict
    benchmark: dict


@router.get("/models")
def list_models():
    """Describe the single signed global-volatility serving contract.

    The legacy per-ticker/browser entries remain as compatibility diagnostics
    during rollout, but are explicitly disabled so clients cannot mistake them
    for the production model.
    """
    from routes.volatility_v2 import volatility_release_readiness

    release = volatility_release_readiness()
    release_configured = release["status"] == "ready"
    server_models = {
        "status": "disabled",
        "reason": "Legacy per-ticker server models are disabled; use the global volatility contract.",
        "training_mode": settings.training_mode,
    }
    return {
        "version": APP_VERSION,
        "manifest": {},
        "server_models": server_models,
        "global_volatility": {
            "status": release["status"],
            "reason": (
                "Signed global volatility release is verified and ready."
                if release_configured
                else "No verified signed global volatility release is ready; requests abstain."
            ),
            "model_id": release.get("model_id"),
            "model_version": release.get("model_version"),
            "certified_horizons": release.get("certified_horizons", []),
            "model_family": release.get("model_family") if release_configured else None,
            "endpoint": "/api/v2/forecast",
            "horizons": [1, 3, 5, 7, 14, 30],
            "certified_heads": (
                release.get("certified_heads")
                if release_configured
                else {
                    "volatility": False,
                    "return_distribution": False,
                    "direction": False,
                }
            ),
            "metric_source": release.get("metric_source") if release_configured else None,
            "certification_scope": release.get("certification_scope")
            if release_configured
            else None,
            "news_status": release.get("news_status", "not_certified"),
            "training": "offline RTX workstation",
        },
        "browser_training": {
            "status": "disabled",
            "reason": "Production training and inference run from the signed global volatility release.",
            "model_family": None,
            "storage": None,
        },
        "model_storage": {
            "location": "signed_release" if release_configured else "none",
            "required": True,
            "detail": "The API loads only a checksum- and signature-verified global volatility bundle.",
        },
        "availability": {
            "price": {
                "status": "ready" if release_configured else "unconfigured_abstaining",
                "engine": release.get("model_family") if release_configured else None,
                "tickers": release.get("supported_tickers", []) if release_configured else [],
            },
            "direction": {
                "status": "not_certified",
                "engine": None,
                "tickers": [],
            },
        },
    }


@router.get("/api/v1/diagnostics/{ticker}")
@limiter.limit("10/minute")
async def diagnostics(
    request: Request,
    ticker: str,
    model_type: str = Query(default="bilstm_attention_direction"),
):
    """Return walk-forward validation diagnostics for a trained ticker model."""
    ticker = validate_ticker(ticker)
    model_type = validate_model_type(model_type)

    try:
        cv_summary = load_cross_validation(ticker, model_type)
        fold_results = load_validation_results(ticker, model_type)
        metadata = load_metadata(ticker, model_type)

        if not cv_summary and not fold_results:
            raise HTTPException(
                status_code=404,
                detail=f"No diagnostics found for {ticker}/{model_type}. Train the model first.",
            )

        return {
            "ticker": ticker,
            "model_type": model_type,
            "cross_validation": cv_summary,
            "fold_results": fold_results,
            "model_metadata": {
                "schema_version": metadata.get("schema_version"),
                "feature_count": metadata.get("feature_count"),
                "dataset_fingerprint": metadata.get("dataset_fingerprint"),
                "training_duration_seconds": metadata.get("training_duration_seconds"),
                "created_at": metadata.get("created_at"),
                "validation_method": metadata.get("validation_method"),
                "validation_folds": metadata.get("validation_folds"),
            },
        }

    except HTTPException:
        raise
    except Exception as err:
        logger.exception("Error fetching diagnostics for %s", ticker)
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch diagnostics. Please try again later.",
        ) from err


@router.get(
    "/api/v1/model-performance/{ticker}",
    response_model=ModelPerformanceResponse,
)
@limiter.limit("20/minute")
def model_performance(
    request: Request,
    ticker: str,
    forecast_type: Literal["price", "direction"] = Query(default="price"),
):
    """Disclose the certified global-volatility contract, not browser training."""
    ticker = validate_ticker(ticker)
    from routes.volatility_v2 import volatility_release_readiness

    release = volatility_release_readiness()
    release_configured = release["status"] == "ready"
    is_volatility = forecast_type == "price"
    return {
        "ticker": ticker,
        "forecast_type": forecast_type,
        "engine": {
            "family": release.get("model_family")
            if (release_configured and is_volatility)
            else None,
            "role": "global_volatility"
            if (release_configured and is_volatility)
            else "not_certified",
            "baseline_fallback": False,
            "artifact_version": release.get("model_id") if release_configured else None,
            "status": release["status"] if is_volatility else "not_certified",
            "certified_head": "volatility" if (release_configured and is_volatility) else None,
        },
        "metrics": {
            "metric_source": release.get("metric_source")
            if (release_configured and is_volatility)
            else "not_certified",
            "detail": (
                "QLIKE and coverage describe the locked offline volatility evaluation."
                if (release_configured and is_volatility)
                else "No certified volatility model is loaded; requests abstain."
                if is_volatility
                else "No direction model is certified for production."
            ),
        },
        "benchmark": {
            "snapshot": None,
            "validation_method": "purged_walk_forward"
            if (release_configured and is_volatility)
            else None,
            "validation_folds": 5 if (release_configured and is_volatility) else None,
            "metric_source": release.get("metric_source")
            if (release_configured and is_volatility)
            else "not_certified",
            "certified_horizons": release.get("certified_horizons", [])
            if (release_configured and is_volatility)
            else [],
        },
    }
