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
    release_configured = bool(
        settings.volatility_release_dir and settings.volatility_public_key_path
    )
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
            "status": "configured" if release_configured else "unconfigured",
            "reason": (
                "Signed global volatility release is configured."
                if release_configured
                else "No signed global volatility release is mounted; requests abstain."
            ),
            "model_family": "baseline_residual_tcn_ensemble",
            "endpoint": "/api/v2/forecast",
            "horizons": [1, 3, 5, 7, 14, 30],
            "certified_heads": {
                "volatility": True,
                "return_distribution": False,
                "direction": False,
            },
            "metric_source": "locked_purged_walk_forward",
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
                "status": "volatility_location_baseline",
                "engine": "baseline_residual_tcn_ensemble",
                "tickers": "global",
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
    """Disclose the browser-training contract and any optional offline evidence."""
    ticker = validate_ticker(ticker)
    return {
        "ticker": ticker,
        "forecast_type": forecast_type,
        "engine": {
            "family": "compact_tfjs_lstm",
            "role": "browser_training_available",
            "baseline_fallback": False,
            "artifact_version": None,
        },
        "metrics": {
            "metric_source": "browser_purged_holdout",
            "detail": "Metrics are produced locally in the browser after training.",
        },
        "benchmark": {
            "snapshot": None,
            "validation_method": "browser_purged_holdout",
            "validation_folds": None,
            "metric_source": "browser_purged_holdout",
        },
    }
