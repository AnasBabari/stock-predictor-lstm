"""Serving layer for server-pretrained forecast bundles (price type).

Readiness policy (explicit 200-vs-503 by mode):

* Expected absence — serving disabled, ticker not allowlisted, no promoted
  artifact (``missing``), artifact too old (``stale``), contract mismatch
  (``incompatible``), or a non-price request (``unsupported_forecast_type``) —
  returns a 200 fallback so the frontend always renders its browser path,
  except in ``server_pretrained`` mode where a missing/stale/incompatible
  artifact is itself an infrastructure failure and returns 503.
* Infrastructure failure — registry unreachable, bundle unreadable, bundle
  digest/signature verification failing — always returns 503 (fail closed),
  never a fallback, and never a 500.
"""

import hashlib
import hmac
import logging
import os
import threading
from datetime import UTC, datetime
from typing import Literal

from cachetools import TTLCache
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel

from artifacts.signing import Ed25519ManifestVerifier
from config import settings
from server_models.compatibility import check_record_compatibility, is_fresh
from server_models.contracts import ServerForecastBundle
from server_models.db import PostgresRegistry
from server_models.signing_manifests import verify_bundle
from server_models.storage import ObjectStoreError, S3ObjectStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/server-forecasts", tags=["Server Forecasts"])

_availability_cache = TTLCache(maxsize=1, ttl=300)
_availability_lock = threading.Lock()
_bundle_cache = TTLCache(maxsize=128, ttl=900)
_bundle_lock = threading.Lock()

SUPPORTED_FORECAST_TYPES = ("price",)


def get_registry() -> PostgresRegistry:
    return PostgresRegistry(database_url=settings.registry_database_url)


def get_storage() -> S3ObjectStore:
    return S3ObjectStore(
        bucket=settings.s3_bucket or "fallback-bucket",
        prefix=settings.s3_key_prefix,
        endpoint_url=settings.s3_endpoint_url,
    )


def get_verifier() -> Ed25519ManifestVerifier | None:
    """Load the Ed25519 verifier from the configured public key path.

    Returns ``None`` (sha256-only verification) when no key is configured; an
    unreadable key file is logged as a warning so operators notice.
    """
    key_path = settings.server_forecast_public_key_path
    if not key_path or not os.path.exists(key_path):
        return None
    try:
        with open(key_path, "rb") as handle:
            key = serialization.load_pem_public_key(handle.read())
        if isinstance(key, Ed25519PublicKey):
            return Ed25519ManifestVerifier(key)
        logger.warning("Configured public key is not an Ed25519 key; falling back to sha256.")
    except Exception as exc:
        logger.warning("Public key could not be loaded (%s); falling back to sha256.", exc)
    return None


def _artifact_absence_is_infrastructure_failure() -> bool:
    """In server_pretrained mode a missing/stale/incompatible artifact is fatal."""
    return bool(
        settings.server_forecast_serving_enabled and settings.training_mode == "server_pretrained"
    )


class TickerAvailability(BaseModel):
    ticker: str
    status: Literal["fresh", "stale", "missing"]
    version_id: str | None = None
    trained_at: datetime | None = None
    age_hours: float | None = None
    expires_at: datetime | None = None


class AvailabilityResponse(BaseModel):
    enabled: bool
    mode: str
    allowlist: list[str]
    tickers: list[TickerAvailability]


class FallbackResponse(BaseModel):
    available: Literal[False]
    reason: Literal["missing", "stale", "disabled", "incompatible", "unsupported_forecast_type"]
    fallback: Literal["browser_training"]


def _fallback(response: Response, reason: str) -> FallbackResponse:
    response.status_code = 200
    response.headers["Cache-Control"] = "no-store"
    return FallbackResponse(available=False, reason=reason, fallback="browser_training")


@router.get("/availability", response_model=AvailabilityResponse)
def get_availability(response: Response) -> AvailabilityResponse:
    response.headers["Cache-Control"] = "public, max-age=300"

    if not settings.server_forecast_serving_enabled:
        return AvailabilityResponse(
            enabled=False,
            mode="browser_only",
            allowlist=settings.server_forecast_allowlist,
            tickers=[],
        )

    with _availability_lock:
        cached = _availability_cache.get("availability")
        if cached is not None:
            return cached

    tickers = []
    for ticker in settings.server_forecast_allowlist:
        try:
            promoted = get_registry().get_promoted(ticker)
        except Exception:
            promoted = None

        if promoted is None:
            tickers.append(TickerAvailability(ticker=ticker, status="missing"))
            continue

        trained_at = promoted.key.trained_at
        age_hours = (datetime.now(UTC) - trained_at).total_seconds() / 3600.0
        status: Literal["fresh", "stale"] = (
            "fresh" if is_fresh(trained_at, settings.server_forecast_max_age_hours) else "stale"
        )
        tickers.append(
            TickerAvailability(
                ticker=ticker,
                status=status,
                version_id=promoted.key.version_id,
                trained_at=trained_at,
                age_hours=round(age_hours, 2),
                expires_at=None,
            )
        )

    result = AvailabilityResponse(
        enabled=True,
        mode=settings.training_mode,
        allowlist=settings.server_forecast_allowlist,
        tickers=tickers,
    )

    with _availability_lock:
        _availability_cache["availability"] = result

    return result


@router.get("/{ticker}")
def get_forecast(
    ticker: str,
    response: Response,
    forecast_type: str = Query("price"),
    days: int = Query(7, ge=1, le=30),
):
    ticker = ticker.upper()

    if (
        not settings.server_forecast_serving_enabled
        or ticker not in settings.server_forecast_allowlist
    ):
        return _fallback(response, "disabled")

    # Only price forecasts are produced today. Direction/trend requests
    # deliberately get a 200 fallback so the frontend renders its browser
    # trend path; there is deliberately no price->probability conversion.
    if forecast_type not in SUPPORTED_FORECAST_TYPES:
        return _fallback(response, "unsupported_forecast_type")

    try:
        promoted = get_registry().get_promoted(ticker, forecast_type="price")
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"Artifact registry unavailable: {exc}"
        ) from exc

    if promoted is None or not is_fresh(
        promoted.key.trained_at, settings.server_forecast_max_age_hours
    ):
        reason = "missing" if promoted is None else "stale"
        if _artifact_absence_is_infrastructure_failure():
            raise HTTPException(
                status_code=503,
                detail=f"Server-pretrained artifact for {ticker} is {reason}.",
            )
        return _fallback(response, reason)

    compat = check_record_compatibility(promoted)
    if not compat.compatible:
        if _artifact_absence_is_infrastructure_failure():
            raise HTTPException(
                status_code=503,
                detail=f"Server-pretrained artifact for {ticker} is incompatible: {compat.reason}",
            )
        return _fallback(response, "incompatible")

    version_id = promoted.key.version_id
    cache_key = f"{version_id}:{days}"

    with _bundle_lock:
        cached = _bundle_cache.get(cache_key)
        if cached is not None:
            response.headers["ETag"] = version_id
            response.headers["Cache-Control"] = "public, max-age=900"
            return cached

    try:
        bundle_bytes = get_storage().get_bundle(version_id)
    except ObjectStoreError as exc:
        raise HTTPException(status_code=503, detail=f"Artifact bundle unavailable: {exc}") from exc

    # Verification is fail-closed: a tampered or mismatched bundle is an
    # infrastructure/security failure (503) in every mode, never a fallback.
    verifier = get_verifier()
    if verifier is not None:
        manifest = {
            "schema_version": 1,
            "signature_algorithm": "ed25519",
            "digest_algorithm": "sha256",
            "sha256": promoted.sha256_digest,
            "signature": promoted.signature,
        }
        try:
            verify_bundle(bundle_bytes, manifest, verifier)
            authenticity = "ed25519_verified"
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail="Artifact signature verification failed."
            ) from exc
    else:
        digest = hashlib.sha256(bundle_bytes).hexdigest()
        if not hmac.compare_digest(digest, promoted.sha256_digest):
            raise HTTPException(status_code=503, detail="Artifact bundle digest mismatch.")
        authenticity = "sha256_only"

    try:
        bundle = ServerForecastBundle.model_validate_json(bundle_bytes)
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail="Artifact bundle failed contract validation."
        ) from exc

    result = {
        "available": True,
        "ticker": bundle.ticker,
        "forecast_days": days,
        "future_dates": [d.isoformat() for d in bundle.future_dates[:days]],
        "predicted_prices": bundle.predicted_prices[:days],
        "historical_dates": [d.isoformat() for d in bundle.historical_dates],
        "historical_prices": bundle.historical_prices,
        "metrics": bundle.evidence,
        "metadata": {
            "engine": {
                "role": "server_pretrained",
                "family": bundle.evidence.get("family", "unknown"),
                "version_id": bundle.version_id,
            },
            "metric_source": "server_purged_walk_forward",
            "browser_training": False,
            "trained_at": bundle.generated_at.isoformat(),
            "origin": {
                "date": bundle.origin_date.isoformat(),
                "close": bundle.origin_close,
            },
            "authenticity": authenticity,
            "evidence": bundle.evidence,
        },
    }

    with _bundle_lock:
        _bundle_cache[cache_key] = result

    response.headers["ETag"] = version_id
    response.headers["Cache-Control"] = "public, max-age=900"

    return result
