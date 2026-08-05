import threading
from datetime import UTC, datetime
from typing import Literal

from cachetools import TTLCache
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from fastapi import APIRouter, Depends, Query, Response
from pydantic import BaseModel

from artifacts.signing import Ed25519ManifestVerifier
from config import settings
from server_models.compatibility import check_record_compatibility, is_fresh
from server_models.contracts import ServerForecastBundle
from server_models.db import PostgresRegistry
from server_models.signing_manifests import verify_bundle
from server_models.storage import S3ObjectStore

router = APIRouter(prefix="/api/v1/server-forecasts", tags=["Server Forecasts"])

_availability_cache = TTLCache(maxsize=1, ttl=300)
_availability_lock = threading.Lock()
_bundle_cache = TTLCache(maxsize=128, ttl=900)
_bundle_lock = threading.Lock()


def get_registry() -> PostgresRegistry:
    return PostgresRegistry(database_url=settings.registry_database_url)


def get_storage() -> S3ObjectStore:
    return S3ObjectStore(
        bucket=settings.s3_bucket or "fallback-bucket",
        prefix=settings.s3_key_prefix,
        endpoint_url=settings.s3_endpoint_url,
    )


def get_verifier() -> Ed25519ManifestVerifier | None:
    # Phase 1: verify signature if public key is configured
    # Will be improved in Phase 2 with proper KMS
    import os

    from cryptography.hazmat.primitives import serialization

    key_path = os.environ.get("SERVER_FORECAST_PUBLIC_KEY_PATH")
    if not key_path or not os.path.exists(key_path):
        return None
    try:
        with open(key_path, "rb") as f:
            key = serialization.load_pem_public_key(f.read())
            if isinstance(key, Ed25519PublicKey):
                return Ed25519ManifestVerifier(key)
    except Exception:
        pass
    return None


class TickerAvailability(BaseModel):
    ticker: str
    status: Literal["fresh", "stale", "missing"]
    version_id: str | None = None
    trained_at: datetime | None = None
    age_hours: float | None = None
    expires_at: datetime | None = None


class AvailabilityResponse(BaseModel):
    mode: str
    allowlist: list[str]
    tickers: list[TickerAvailability]


@router.get("/availability", response_model=AvailabilityResponse)
def get_availability(response: Response, registry: PostgresRegistry = Depends(get_registry)):
    response.headers["Cache-Control"] = "public, max-age=300"

    with _availability_lock:
        cached = _availability_cache.get("availability")
        if cached is not None:
            return cached

    tickers = []
    for ticker in settings.server_forecast_allowlist:
        try:
            promoted = registry.get_promoted(ticker)
        except Exception:
            promoted = None

        if promoted is None:
            tickers.append(TickerAvailability(ticker=ticker, status="missing"))
            continue

        trained_at = promoted.key.trained_at
        age_delta = datetime.now(UTC) - trained_at
        age_hours = age_delta.total_seconds() / 3600.0

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
        mode=settings.training_mode, allowlist=settings.server_forecast_allowlist, tickers=tickers
    )

    with _availability_lock:
        _availability_cache["availability"] = result

    return result


class FallbackResponse(BaseModel):
    available: Literal[False]
    reason: Literal["missing", "stale", "disabled", "incompatible"]
    fallback: Literal["browser_training"]


@router.get("/{ticker}")
def get_forecast(
    ticker: str,
    response: Response,
    forecast_type: str = Query("price"),
    days: int = Query(7, ge=1, le=30),
    registry: PostgresRegistry = Depends(get_registry),
    storage: S3ObjectStore = Depends(get_storage),
    verifier: Ed25519ManifestVerifier | None = Depends(get_verifier),
):
    ticker = ticker.upper()

    def fallback(
        reason: Literal["missing", "stale", "disabled", "incompatible"],
    ) -> FallbackResponse:
        response.headers["Cache-Control"] = "no-store"
        return FallbackResponse(available=False, reason=reason, fallback="browser_training")

    if (
        not settings.server_forecast_serving_enabled
        or ticker not in settings.server_forecast_allowlist
    ):
        return fallback("disabled")

    try:
        promoted = registry.get_promoted(ticker, forecast_type=forecast_type)
    except Exception:
        promoted = None

    if promoted is None:
        return fallback("missing")

    if not is_fresh(promoted.key.trained_at, settings.server_forecast_max_age_hours):
        return fallback("stale")

    compat = check_record_compatibility(promoted)
    if not compat.compatible:
        return fallback("incompatible")

    version_id = promoted.key.version_id
    cache_key = f"{version_id}:{days}"

    with _bundle_lock:
        cached = _bundle_cache.get(cache_key)
        if cached is not None:
            response.headers["ETag"] = version_id
            response.headers["Cache-Control"] = "public, max-age=900"
            return cached

    try:
        bundle_bytes = storage.get_bundle(version_id)
        if verifier is not None:
            # Reconstruct the manifest from the record
            manifest = {
                "schema_version": 1,
                "signature_algorithm": "ed25519",
                "digest_algorithm": "sha256",
                "sha256": promoted.sha256_digest,
                "signature": promoted.signature,
            }
            verify_bundle(bundle_bytes, manifest, verifier)

        bundle = ServerForecastBundle.model_validate_json(bundle_bytes)
    except Exception:
        return fallback("missing")

    # Slice the output vectors up to `days`
    bundle_dict = bundle.model_dump()
    bundle_dict["predicted_prices"] = bundle_dict["predicted_prices"][:days]
    bundle_dict["predicted_log_returns"] = bundle_dict["predicted_log_returns"][:days]
    bundle_dict["future_dates"] = bundle_dict["future_dates"][:days]

    # Map bundle structure to the frontend format exactly
    result = {
        "ticker": bundle.ticker,
        "forecast_days": days,
        "future_dates": [d.isoformat() for d in bundle_dict["future_dates"]],
        "metrics": bundle.evidence,
        "metadata": {
            "engine": {
                "role": "server_pretrained",
                "family": "elastic_net",
                "version_id": bundle.version_id,
            },
            "metric_source": "server_purged_walk_forward",
            "browser_training": False,
            "trained_at": bundle.generated_at.isoformat(),
        },
    }
    # For price forecasts, add prices array
    if forecast_type == "price":
        result["predicted_prices"] = bundle_dict["predicted_prices"]
    else:
        # TODO: Handle direction format mapping if needed
        pass

    with _bundle_lock:
        _bundle_cache[cache_key] = result

    response.headers["ETag"] = version_id
    response.headers["Cache-Control"] = "public, max-age=900"

    return result
