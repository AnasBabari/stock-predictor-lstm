"""Serving layer for server-pretrained forecast bundles (price type).

Readiness policy (explicit 200-vs-503 by mode):

* Soft absence — serving disabled (``disabled``), ticker not allowlisted
  (``not_allowlisted``), or a non-price request (``unsupported_forecast_type``)
  — always returns a 200 fallback so the frontend renders its browser path.
* Expected absence — no promoted artifact (``missing``), artifact too old
  (``stale``), contract mismatch (``incompatible``), or missing serving
  configuration (``unconfigured``) — returns a 200 fallback in the browser
  training modes, but in ``server_pretrained`` mode these are infrastructure
  failures and return 503 (``fallback: null``).
* Hard failure — registry unreachable, bundle unreadable, digest/signature
  verification failure, contract violation, identity mismatch, or an
  unreadable/invalid public key (``integrity_failure``) — always returns 503
  (fail closed), never a fallback. In the browser training modes the 503 body
  still carries ``fallback: "browser_training"`` so the frontend may degrade;
  in ``server_pretrained`` mode it carries ``fallback: null`` and the frontend
  must surface the error instead of silently training in the browser.

Signing policy: configured server serving requires an Ed25519 public key. There
is deliberately no digest-only acceptance mode — a missing key is
``unconfigured`` and a broken key is ``integrity_failure`` — so a served bundle
always carries ``authenticity: "ed25519_verified"``.
"""

import logging
import os
import threading
import time
from datetime import UTC, datetime
from typing import Any, Literal

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
from server_models.response_models import (
    ForecastMetadata,
    PredictionExecution,
    PredictionTimings,
    PriceForecastResponse,
)
from server_models.signing_manifests import verify_bundle
from server_models.storage import ObjectStoreError, S3ObjectStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/server-forecasts", tags=["Server Forecasts"])

_availability_cache = TTLCache(maxsize=1, ttl=300)
_availability_lock = threading.Lock()
_bundle_cache = TTLCache(maxsize=128, ttl=settings.server_forecast_cache_ttl)
_bundle_lock = threading.Lock()

SUPPORTED_FORECAST_TYPES = ("price",)

MESSAGES = {
    "disabled": "Server forecast serving is disabled.",
    "not_allowlisted": "The requested ticker is not on the server forecast allowlist.",
    "unconfigured": "Server forecast serving is not fully configured.",
    "integrity_failure": "Server forecast signing configuration is invalid.",
    "missing": "No server forecast artifact is available for this ticker.",
    "stale": "The server forecast artifact is too old to serve.",
    "incompatible": "The server forecast artifact is incompatible with the snapshot contract.",
    "unsupported_forecast_type": "Only price forecasts are produced by the server.",
    "registry_unavailable": "Server forecast infrastructure is unavailable.",
    "bundle_unavailable": "Server forecast infrastructure is unavailable.",
    "signature_verification_failed": "Server forecast bundle verification failed.",
    "contract_validation_failed": "Server forecast bundle failed contract validation.",
    "identity_mismatch": "Server forecast bundle identity does not match its registry record.",
}

# Reasons that are always a 200 browser fallback, even in server_pretrained mode.
_SOFT_ABSENCES = {"disabled", "not_allowlisted", "unsupported_forecast_type"}
# Expected absences that become 503s in server_pretrained mode.
_HARD_ABSENCES = {"unconfigured", "missing", "stale", "incompatible"}


def get_registry() -> PostgresRegistry:
    return PostgresRegistry(database_url=settings.registry_database_url)


def get_storage() -> S3ObjectStore:
    return S3ObjectStore(
        bucket=settings.s3_bucket or "fallback-bucket",
        prefix=settings.s3_key_prefix,
        endpoint_url=settings.s3_endpoint_url,
    )


def load_verifier() -> tuple[Ed25519ManifestVerifier | None, str | None]:
    """Load the Ed25519 verifier from the configured public key path.

    Returns ``(None, "unconfigured")`` when no key is configured,
    ``(None, "integrity_failure")`` when a configured key is missing or cannot
    be loaded, and ``(verifier, None)`` when verification is ready.
    """
    key_path = settings.server_forecast_public_key_path
    if not key_path:
        return None, "unconfigured"
    if not os.path.exists(key_path):
        logger.error("Server forecast public key path configured but missing: %s", key_path)
        return None, "integrity_failure"
    try:
        with open(key_path, "rb") as handle:
            key = serialization.load_pem_public_key(handle.read())
    except Exception:
        logger.exception("Server forecast public key could not be loaded: %s", key_path)
        return None, "integrity_failure"
    if not isinstance(key, Ed25519PublicKey):
        logger.error("Configured server forecast public key is not an Ed25519 key.")
        return None, "integrity_failure"
    return Ed25519ManifestVerifier(key), None


def _server_pretrained() -> bool:
    return settings.training_mode == "server_pretrained"


class ServerForecastReadiness(BaseModel):
    """Outcome of checking whether serving can run for a ticker.

    ``configured`` means infrastructure (Postgres, S3) and the Ed25519 public
    key are all present and usable. ``reason`` describes the first unmet
    requirement; ``verifier`` is populated when configured.
    """

    model_config = {"arbitrary_types_allowed": True}

    configured: bool
    reason: str | None = None
    fallback_allowed: bool = True
    verifier: Any | None = None


def server_forecast_readiness(ticker: str | None = None) -> ServerForecastReadiness:
    """Decide whether serving can run at all, and how the frontend may react."""
    if not settings.server_forecast_serving_enabled:
        return ServerForecastReadiness(configured=False, reason="disabled", fallback_allowed=True)
    if ticker is not None and ticker.upper() not in settings.server_forecast_allowlist:
        return ServerForecastReadiness(
            configured=False, reason="not_allowlisted", fallback_allowed=True
        )
    if not settings.registry_database_url or not settings.s3_bucket:
        return ServerForecastReadiness(
            configured=False,
            reason="unconfigured",
            fallback_allowed=not _server_pretrained(),
        )
    verifier, reason = load_verifier()
    if reason is not None:
        return ServerForecastReadiness(
            configured=False, reason=reason, fallback_allowed=reason != "integrity_failure"
        )
    return ServerForecastReadiness(
        configured=True,
        reason=None,
        fallback_allowed=not _server_pretrained(),
        verifier=verifier,
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
    configured: bool
    reason: str | None = None
    allowlist: list[str]
    tickers: list[TickerAvailability]


class FallbackResponse(BaseModel):
    available: Literal[False] = False
    reason: str
    fallback: Literal["browser_training"] | None = None
    code: str | None = None


def _fallback_response(response: Response, reason: str) -> FallbackResponse:
    response.status_code = 200
    response.headers["Cache-Control"] = "no-store"
    return FallbackResponse(available=False, reason=reason, fallback="browser_training")


def _absence_response(response: Response, reason: str) -> FallbackResponse:
    """Map an absence reason to a 200 fallback or a mode-aware 503."""
    if reason == "integrity_failure" or (_server_pretrained() and reason in _HARD_ABSENCES):
        raise HTTPException(
            status_code=503,
            detail={
                "available": False,
                "code": reason,
                "message": MESSAGES[reason],
                "fallback": None,
            },
        )
    return _fallback_response(response, reason)


def _raise_infrastructure_error(code: str, exc: Exception | None, *args: Any) -> None:
    logger.exception(
        "%s", MESSAGES[code], exc_info=(type(exc), exc, exc.__traceback__) if exc else None
    )
    fallback = None if _server_pretrained() else "browser_training"
    raise HTTPException(
        status_code=503,
        detail={"available": False, "code": code, "message": MESSAGES[code], "fallback": fallback},
    )


@router.get("/availability", response_model=AvailabilityResponse)
def get_availability(response: Response) -> AvailabilityResponse:
    response.headers["Cache-Control"] = "public, max-age=300"

    readiness = server_forecast_readiness()
    if not settings.server_forecast_serving_enabled:
        return AvailabilityResponse(
            enabled=False,
            mode="browser_only",
            configured=False,
            reason="disabled",
            allowlist=settings.server_forecast_allowlist,
            tickers=[],
        )
    if not readiness.configured:
        return AvailabilityResponse(
            enabled=True,
            mode=settings.training_mode,
            configured=False,
            reason=readiness.reason,
            allowlist=settings.server_forecast_allowlist,
            tickers=[],
        )

    with _availability_lock:
        cached = _availability_cache.get("availability")
        if cached is not None:
            return cached

    # One registry instance for the whole allowlist, closed once. Per-ticker
    # connections leak and can exhaust the Postgres pool under repeated probes.
    registry = get_registry()
    try:
        tickers = []
        for ticker in settings.server_forecast_allowlist:
            try:
                promoted = registry.get_promoted(ticker)
            except Exception as exc:
                _raise_infrastructure_error("registry_unavailable", exc)
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
    finally:
        registry.close()

    result = AvailabilityResponse(
        enabled=True,
        mode=settings.training_mode,
        configured=True,
        reason=None,
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

    readiness = server_forecast_readiness(ticker)
    if not readiness.configured:
        return _absence_response(response, readiness.reason or "disabled")

    # Only price forecasts are produced today. Direction/trend requests
    # deliberately get a 200 fallback so the frontend renders its browser
    # trend path; there is deliberately no price->probability conversion.
    if forecast_type not in SUPPORTED_FORECAST_TYPES:
        return _fallback_response(response, "unsupported_forecast_type")

    registry = get_registry()
    try:
        try:
            promoted = registry.get_promoted(ticker, forecast_type="price")
        except Exception as exc:
            _raise_infrastructure_error("registry_unavailable", exc)
    finally:
        registry.close()

    if promoted is None:
        return _absence_response(response, "missing")
    if not is_fresh(promoted.key.trained_at, settings.server_forecast_max_age_hours):
        return _absence_response(response, "stale")

    compat = check_record_compatibility(promoted)
    if not compat.compatible:
        return _absence_response(response, "incompatible")

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
        _raise_infrastructure_error("bundle_unavailable", exc)

    # Verification is fail-closed: a tampered or mismatched bundle is an
    # infrastructure/security failure (503) in every mode, never a fallback.
    verifier = readiness.verifier
    manifest = {
        "schema_version": 1,
        "signature_algorithm": "ed25519",
        "digest_algorithm": "sha256",
        "sha256": promoted.sha256_digest,
        "signature": promoted.signature,
    }
    started = time.perf_counter()
    try:
        verify_bundle(bundle_bytes, manifest, verifier)
    except Exception as exc:
        _raise_infrastructure_error("signature_verification_failed", exc)
    verification_seconds = time.perf_counter() - started

    try:
        bundle = ServerForecastBundle.model_validate_json(bundle_bytes)
    except Exception as exc:
        _raise_infrastructure_error("contract_validation_failed", exc)

    # Identity cross-check: the signed bytes must agree with the registry row
    # so a correctly signed but wrong bundle can never be served for another
    # ticker or version.
    if (
        bundle.ticker != promoted.key.ticker
        or bundle.ticker != ticker
        or bundle.version_id != version_id
        or bundle.forecast_type != "price"
        or bundle.forecast_type != promoted.key.forecast_type
    ):
        _raise_infrastructure_error("identity_mismatch", RuntimeError("bundle identity mismatch"))

    result = PriceForecastResponse(
        available=True,
        ticker=bundle.ticker,
        forecast_days=days,
        future_dates=[d.isoformat() for d in bundle.future_dates[:days]],
        predicted_prices=bundle.predicted_prices[:days],
        historical_dates=[d.isoformat() for d in bundle.historical_dates],
        historical_prices=bundle.historical_prices,
        metrics=bundle.evidence,
        metadata=ForecastMetadata(
            timings_seconds=PredictionTimings(
                queue_wait=None,
                market_data=None,
                feature_preparation=None,
                artifact_load_validation=round(verification_seconds, 6),
                training=None,
                inference=None,
                total=round(verification_seconds, 6),
            ),
            execution=PredictionExecution(mode="artifact_loaded", coalesced=False),
            artifact_state_before="fresh",
            artifact_action="loaded",
            engine={
                "role": "server_pretrained",
                "family": bundle.evidence.get("family", "unknown"),
                "version_id": bundle.version_id,
            },
            metric_source="server_purged_walk_forward",
            browser_training=False,
            trained_at=bundle.generated_at.isoformat(),
            origin={
                "date": bundle.origin_date.isoformat(),
                "close": bundle.origin_close,
            },
            authenticity="ed25519_verified",
            evidence=bundle.evidence,
        ),
    )

    with _bundle_lock:
        _bundle_cache[cache_key] = result

    response.headers["ETag"] = version_id
    response.headers["Cache-Control"] = "public, max-age=900"

    return result
