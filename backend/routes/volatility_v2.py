"""Certified global-volatility forecast endpoint (v2, fail-closed).

One frozen ensemble serves every supported ticker. Legacy releases may certify
conditional volatility only, in which case price ranges are reconstructed as
zero-location cones around the current close. A V11.2 release may additionally
certify a Student-t return distribution and expose its terminal learned
location. Without a signed, verified release the endpoint answers with an
explicit no-certified-model abstention instead of any baseline path.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel
from scipy.stats import t as student_t

from config import settings
from routes.common import limiter, validate_ticker
from services.volatility_release_bootstrap import (
    ReleaseBootstrapError,
    release_source_configured,
    resolve_release_dir,
)
from services.volatility_runtime.contracts import VOLATILITY_HORIZONS

logger = logging.getLogger(__name__)

router = APIRouter(tags=["volatility-v2"])

# Standard-normal quantiles for the disclosed price bands.
_QUANTILE_Z_SCORES: dict[str, float] = {
    "p05": -1.6448536269514722,
    "p10": -1.2815515655446004,
    "p25": -0.6744897501960817,
    "p50": 0.0,
    "p75": 0.6744897501960817,
    "p90": 1.2815515655446004,
    "p95": 1.6448536269514722,
}

TRADING_SESSIONS_PER_YEAR = 252

LEGACY_CERTIFIED_HEADS = {
    "volatility": True,
    "return_distribution": False,
    "direction": False,
}
_QUANTILE_PROBABILITIES: dict[str, float] = {
    "p05": 0.05,
    "p10": 0.10,
    "p25": 0.25,
    "p50": 0.50,
    "p75": 0.75,
    "p90": 0.90,
    "p95": 0.95,
}


class VolatilityForecastResponse(BaseModel):
    """v2 response contract: distribution evidence, never a point promise."""

    ticker: str
    as_of: str
    horizon: int
    current_price: float
    historical_dates: list[str]
    historical_prices: list[float]
    forecast: dict[str, Any]
    evidence: dict[str, Any]


class _ReleaseState:
    """Process-wide verified runtime; failures stay uncached and re-checked."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runtime: Any = None

    def get(self) -> tuple[Any, str | None]:
        with self._lock:
            if self._runtime is not None:
                return self._runtime, None
            public_key_path = settings.volatility_public_key_path
            if not release_source_configured(settings) or not public_key_path:
                return None, "no certified volatility release is configured"
            try:
                from services.volatility_runtime import VolatilityOnnxRuntime

                release_dir = resolve_release_dir(settings)
                runtime = VolatilityOnnxRuntime.from_release_bundle(
                    release_dir,
                    public_key_path=Path(public_key_path),
                )
            except ReleaseBootstrapError as error:
                logger.warning("volatility release bootstrap failed: %s", error)
                return None, f"volatility release unavailable: {error}"
            except RuntimeError as error:
                logger.warning("volatility release failed to load: %s", error)
                return None, f"artifact integrity failure: {error}"
            except (OSError, ValueError) as error:
                logger.warning("volatility release unavailable: %s", error)
                return None, f"volatility release unavailable: {error}"
            self._runtime = runtime
            return runtime, None


_RELEASE_STATE = _ReleaseState()


def _reset_release_state() -> None:
    """Test and operations hook to force a fresh bundle verification."""
    global _RELEASE_STATE
    _RELEASE_STATE = _ReleaseState()


class _ResponseCache:
    """Small bounded TTL cache keyed by (signed model id, ticker, horizon)."""

    def __init__(self, max_entries: int = 128) -> None:
        self._entries: dict[tuple[str, str, int], tuple[float, dict[str, Any]]] = {}
        self._max_entries = max_entries

    def get(self, key: tuple[str, str, int]) -> dict[str, Any] | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        stored_at, payload = entry
        ttl = max(settings.volatility_forecast_cache_ttl, 0)
        if ttl == 0 or time.monotonic() - stored_at > ttl:
            self._entries.pop(key, None)
            return None
        return payload

    def put(self, key: tuple[str, str, int], payload: dict[str, Any]) -> None:
        if len(self._entries) >= self._max_entries:
            oldest = min(self._entries, key=lambda item: self._entries[item][0])
            self._entries.pop(oldest, None)
        self._entries[key] = (time.monotonic(), payload)


_response_cache = _ResponseCache()


def volatility_release_readiness() -> dict[str, Any]:
    """Return signed-release readiness without exposing internal errors."""
    if not release_source_configured(settings):
        return {
            "configured": False,
            "status": "unconfigured",
            "certified_horizons": [],
        }
    runtime, failure = _RELEASE_STATE.get()
    if runtime is None:
        return {
            # A configured source that cannot be verified is materially
            # different from no source at all.  Preserve that distinction so
            # readiness and operators report an integrity/availability fault
            # instead of a misleading configuration gap.
            "configured": True,
            "status": "integrity_failure" if failure and "integrity" in failure else "unavailable",
            "certified_horizons": [],
        }
    if runtime.news_status == "certified":
        if not settings.volatility_news_provider_enabled:
            return {
                "configured": True,
                "status": "news_input_unavailable",
                "model_id": runtime.model_id,
                "model_version": runtime.model_version,
                "metric_source": runtime.metric_source,
                "certification_scope": runtime.certification_scope,
                "news_status": runtime.news_status,
                "certified_heads": getattr(
                    runtime, "certified_heads", dict(LEGACY_CERTIFIED_HEADS)
                ),
                "certified_horizons": [],
            }
        return {
            "configured": True,
            "status": "ready",
            "model_id": runtime.model_id,
            "model_version": runtime.model_version,
            "metric_source": runtime.metric_source,
            "certification_scope": runtime.certification_scope,
            "news_status": runtime.news_status,
            "certified_heads": getattr(runtime, "certified_heads", dict(LEGACY_CERTIFIED_HEADS)),
            "news_provider_enabled": True,
            "certified_horizons": list(runtime.certified_horizon_list()),
        }
    return {
        "configured": True,
        "status": "ready",
        "model_id": runtime.model_id,
        "model_version": runtime.model_version,
        "metric_source": runtime.metric_source,
        "certification_scope": runtime.certification_scope,
        "news_status": runtime.news_status,
        "certified_heads": getattr(runtime, "certified_heads", dict(LEGACY_CERTIFIED_HEADS)),
        "certified_horizons": list(runtime.certified_horizon_list()),
    }


def _price_quantiles(
    current_price: float,
    cumulative_variance: float,
    horizon: int,
    *,
    cumulative_location: float = 0.0,
    distribution_family: str = "zero_location_normal",
    degrees_of_freedom: float | None = None,
) -> dict[str, list[float]]:
    """Interpolate a certified terminal log-return distribution to daily bands.

    The selected horizon is the certified endpoint. Intermediate sessions use
    transparent linear interpolation of cumulative location and variance; they
    are visualisation points, not separately certified horizons.
    """
    if horizon < 1:
        raise ValueError("quantile horizon must be positive")
    if (
        not np.isfinite(current_price)
        or current_price <= 0
        or not np.isfinite(cumulative_variance)
        or cumulative_variance <= 0
        or not np.isfinite(cumulative_location)
    ):
        raise ValueError("return-distribution inputs must be finite and positive where required")
    if distribution_family == "student_t":
        if (
            isinstance(degrees_of_freedom, bool)
            or not isinstance(degrees_of_freedom, (int, float))
            or not np.isfinite(float(degrees_of_freedom))
            or float(degrees_of_freedom) <= 2.0
        ):
            raise ValueError("Student-t distribution requires finite degrees of freedom above two")
        degrees = float(degrees_of_freedom)
        quantile_scores = {
            name: float(student_t.ppf(probability, df=degrees))
            for name, probability in _QUANTILE_PROBABILITIES.items()
        }
        variance_to_scale = (degrees - 2.0) / degrees
    elif distribution_family == "zero_location_normal":
        if degrees_of_freedom is not None or abs(cumulative_location) > 1e-15:
            raise ValueError("legacy normal cone must have zero location and no degrees of freedom")
        quantile_scores = _QUANTILE_Z_SCORES
        variance_to_scale = 1.0
    else:
        raise ValueError("return-distribution family is unsupported")
    return {
        name: [
            float(
                current_price
                * np.exp(
                    cumulative_location * day / horizon
                    + score * np.sqrt(cumulative_variance * day / horizon * variance_to_scale)
                )
            )
            for day in range(1, horizon + 1)
        ]
        for name, score in quantile_scores.items()
    }


def _abstain(detail: str) -> HTTPException:
    return HTTPException(
        status_code=503,
        detail={"status": "abstain_no_certified_model", "reason": detail},
    )


def _live_news_vector(
    runtime: Any,
    symbol: str,
    origin_date: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fetch the schema-exact live news vector at the origin-session cutoff.

    The cutoff is the origin session's official close (20:00 UTC): the
    inference snapshot is complete only after that close, and later events
    belong to future sessions. The provider additionally applies its own
    conservative availability delay. Any provider failure is a structured,
    fail-closed abstention.
    """
    import pandas as pd

    from services.news_aggregator import NewsProviderUnavailable, get_news_provider

    feature_names = tuple(getattr(runtime, "news_feature_names", ()) or ())
    if not feature_names:
        raise _abstain("the certified news release does not declare its news feature schema")
    cutoff = pd.Timestamp(origin_date, tz="UTC") + pd.Timedelta(hours=20)
    try:
        vector = get_news_provider().features_for(
            symbol,
            cutoff_at=cutoff,
            feature_names=feature_names,
        )
    except NewsProviderUnavailable as error:
        raise _abstain(f"live point-in-time news vector unavailable: {error}") from error
    evidence = {
        "provider_cutoff_utc": vector.cutoff_at,
        "eligible_article_count": int(vector.eligible_article_count),
        "news_feature_count": int(vector.values.size),
    }
    return vector.values, evidence


@router.get("/api/v2/forecast", response_model=VolatilityForecastResponse)
@limiter.limit("30/minute")
async def volatility_forecast_v2(
    request: Request,
    ticker: str = Query(..., min_length=1, max_length=12),
    horizon: int = Query(...),
) -> VolatilityForecastResponse:
    """Serve the certified volatility distribution for one supported horizon."""
    symbol = validate_ticker(ticker)
    if horizon not in VOLATILITY_HORIZONS:
        raise HTTPException(
            status_code=400,
            detail=f"horizon must be one of {list(VOLATILITY_HORIZONS)}",
        )

    runtime, load_failure = _RELEASE_STATE.get()
    if runtime is None:
        raise _abstain(load_failure or "no certified volatility model is available")
    if not runtime.is_certified_horizon(horizon):
        raise _abstain(f"the certified ensemble did not clear the {horizon}-session guardrails")
    if runtime.news_status == "certified" and not settings.volatility_news_provider_enabled:
        raise _abstain(
            "the signed release requires a live point-in-time news vector, but no "
            "production provider with the certified schema is configured"
        )
    # Include the signed model id so a promoted release cannot serve a prior
    # bundle's response until the generic TTL expires.
    cache_key = (runtime.model_id, symbol, horizon)
    cached = _response_cache.get(cache_key)
    if cached is not None:
        return VolatilityForecastResponse.model_validate(cached)

    from services.volatility_snapshot import build_volatility_inference_snapshot

    try:
        snapshot = build_volatility_inference_snapshot(symbol)
    except ValueError as error:
        raise HTTPException(
            status_code=409,
            detail=f"market history cannot support a certified forecast: {error}",
        ) from error
    except Exception as error:  # upstream market-data transport failures
        logger.warning("upstream market data failed for %s: %s", symbol, error)
        raise HTTPException(
            status_code=502,
            detail="upstream market data is temporarily unavailable",
        ) from error

    news_features: np.ndarray | None = None
    news_input: dict[str, Any] | None = None
    if runtime.news_status == "certified":
        news_features, news_input = _live_news_vector(runtime, symbol, snapshot.origin_date)

    try:
        if news_features is None:
            forecast = runtime.forecast(snapshot)
        else:
            forecast = runtime.forecast(snapshot, news_features=news_features)
    except RuntimeError as error:
        logger.error("certified inference failed for %s: %s", symbol, error)
        raise HTTPException(
            status_code=503,
            detail={
                "status": "artifact_integrity_failure",
                "reason": "the certified model could not produce a valid forecast",
            },
        ) from error

    horizon_index = VOLATILITY_HORIZONS.index(horizon)
    variance_at_horizon = float(forecast.forecast_variance[horizon_index])
    release_certified_heads = dict(
        getattr(runtime, "certified_heads", dict(LEGACY_CERTIFIED_HEADS))
    )
    # A V11.2 bundle may mix learned and HAR routes by horizon.  Use the
    # signed per-horizon declaration when available instead of promoting every
    # output merely because another horizon has a learned route.
    if hasattr(runtime, "is_return_distribution_horizon"):
        return_distribution_certified = bool(runtime.is_return_distribution_horizon(horizon))
    else:
        return_distribution_certified = release_certified_heads.get("return_distribution") is True
    certified_heads = {
        **release_certified_heads,
        "return_distribution": return_distribution_certified,
    }
    if hasattr(runtime, "certified_horizon_list"):
        certified_horizon_list = list(runtime.certified_horizon_list())
    else:
        certified_horizon_list = [horizon] if runtime.is_certified_horizon(horizon) else []
    if hasattr(runtime, "return_distribution_horizon_list"):
        return_distribution_horizon_list = list(runtime.return_distribution_horizon_list())
    else:
        return_distribution_horizon_list = [horizon] if return_distribution_certified else []
    if return_distribution_certified:
        cumulative_location = float(forecast.return_location[horizon_index])
        distribution_variance = float(forecast.return_variance[horizon_index])
        distribution_family = str(runtime.return_distribution_family)
        distribution_df = runtime.return_distribution_degrees_of_freedom
    else:
        cumulative_location = 0.0
        distribution_variance = variance_at_horizon
        distribution_family = "zero_location_normal"
        distribution_df = None
    daily_variance = variance_at_horizon / horizon
    future_dates = tuple(snapshot.future_dates[:horizon])
    if len(future_dates) != horizon:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "forecast_calendar_unavailable",
                "reason": "future trading dates could not be generated for this forecast",
            },
        )
    evidence = {
        "model_id": runtime.model_id,
        "model_version": runtime.model_version,
        "member_seeds": list(runtime.member_seeds),
        "snapshot_id": snapshot.snapshot_id,
        "metric_source": runtime.metric_source,
        "certification_scope": runtime.certification_scope,
        "news_enabled": runtime.news_status == "certified",
        "news_status": runtime.news_status,
        "news_input": news_input,
        "quantile_model": (
            "student_t_return_distribution: terminal location and variance are certified; "
            "intermediate sessions linearly interpolate cumulative moments; direction head "
            "is not certified"
            if return_distribution_certified
            else "zero_location_volatility_cone: bands derive from the certified variance "
            "around the unchanged close; no learned direction claim"
        ),
        "certified_heads": certified_heads,
        "certified_head_horizons": {
            "volatility": certified_horizon_list,
            "return_distribution": return_distribution_horizon_list,
            "direction": [],
        },
        "certified": True,
    }
    certification_summary = runtime.certification_summary(horizon)
    if certification_summary is not None:
        evidence["horizon_certification"] = {str(horizon): certification_summary}
    payload = {
        "ticker": symbol,
        "as_of": snapshot.origin_date,
        "horizon": int(horizon),
        "current_price": float(snapshot.origin_close),
        "historical_dates": list(snapshot.historical_dates),
        "historical_prices": [float(value) for value in snapshot.historical_prices],
        "forecast": {
            "price_quantiles": _price_quantiles(
                snapshot.origin_close,
                distribution_variance,
                horizon,
                cumulative_location=cumulative_location,
                distribution_family=distribution_family,
                degrees_of_freedom=distribution_df,
            ),
            "future_dates": list(future_dates),
            "probability_up": None,
            "expected_cumulative_variance": variance_at_horizon,
            "expected_cumulative_return": (
                cumulative_location if return_distribution_certified else None
            ),
            "return_distribution_variance": (
                distribution_variance if return_distribution_certified else None
            ),
            "return_distribution_family": distribution_family,
            "expected_annualized_volatility": float(
                np.sqrt(daily_variance * TRADING_SESSIONS_PER_YEAR)
            ),
        },
        "evidence": evidence,
    }
    _response_cache.put(cache_key, payload)
    return VolatilityForecastResponse.model_validate(payload)
