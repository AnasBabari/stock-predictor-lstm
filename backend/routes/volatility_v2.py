"""Certified global-volatility forecast endpoint (v2, fail-closed).

One frozen ensemble serves every supported ticker. The return-distribution
head is withheld until its own evidence gate passes, so price ranges are
reconstructed as zero-location volatility cones around the current close and
labelled accordingly. Without a signed, verified release the endpoint answers
with an explicit no-certified-model abstention instead of any baseline path.
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

from config import settings
from routes.common import limiter, validate_ticker
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

CERTIFIED_HEADS = {"volatility": True, "return_distribution": False, "direction": False}


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
            release_dir = settings.volatility_release_dir
            public_key_path = settings.volatility_public_key_path
            if not release_dir or not public_key_path:
                return None, "no certified volatility release is configured"
            try:
                from services.volatility_runtime import VolatilityOnnxRuntime

                runtime = VolatilityOnnxRuntime.from_release_bundle(
                    Path(release_dir),
                    public_key_path=Path(public_key_path),
                )
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
    if not settings.volatility_release_dir or not settings.volatility_public_key_path:
        return {
            "configured": False,
            "status": "unconfigured",
            "certified_horizons": [],
        }
    runtime, failure = _RELEASE_STATE.get()
    if runtime is None:
        return {
            "configured": False,
            "status": "integrity_failure" if failure and "integrity" in failure else "unavailable",
            "certified_horizons": [],
        }
    return {
        "configured": True,
        "status": "ready",
        "model_id": runtime.model_id,
        "certified_horizons": list(runtime.certified_horizon_list()),
    }


def _price_quantiles(
    current_price: float,
    cumulative_variance: float,
    horizon: int,
) -> dict[str, list[float]]:
    """Zero-location lognormal bands derived from certified variance only.

    The certified head emits cumulative variance at selected horizons. Between
    the origin and a requested horizon we expose a transparent linear variance
    cone, never a learned directional path.
    """
    if horizon < 1:
        raise ValueError("quantile horizon must be positive")
    return {
        name: [
            float(
                current_price * np.exp(z * np.sqrt(max(cumulative_variance, 0.0) * day / horizon))
            )
            for day in range(1, horizon + 1)
        ]
        for name, z in _QUANTILE_Z_SCORES.items()
    }


def _abstain(detail: str) -> HTTPException:
    return HTTPException(
        status_code=503,
        detail={"status": "abstain_no_certified_model", "reason": detail},
    )


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

    try:
        forecast = runtime.forecast(snapshot)
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
        "member_seeds": list(runtime.member_seeds),
        "snapshot_id": snapshot.snapshot_id,
        "metric_source": "locked_purged_walk_forward",
        "quantile_model": (
            "zero_location_volatility_cone: bands derive from the certified "
            "variance around the unchanged close; no learned direction claim"
        ),
        "certified_heads": dict(CERTIFIED_HEADS),
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
                variance_at_horizon,
                horizon,
            ),
            "future_dates": list(future_dates),
            "probability_up": None,
            "expected_cumulative_variance": variance_at_horizon,
            "expected_annualized_volatility": float(
                np.sqrt(daily_variance * TRADING_SESSIONS_PER_YEAR)
            ),
        },
        "evidence": evidence,
    }
    _response_cache.put(cache_key, payload)
    return VolatilityForecastResponse.model_validate(payload)
