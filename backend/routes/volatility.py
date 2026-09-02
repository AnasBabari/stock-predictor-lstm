"""Active volatility forecasting route.

This route is intentionally separate from ``volatility_v2``.  The v2 route is
the historical signed-release contract and remains fail-closed for clients
that explicitly depend on certification.  The active product route serves a
causal, reproducible statistical baseline while the offline learned-model
benchmark is being rebuilt.
"""

from __future__ import annotations

import logging
import os
import secrets
from dataclasses import dataclass
from datetime import date
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from data_pipeline import MarketDataUnavailable, MarketTransportError, UnknownTickerError
from features.market import MarketContextUnavailable
from routes.common import limiter, validate_ticker
from services.forecast_ledger import (
    LedgerConflictError,
    LedgerUnavailableError,
    compute_forecast_fingerprint,
    get_current_code_commit,
    get_forecast_ledger,
)
from services.live_collection import (
    LIVE_MODEL_POLICY_V1,
    LIVE_START_DATE,
    LIVE_UNIVERSE_VERSION,
    validate_live_collection_item,
)
from services.live_volatility import SUPPORTED_BASELINES, build_live_volatility_forecast
from services.volatility_contract import (
    AUTO_MODEL_POLICY,
    VOLATILITY_FEATURE_SET_VERSION,
    VOLATILITY_MODEL_POLICY_VERSION,
    VOLATILITY_MODEL_VERSION,
    validate_volatility_horizon,
)
from services.volatility_snapshot import VOLATILITY_HORIZONS, build_volatility_inference_snapshot

logger = logging.getLogger(__name__)
router = APIRouter(tags=["volatility"])

MARKET_DATA_UNAVAILABLE_BODY = {
    "error": "MARKET_DATA_UNAVAILABLE",
    "message": "Current market data is temporarily unavailable. Please try again later.",
}

FORECAST_LEDGER_UNAVAILABLE_BODY = {
    "error": "FORECAST_LEDGER_UNAVAILABLE",
    "message": "The forecast was not recorded because the forecast ledger is temporarily unavailable.",
}

COLLECTOR_TOKEN_ENV = "FORECAST_COLLECTOR_TOKEN"


@dataclass(frozen=True)
class _PreparedForecast:
    result: dict[str, Any]
    data_as_of: str
    record_kwargs: dict[str, Any]


def require_collector_auth(
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    """Require the server-side collector bearer token without logging it."""
    configured = os.getenv(COLLECTOR_TOKEN_ENV, "")
    if not configured:
        raise HTTPException(status_code=503, detail="Collector authentication is unavailable.")
    scheme, separator, candidate = (authorization or "").partition(" ")
    authenticated = (
        separator == " "
        and scheme.lower() == "bearer"
        and bool(candidate)
        and secrets.compare_digest(candidate, configured)
    )
    if not authenticated:
        raise HTTPException(
            status_code=401,
            detail="Collector authentication failed.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _prepare_forecast(symbol: str, horizon: int, requested_model: str) -> _PreparedForecast:
    snapshot = build_volatility_inference_snapshot(symbol)
    forecast_result = build_live_volatility_forecast(
        snapshot,
        horizon=horizon,
        model=requested_model,
    )
    code_commit = get_current_code_commit()
    data_as_of = str(getattr(snapshot, "data_as_of", None) or snapshot.origin_date)
    data_provider = getattr(snapshot, "data_provider", "unknown")
    evidence = forecast_result.setdefault("evidence", {})
    evidence.update(
        {
            "code_commit": code_commit,
            "model_policy_version": VOLATILITY_MODEL_POLICY_VERSION,
            "auto_model_policy": {
                str(policy_horizon): policy_model
                for policy_horizon, policy_model in AUTO_MODEL_POLICY.items()
            },
        }
    )

    future_dates = forecast_result.get("forecast", {}).get("future_dates", [])
    target_date = future_dates[-1] if future_dates else ""
    quantiles = forecast_result.get("forecast", {}).get("price_quantiles", {})
    p05 = quantiles.get("p05", [snapshot.origin_close])[-1]
    p95 = quantiles.get("p95", [snapshot.origin_close])[-1]
    pred_vol = float(forecast_result.get("forecast", {}).get("predicted_volatility", 0.0))
    recent_values = snapshot.baseline_candidates.get("rolling_c2c_20", [pred_vol])
    recent_vol = float(recent_values[0])
    active_model = str(forecast_result.get("forecast", {}).get("model", requested_model))
    record_kwargs = {
        "forecast_date": snapshot.origin_date,
        "ticker": symbol,
        "horizon": horizon,
        "target_date": target_date,
        "model_name": active_model,
        "predicted_volatility": pred_vol,
        "recent_realized_volatility": recent_vol,
        "origin_price": float(snapshot.origin_close),
        "lower_scenario_price": float(p05),
        "upper_scenario_price": float(p95),
        "record_source": "live",
        "model_version": VOLATILITY_MODEL_VERSION,
        "feature_set_version": VOLATILITY_FEATURE_SET_VERSION,
        "code_commit": code_commit,
        "data_as_of": data_as_of,
        "data_provider": data_provider,
    }
    evidence["forecast_fingerprint"] = compute_forecast_fingerprint(**record_kwargs)
    return _PreparedForecast(
        result=forecast_result,
        data_as_of=data_as_of,
        record_kwargs=record_kwargs,
    )


def _execute_forecast(
    symbol: str,
    horizon: int,
    requested_model: str,
    *,
    record_live: bool,
) -> Any:
    try:
        prepared = _prepare_forecast(symbol, horizon, requested_model)
        evidence = prepared.result.setdefault("evidence", {})
        if not record_live:
            evidence["ledger_write"] = "disabled_public_preview"
            return prepared.result

        if date.fromisoformat(prepared.data_as_of) < LIVE_START_DATE:
            return JSONResponse(
                status_code=409,
                content={
                    "error": "LIVE_COLLECTION_NOT_ELIGIBLE",
                    "message": "The market snapshot predates the frozen live collection start.",
                },
            )
        expected_model = LIVE_MODEL_POLICY_V1[horizon]
        if prepared.record_kwargs["model_name"] != expected_model:
            logger.error("Frozen model policy mismatch for %s horizon %s", symbol, horizon)
            return JSONResponse(
                status_code=409,
                content={
                    "error": "LIVE_COLLECTION_POLICY_MISMATCH",
                    "message": "The forecast did not match the frozen live model policy.",
                },
            )
        ledger = get_forecast_ledger()
        record = ledger.record_forecast(**prepared.record_kwargs)
        evidence.update(
            {
                "ledger_write": "recorded_live",
                "live_universe_version": LIVE_UNIVERSE_VERSION,
                "ledger_record_id": record.id,
            }
        )
        return prepared.result
    except LedgerConflictError:
        logger.error("Conflicting immutable ledger evidence for %s", symbol)
        return JSONResponse(
            status_code=409,
            content={
                "error": "FORECAST_LEDGER_CONFLICT",
                "message": "A forecast with this origin already exists with different evidence.",
            },
        )
    except LedgerUnavailableError:
        logger.error("Forecast ledger unavailable for %s", symbol)
        return JSONResponse(status_code=503, content=FORECAST_LEDGER_UNAVAILABLE_BODY)
    except UnknownTickerError as err:
        raise HTTPException(
            status_code=404, detail="No market data is available for this ticker."
        ) from err
    except (MarketTransportError, MarketContextUnavailable) as err:
        logger.warning("Market data unavailable for %s: %s", symbol, type(err).__name__)
        return JSONResponse(status_code=503, content=MARKET_DATA_UNAVAILABLE_BODY)
    except MarketDataUnavailable as err:
        raise HTTPException(
            status_code=422,
            detail="Not enough valid market history is available for this ticker.",
        ) from err
    except ValueError as err:
        logger.info("Volatility request rejected for %s: %s", symbol, err)
        raise HTTPException(
            status_code=422,
            detail="The volatility snapshot could not be formed from valid market data.",
        ) from err
    except Exception as err:
        logger.exception("Volatility forecast failed for %s", symbol)
        raise HTTPException(
            status_code=503,
            detail="Volatility forecasting is temporarily unavailable. Please retry shortly.",
        ) from err


@router.get("/api/v1/volatility/forecast")
@limiter.limit("30/minute")
def volatility_forecast(
    request: Request,
    ticker: str = Query(default="AAPL", min_length=1, max_length=12),
    horizon: int = Query(default=5, ge=1, le=20),
    model: str = Query(default="auto"),
) -> Any:
    """Return a read-only causal volatility preview for one trading horizon.

    The endpoint never accepts client-provided features or model paths.  It
    downloads the latest validated OHLCV snapshot, computes features through
    the last observed session, and applies the selected train-free baseline.
    Public GET requests never mutate the forecast ledger.
    """

    symbol = validate_ticker(ticker)
    try:
        horizon = validate_volatility_horizon(horizon)
    except ValueError as err:
        raise HTTPException(
            status_code=400,
            detail=f"Horizon must be one of {list(VOLATILITY_HORIZONS)}.",
        ) from err
    requested_model = str(model).strip().lower()
    if requested_model not in SUPPORTED_BASELINES:
        raise HTTPException(
            status_code=400,
            detail=f"Model must be one of {list(SUPPORTED_BASELINES)}.",
        )
    return _execute_forecast(symbol, horizon, requested_model, record_live=False)


@router.post("/api/v1/volatility/collect")
@limiter.limit("30/minute")
def collect_volatility_forecast(
    request: Request,
    _authorization: Annotated[None, Depends(require_collector_auth)],
    ticker: str = Query(min_length=1, max_length=12),
    horizon: int = Query(ge=1, le=20),
) -> Any:
    """Record one authenticated item from the frozen live collection contract."""
    symbol = validate_ticker(ticker)
    try:
        symbol, normalized_horizon = validate_live_collection_item(symbol, horizon)
    except (TypeError, ValueError) as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    return _execute_forecast(symbol, normalized_horizon, "auto", record_live=True)


@router.get("/api/v1/volatility/ledger")
@limiter.limit("30/minute")
def get_volatility_ledger_route(
    request: Request,
    ticker: str = Query(default="AAPL", min_length=1, max_length=12),
    horizon: int | None = Query(default=None, ge=1, le=20),
    record_source: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    """Retrieve historical forecast ledger and empirical accuracy track record."""
    symbol = validate_ticker(ticker)
    if horizon is not None:
        try:
            horizon = validate_volatility_horizon(horizon)
        except ValueError as err:
            raise HTTPException(
                status_code=400,
                detail=f"Horizon must be one of {list(VOLATILITY_HORIZONS)}.",
            ) from err
    try:
        ledger = get_forecast_ledger()
    except LedgerUnavailableError:
        logger.error("Forecast ledger unavailable while reading %s", symbol)
        return JSONResponse(status_code=503, content=FORECAST_LEDGER_UNAVAILABLE_BODY)

    try:
        entries = ledger.get_ledger_entries(
            ticker=symbol, horizon=horizon, record_source=record_source, limit=limit
        )
        live_metrics = ledger.get_track_record_metrics(
            ticker=symbol, horizon=horizon, record_source="live"
        )
        replay_metrics = ledger.get_track_record_metrics(
            ticker=symbol, horizon=horizon, record_source="historical_replay"
        )

        return {
            "ticker": symbol,
            "horizon": horizon,
            "live_track_record": live_metrics,
            "replay_track_record": replay_metrics,
            "track_record": live_metrics,
            "entries": entries,
        }
    except LedgerUnavailableError:
        logger.error("Forecast ledger unavailable while reading %s", symbol)
        return JSONResponse(status_code=503, content=FORECAST_LEDGER_UNAVAILABLE_BODY)
    except Exception:
        logger.exception("Could not read forecast ledger for %s", symbol)
        return JSONResponse(status_code=503, content=FORECAST_LEDGER_UNAVAILABLE_BODY)


@router.get("/api/v1/volatility/export-ledger")
@limiter.limit("10/minute")
def export_live_volatility_ledger(
    request: Request,
    _authorization: Annotated[None, Depends(require_collector_auth)],
) -> Any:
    """Return the complete deterministic live track for authenticated backups."""
    try:
        ledger = get_forecast_ledger()
        records = [record.to_dict() for record in ledger.export_records(record_source="live")]
        return {
            "record_source": "live",
            "storage_backend": ledger.storage_kind,
            "entries": records,
        }
    except LedgerUnavailableError:
        return JSONResponse(status_code=503, content=FORECAST_LEDGER_UNAVAILABLE_BODY)
    except Exception:
        logger.exception("Could not export the live volatility ledger")
        return JSONResponse(status_code=503, content=FORECAST_LEDGER_UNAVAILABLE_BODY)


@router.post("/api/v1/volatility/score-ledger")
@limiter.limit("10/minute")
def score_volatility_ledger_route(
    request: Request,
    _authorization: Annotated[None, Depends(require_collector_auth)],
    ticker: str = Query(default="AAPL", min_length=1, max_length=12),
) -> dict[str, Any]:
    """Score pending ledger forecasts against subsequent realized volatility."""
    symbol = validate_ticker(ticker)
    try:
        ledger = get_forecast_ledger()
    except LedgerUnavailableError:
        logger.error("Forecast ledger unavailable while scoring %s", symbol)
        return JSONResponse(status_code=503, content=FORECAST_LEDGER_UNAVAILABLE_BODY)
    try:
        from data_pipeline import fetch_historical_frame

        df = fetch_historical_frame(symbol)
        if df is None or df.empty:
            raise HTTPException(
                status_code=422, detail="No historical market data available for scoring."
            )
        scored_count = ledger.score_pending_forecasts(symbol, df)
        live_metrics = ledger.get_track_record_metrics(ticker=symbol, record_source="live")
        replay_metrics = ledger.get_track_record_metrics(
            ticker=symbol, record_source="historical_replay"
        )
        return {
            "ticker": symbol,
            "scored_count": scored_count,
            "live_track_record": live_metrics,
            "replay_track_record": replay_metrics,
            "track_record": live_metrics,
        }
    except HTTPException:
        raise
    except LedgerUnavailableError:
        logger.error("Forecast ledger unavailable while scoring %s", symbol)
        return JSONResponse(status_code=503, content=FORECAST_LEDGER_UNAVAILABLE_BODY)
    except Exception:
        logger.exception("Failed to score volatility ledger for %s", symbol)
        return JSONResponse(status_code=503, content=FORECAST_LEDGER_UNAVAILABLE_BODY)


__all__ = [
    "collect_volatility_forecast",
    "export_live_volatility_ledger",
    "require_collector_auth",
    "router",
    "volatility_forecast",
    "get_volatility_ledger_route",
    "score_volatility_ledger_route",
]
