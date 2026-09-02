"""Active volatility forecasting route.

This route is intentionally separate from ``volatility_v2``.  The v2 route is
the historical signed-release contract and remains fail-closed for clients
that explicitly depend on certification.  The active product route serves a
causal, reproducible statistical baseline while the offline learned-model
benchmark is being rebuilt.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from data_pipeline import MarketDataUnavailable, MarketTransportError, UnknownTickerError
from features.market import MarketContextUnavailable
from routes.common import limiter, validate_ticker
from services.forecast_ledger import get_current_code_commit, get_forecast_ledger
from services.live_volatility import SUPPORTED_BASELINES, build_live_volatility_forecast
from services.volatility_snapshot import VOLATILITY_HORIZONS, build_volatility_inference_snapshot

logger = logging.getLogger(__name__)
router = APIRouter(tags=["volatility"])

MARKET_DATA_UNAVAILABLE_BODY = {
    "error": "MARKET_DATA_UNAVAILABLE",
    "message": "Current market data is temporarily unavailable. Please try again later.",
}


@router.get("/api/v1/volatility/forecast")
@limiter.limit("30/minute")
def volatility_forecast(
    request: Request,
    ticker: str = Query(default="AAPL", min_length=1, max_length=12),
    horizon: int = Query(default=7, ge=1, le=30),
    model: str = Query(default="auto"),
    record_ledger: bool = Query(default=True),
) -> Any:
    """Return a causal volatility cone for one supported trading horizon.

    The endpoint never accepts client-provided features or model paths.  It
    downloads the latest validated OHLCV snapshot, computes features through
    the last observed session, and applies the selected train-free baseline.
    """

    symbol = validate_ticker(ticker)
    if horizon not in VOLATILITY_HORIZONS:
        raise HTTPException(
            status_code=400,
            detail=f"Horizon must be one of {list(VOLATILITY_HORIZONS)}.",
        )
    requested_model = str(model).strip().lower()
    if requested_model not in SUPPORTED_BASELINES:
        raise HTTPException(
            status_code=400,
            detail=f"Model must be one of {list(SUPPORTED_BASELINES)}.",
        )

    try:
        snapshot = build_volatility_inference_snapshot(symbol)
        forecast_result = build_live_volatility_forecast(
            snapshot,
            horizon=horizon,
            model=requested_model,
        )

        # Record forecast in persistent ledger for subsequent track record scoring
        try:
            if not record_ledger:
                return forecast_result
            ledger = get_forecast_ledger()
            future_dates = forecast_result.get("forecast", {}).get("future_dates", [])
            target_date = future_dates[-1] if future_dates else ""
            quantiles = forecast_result.get("forecast", {}).get("price_quantiles", {})
            p05 = quantiles.get("p05", [snapshot.origin_close])[-1]
            p95 = quantiles.get("p95", [snapshot.origin_close])[-1]
            pred_vol = float(forecast_result.get("forecast", {}).get("predicted_volatility", 0.0))
            recent_vol = float(snapshot.baseline_candidates.get("rolling_c2c_20", [pred_vol])[0])
            active_model = str(forecast_result.get("forecast", {}).get("model", requested_model))

            ledger.record_forecast(
                forecast_date=snapshot.origin_date,
                ticker=symbol,
                horizon=horizon,
                target_date=target_date,
                model_name=active_model,
                predicted_volatility=pred_vol,
                recent_realized_volatility=recent_vol,
                origin_price=float(snapshot.origin_close),
                lower_scenario_price=float(p05),
                upper_scenario_price=float(p95),
                record_source="live",
                model_version="deployable_v5",
                feature_set_version="deployable_feature_columns_v5",
                code_commit=get_current_code_commit(),
                data_as_of=snapshot.origin_date,
                data_provider=getattr(snapshot, "data_provider", "unknown"),
            )
        except Exception as ledger_err:
            logger.warning("Could not record forecast to ledger for %s: %s", symbol, ledger_err)

        return forecast_result
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


@router.get("/api/v1/volatility/ledger")
@limiter.limit("30/minute")
def get_volatility_ledger_route(
    request: Request,
    ticker: str = Query(default="AAPL", min_length=1, max_length=12),
    horizon: int | None = Query(default=None),
    record_source: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    """Retrieve historical forecast ledger and empirical accuracy track record."""
    symbol = validate_ticker(ticker)
    ledger = get_forecast_ledger()

    entries = ledger.get_ledger_entries(
        ticker=symbol, horizon=horizon, record_source=record_source, limit=limit
    )
    if not entries:
        try:
            from data_pipeline import fetch_historical_frame

            df = fetch_historical_frame(symbol)
            if df is not None and not df.empty:
                replay_model = "garch_11" if horizon == 1 else "rolling_mean"
                ledger.generate_historical_replay_ledger(
                    symbol, df, horizon=horizon or 5, model_name=replay_model
                )
                entries = ledger.get_ledger_entries(
                    ticker=symbol, horizon=horizon, record_source=record_source, limit=limit
                )
        except Exception as seed_err:
            logger.debug("Could not auto-generate historical replay for %s: %s", symbol, seed_err)

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


@router.post("/api/v1/volatility/score-ledger")
@limiter.limit("10/minute")
def score_volatility_ledger_route(
    request: Request,
    ticker: str = Query(default="AAPL", min_length=1, max_length=12),
) -> dict[str, Any]:
    """Score pending ledger forecasts against subsequent realized volatility."""
    symbol = validate_ticker(ticker)
    ledger = get_forecast_ledger()
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
    except Exception as err:
        logger.exception("Failed to score volatility ledger for %s", symbol)
        raise HTTPException(status_code=500, detail=str(err)) from err


__all__ = [
    "router",
    "volatility_forecast",
    "get_volatility_ledger_route",
    "score_volatility_ledger_route",
]
