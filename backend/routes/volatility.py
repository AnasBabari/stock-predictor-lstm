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

from data_pipeline import MarketDataUnavailable, MarketTransportError, UnknownTickerError
from features.market import MarketContextUnavailable
from routes.common import limiter, validate_ticker
from services.live_volatility import SUPPORTED_BASELINES, build_live_volatility_forecast
from services.volatility_snapshot import VOLATILITY_HORIZONS, build_volatility_inference_snapshot

logger = logging.getLogger(__name__)
router = APIRouter(tags=["volatility"])


@router.get("/api/v1/volatility/forecast")
@limiter.limit("30/minute")
def volatility_forecast(
    request: Request,
    ticker: str = Query(default="AAPL", min_length=1, max_length=12),
    horizon: int = Query(default=7, ge=1, le=30),
    model: str = Query(default="har_rv"),
) -> dict[str, Any]:
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
        return build_live_volatility_forecast(
            snapshot,
            horizon=horizon,
            model=requested_model,
        )
    except UnknownTickerError as err:
        raise HTTPException(
            status_code=404, detail="No market data is available for this ticker."
        ) from err
    except (MarketTransportError, MarketContextUnavailable) as err:
        raise HTTPException(
            status_code=503,
            detail="Market data is temporarily unavailable. Please retry shortly.",
        ) from err
    except MarketDataUnavailable as err:
        raise HTTPException(
            status_code=422,
            detail="Not enough valid market history is available for this ticker.",
        ) from err
    except ValueError as err:
        # Validation failures are safe to expose only as a generic contract
        # error; provider internals and filesystem details stay private.
        logger.info("Volatility request rejected for %s: %s", symbol, err)
        raise HTTPException(
            status_code=422,
            detail="The volatility snapshot could not be formed from valid market data.",
        ) from err
    except Exception as err:  # pragma: no cover - defensive production boundary
        logger.exception("Volatility forecast failed for %s", symbol)
        raise HTTPException(
            status_code=503,
            detail="Volatility forecasting is temporarily unavailable. Please retry shortly.",
        ) from err


__all__ = ["router", "volatility_forecast"]
