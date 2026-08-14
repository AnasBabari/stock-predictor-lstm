"""Training-data route delivering bounded feature snapshots for browser training."""

from __future__ import annotations

import asyncio
import copy
import logging
import time
from collections import OrderedDict

from fastapi import APIRouter, HTTPException, Request

from data_pipeline import MarketDataUnavailable, MarketTransportError
from features.market import MarketContextUnavailable
from routes.common import limiter, validate_ticker

logger = logging.getLogger(__name__)
router = APIRouter(tags=["training-data"])

# ── Training Snapshot Cache & Concurrency (Process-Local) ────────────
# Note: This semaphore, in-flight coalescing map, and LRU cache are process-local
# to each backend worker. Edge/API rate limiting (10/minute) provides outer bounding.
_training_semaphore = asyncio.Semaphore(2)
_snapshot_lock = asyncio.Lock()
_in_flight_tasks: dict[str, asyncio.Task] = {}
_snapshot_cache: OrderedDict[str, tuple[float, dict]] = OrderedDict()
_SNAPSHOT_CACHE_TTL = 60.0
_SNAPSHOT_CACHE_MAX = 6


async def _execute_snapshot_build(ticker: str) -> dict:
    import api

    async with _training_semaphore:
        return await asyncio.to_thread(api.build_training_snapshot, ticker)


@router.get("/api/v1/training-data")
@limiter.limit("10/minute")
async def training_data(request: Request, ticker: str = "AAPL"):
    """Return a bounded feature snapshot for browser-side training."""
    import api

    ticker = validate_ticker(ticker)

    cache = getattr(api, "_snapshot_cache", _snapshot_cache)
    lock = getattr(api, "_snapshot_lock", _snapshot_lock)
    tasks = getattr(api, "_in_flight_tasks", _in_flight_tasks)

    # 1. Check short-lived cache (hits bypass semaphore and update LRU order)
    async with lock:
        now = time.time()
        if ticker in cache:
            cached_time, cached_payload = cache[ticker]
            if now - cached_time < _SNAPSHOT_CACHE_TTL:
                cache.move_to_end(ticker)
                return copy.deepcopy(cached_payload)
            del cache[ticker]

        # 2. Coalesce in-flight builds for the same ticker
        task = tasks.get(ticker)
        if task is None or task.done():
            task = asyncio.create_task(
                getattr(api, "_execute_snapshot_build", _execute_snapshot_build)(ticker)
            )
            tasks[ticker] = task

            def _cleanup(t: asyncio.Task, tkr: str = ticker) -> None:
                async def _remove_task():
                    async with lock:
                        if tasks.get(tkr) is t:
                            tasks.pop(tkr, None)

                asyncio.create_task(_remove_task())

            task.add_done_callback(_cleanup)

    try:
        # Shield task so client cancellation doesn't kill shared in-flight build
        result = await asyncio.shield(task)
        completion_time = time.time()
        async with lock:
            if len(cache) >= _SNAPSHOT_CACHE_MAX:
                cache.popitem(last=False)
            cache[ticker] = (completion_time, copy.deepcopy(result))
            cache.move_to_end(ticker)
        return result
    except MarketContextUnavailable as err:
        raise HTTPException(status_code=503, detail=str(err)) from err
    except (ValueError, MarketDataUnavailable) as err:
        safe_messages = (
            "Not enough historical data",
            "Not enough feature rows",
            "No market data is available",
        )
        detail = (
            str(err) if str(err).startswith(safe_messages) else "Invalid input data for training."
        )
        raise HTTPException(status_code=400, detail=detail) from err
    except MarketTransportError as err:
        raise HTTPException(status_code=503, detail=str(err)) from err
    except Exception as err:
        logger.exception("Training-data snapshot failed for %s", ticker)
        raise HTTPException(
            status_code=503,
            detail="Training data is temporarily unavailable. Please try again later.",
        ) from err
