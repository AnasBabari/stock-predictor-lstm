"""Forecast compatibility routes, baselines, and execution telemetry."""

from __future__ import annotations

import asyncio
import copy
import logging
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
from cachetools import TTLCache
from fastapi import APIRouter, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from calendars import future_trading_dates as default_future_trading_dates
from config import (
    APP_VERSION,
    DEFAULT_FORECAST_DAYS,
    FEATURES,
    MAX_FORECAST_DAYS,
    SCHEMA_VERSION,
    WINDOW_SIZE,
    settings,
)
from data_pipeline import MarketDataUnavailable, MarketTransportError, UnknownTickerError
from data_pipeline import fetch_data as default_fetch_data
from features.market import MarketContextUnavailable
from news_features import get_live_financial_sentiment as get_financial_sentiment
from routes.common import limiter, validate_ticker
from server_models.response_models import (
    TIMING_FIELDS,
    DirectionForecastResponse,
    PredictionStatusResponse,
    PriceForecastResponse,
)
from services.baselines import base_rate_direction_forecast, persistence_price_forecast

logger = logging.getLogger(__name__)
router = APIRouter(tags=["forecasts"])


class ServiceBusyError(RuntimeError):
    """The bounded prediction executor has no queue capacity."""


class ArtifactValidationError(RuntimeError):
    """Compatibility exception retained while server artifacts are disabled."""


class TrainingCapacityError(RuntimeError):
    """Compatibility exception retained for legacy clients."""


class PredictionJob:
    """Shared, in-process state for one bounded prediction job."""

    def __init__(self, key: str):
        self.key = key
        self.created_at = time.perf_counter()
        self.started_at: float | None = None
        self.stage = "queued"
        self.status = "queued"
        self.timings: dict[str, float | None] = {name: None for name in TIMING_FIELDS}
        self.artifact_state_before = "missing"
        self.artifact_action = "not_applicable"
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            self.started_at = time.perf_counter()
            self.status = "running"

    def set_stage(self, stage: str) -> None:
        with self._lock:
            self.stage = stage
            self.status = "running"

    def add_timing(self, name: str, duration: float) -> None:
        with self._lock:
            current = self.timings[name]
            self.timings[name] = round((current or 0.0) + duration, 4)

    def set_artifact(self, state: str, action: str) -> None:
        with self._lock:
            self.artifact_state_before = state
            self.artifact_action = action

    def finish(self, succeeded: bool) -> None:
        with self._lock:
            self.stage = "completed" if succeeded else "failed"
            self.status = self.stage

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "status": self.status,
                "stage": self.stage,
                "timings": dict(self.timings),
                "artifact_state_before": self.artifact_state_before,
                "artifact_action": self.artifact_action,
                "started_at": self.started_at,
            }


class PredictionStatusRegistry:
    """Bounded request views over coordinator-owned jobs; no exception details."""

    def __init__(self, max_entries: int = 512, ttl_seconds: int = 600):
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.watchdog_multiplier = 60
        self._lock = threading.Lock()
        self._views: dict[str, dict] = {}

    def _prune(self, now: float) -> None:
        expired = [key for key, view in self._views.items() if view["expires_at"] <= now]
        for key in expired:
            self._views.pop(key, None)

    def _make_room(self, now: float) -> bool:
        self._prune(now)
        if len(self._views) < self.max_entries:
            return True
        terminal = next((key for key, view in self._views.items() if view["terminal"]), None)
        if terminal is None:
            return False
        self._views.pop(terminal, None)
        return True

    def attach(
        self, request_id: str, job: PredictionJob, coalesced: bool, terminal: bool = False
    ) -> bool:
        with self._lock:
            if not self._make_room(time.monotonic()):
                return False
            self._views[request_id] = {
                "job": job,
                "coalesced": coalesced,
                "terminal": terminal,
                "lifecycle": "completed" if terminal else "active",
                "expires_at": time.monotonic()
                + self.ttl_seconds * (self.watchdog_multiplier if not terminal else 1),
            }
            return True

    def cache_hit(self, request_id: str, work_key: str) -> bool:
        job = PredictionJob(work_key)
        job.start()
        job.finish(True)
        return self.attach(request_id, job, False, terminal=True)

    def finish_view(self, request_id: str, succeeded: bool) -> None:
        with self._lock:
            view = self._views.get(request_id)
            if view is not None:
                view["terminal"] = True
                view["lifecycle"] = "completed" if succeeded else "failed"
                view["expires_at"] = time.monotonic() + self.ttl_seconds

    def get(self, request_id: str) -> dict | None:
        with self._lock:
            self._prune(time.monotonic())
            view = self._views.get(request_id)
            if view is None:
                return None
            job = view["job"].snapshot()
            return {
                "status": view["lifecycle"] if view["terminal"] else job["status"],
                "stage": job["stage"],
                "coalesced": view["coalesced"],
            }


_status_registry = PredictionStatusRegistry()


def _parse_request_id(request_id: str | None) -> str | None:
    if request_id is None:
        return None
    try:
        parsed = uuid.UUID(request_id)
    except ValueError as err:
        raise HTTPException(status_code=400, detail="Invalid request identifier.") from err
    if parsed.version != 4:
        raise HTTPException(status_code=400, detail="Invalid request identifier.")
    return str(parsed)


class WorkCoordinator:
    """Bounded executor with exact-request coalescing."""

    def __init__(self, workers: int, queue_size: int):
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="prediction")
        self._capacity = threading.BoundedSemaphore(workers + queue_size)
        self._lock = threading.Lock()
        self._inflight: dict[str, Future] = {}
        self._jobs: dict[str, PredictionJob] = {}

    def submit(self, key: str, function, *args) -> Future:
        with self._lock:
            existing = self._inflight.get(key)
            if existing is not None:
                return existing
            if not self._capacity.acquire(blocking=False):
                raise ServiceBusyError("Prediction queue is full.")
            future = self._executor.submit(function, *args)
            self._inflight[key] = future

        def complete(_future):
            with self._lock:
                self._inflight.pop(key, None)
            self._capacity.release()

        future.add_done_callback(complete)
        return future

    def submit_with_state(self, key: str, function) -> tuple[Future, bool, PredictionJob]:
        with self._lock:
            existing = self._inflight.get(key)
            if existing is not None:
                return existing, True, self._jobs[key]
            if not self._capacity.acquire(blocking=False):
                raise ServiceBusyError("Prediction queue is full.")
            job = PredictionJob(key)
            future = self._executor.submit(function, job)
            self._inflight[key] = future
            self._jobs[key] = job

        def complete(_future):
            with self._lock:
                self._inflight.pop(key, None)
                self._jobs.pop(key, None)
            self._capacity.release()

        future.add_done_callback(complete)
        return future, False, job


_work_coordinator = WorkCoordinator(settings.prediction_workers, settings.prediction_queue_size)
_predict_cache_lock = threading.Lock()
_predict_cache: TTLCache = TTLCache(
    maxsize=settings.cache_max_size,
    ttl=settings.cache_ttl,
)


def _resolve_runtime_commit() -> str:
    commit = (
        settings.deployment_commit
        or os.getenv("GIT_COMMIT")
        or os.getenv("RENDER_GIT_COMMIT")
        or os.getenv("VERCEL_GIT_COMMIT_SHA")
        or os.getenv("GITHUB_SHA")
    )
    if commit and re.fullmatch(r"[0-9a-fA-F]{7,40}", commit):
        return commit[:7].lower()
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()[:7]
    except Exception:
        return "unknown"


_RUNTIME_COMMIT = _resolve_runtime_commit()


def get_runtime_metadata(ticker: str, model_type: str = "lstm") -> dict:
    """Return runtime metadata for a browser-trained or baseline forecast."""
    return {
        "model_version": APP_VERSION,
        "schema_version": SCHEMA_VERSION,
        "git_commit": _RUNTIME_COMMIT,
        "python_version": sys.version.split()[0],
        "window_size": WINDOW_SIZE,
        "feature_count": len(FEATURES),
        "architecture": "browser_trained_compact_lstm",
        "output_width": MAX_FORECAST_DAYS,
        "metric_source": "baseline_definition",
        "browser_training": True,
        "forecast_type": "direction" if "direction" in model_type else "price",
    }


def _fetch_snapshot(ticker: str):
    import api

    fetch_fn = getattr(api, "fetch_data", default_fetch_data)
    try:
        snapshot = fetch_fn(ticker)
        feature_df, prices, dates, metadata = snapshot
        return feature_df.copy(deep=True), prices.copy(), dates.copy(), dict(metadata)
    except MarketTransportError as err:
        raise MarketContextUnavailable(str(err)) from err


def _validated_future_dates(ticker: str, last_date, days: int) -> tuple[list[str], str]:
    import api

    cal_fn = getattr(api, "future_trading_dates", default_future_trading_dates)
    dates, calendar_id = cal_fn(ticker, last_date, days)
    if len(dates) != days:
        raise RuntimeError("Calendar provider returned an incomplete forecast horizon.")
    return dates, calendar_id


def _measure(job: PredictionJob, timing: str, stage: str, function):
    job.set_stage(stage)
    started = time.perf_counter()
    try:
        return function()
    finally:
        job.add_timing(timing, time.perf_counter() - started)


def _price_prediction_pipeline(ticker: str, days: int, job: PredictionJob | None = None) -> dict:
    import api

    job = job or PredictionJob(f"price_{ticker}_{days}")
    job.set_artifact("missing", "not_applicable")
    fetcher = getattr(api, "_fetch_snapshot", _fetch_snapshot)
    feature_df, closing_prices, historical_dates, feature_metadata = _measure(
        job, "market_data", "downloading_market_data", lambda: fetcher(ticker)
    )
    predictions, metrics = persistence_price_forecast(closing_prices, days)
    future_dates, calendar_id = getattr(api, "_validated_future_dates", _validated_future_dates)(
        ticker, historical_dates[-1], days
    )
    runtime = getattr(api, "get_runtime_metadata", get_runtime_metadata)(ticker, "lstm")
    runtime.update(
        {
            "calendar": calendar_id,
            "data_snapshot": feature_metadata,
            "data_quality": feature_metadata.get("market_context", {}),
            "engine": {
                "family": "persistence",
                "role": "server_disabled_fallback",
                "baseline_fallback": True,
            },
        }
    )
    return {
        "ticker": ticker,
        "historical_dates": historical_dates.strftime("%Y-%m-%d").tolist(),
        "historical_prices": np.asarray(closing_prices, dtype=float).flatten().tolist(),
        "future_dates": future_dates,
        "predicted_prices": [float(value) for value in predictions],
        "forecast_days": days,
        "metrics": metrics,
        "metadata": runtime,
    }


def _direction_prediction_pipeline(
    ticker: str, days: int, job: PredictionJob | None = None
) -> dict:
    import api

    job = job or PredictionJob(f"direction_{ticker}_{days}")
    job.set_artifact("missing", "not_applicable")
    fetcher = getattr(api, "_fetch_snapshot", _fetch_snapshot)
    feature_df, closing_prices, historical_dates, feature_metadata = _measure(
        job, "market_data", "downloading_market_data", lambda: fetcher(ticker)
    )
    directions, probabilities, metrics = base_rate_direction_forecast(closing_prices, days)
    future_dates, calendar_id = getattr(api, "_validated_future_dates", _validated_future_dates)(
        ticker, historical_dates[-1], days
    )
    sentiment_fn = getattr(api, "get_financial_sentiment", get_financial_sentiment)
    sentiment_data = sentiment_fn(ticker)
    runtime = getattr(api, "get_runtime_metadata", get_runtime_metadata)(
        ticker, "bilstm_attention_direction"
    )

    runtime.update(
        {
            "calendar": calendar_id,
            "data_snapshot": feature_metadata,
            "data_quality": feature_metadata.get("market_context", {}),
            "engine": {
                "family": "recent_base_rate",
                "role": "server_disabled_fallback",
                "baseline_fallback": True,
            },
        }
    )
    return {
        "ticker": ticker,
        "forecast_days": days,
        "future_dates": future_dates,
        "directions": directions,
        "probabilities": probabilities,
        "attention_weights": [],
        "metrics": metrics,
        "sentiment": sentiment_data.get("sentiment", sentiment_data)
        or {
            "score": 0.0,
            "status": "fallback",
            "provider": "yfinance",
            "method": "vader_financial",
        },
        "metadata": runtime,
    }


def _with_execution_metadata(
    data: dict,
    request_started: float,
    job: PredictionJob | None,
    coalesced: bool,
    response_cache_hit: bool = False,
) -> dict:
    response = copy.deepcopy(data)
    metadata = response.setdefault("metadata", {})
    timings: dict[str, float | None] = {name: None for name in TIMING_FIELDS}
    artifact_state: str | None
    if job is not None and not response_cache_hit:
        snapshot = job.snapshot()
        if not coalesced:
            timings.update(snapshot["timings"])
            started_at = snapshot["started_at"]
            if started_at is not None and started_at >= request_started:
                timings["queue_wait"] = round(started_at - request_started, 4)
        artifact_state = snapshot["artifact_state_before"]
        artifact_action = snapshot["artifact_action"]
    else:
        artifact_state = None
        artifact_action = "not_applicable"
    timings["total"] = round(time.perf_counter() - request_started, 4)
    if response_cache_hit:
        mode = "response_cache_hit"
    elif coalesced:
        mode = "coalesced"
    elif metadata.get("engine", {}).get("baseline_fallback"):
        mode = "baseline_fallback"
    elif artifact_action == "retrained":
        mode = "trained"
    else:
        mode = "artifact_loaded"
    metadata.update(
        {
            "timings_seconds": timings,
            "execution": {"mode": mode, "coalesced": coalesced},
            "artifact_state_before": artifact_state,
            "artifact_action": artifact_action,
        }
    )
    return response


async def _get_fresh_cached_response(cache_key: str, ticker: str, model_type: str) -> dict | None:
    import api

    lock = getattr(api, "_predict_cache_lock", _predict_cache_lock)
    cache = getattr(api, "_predict_cache", _predict_cache)
    with lock:
        return cache.get(cache_key)


async def _await_prediction(
    key: str,
    function,
    ticker: str,
    days: int,
    request_started: float,
    request_id: str | None = None,
) -> dict:
    import api

    coordinator = getattr(api, "_work_coordinator", _work_coordinator)
    registry = getattr(api, "_status_registry", _status_registry)
    predict_cache = getattr(api, "_predict_cache", _predict_cache)
    predict_cache_lock = getattr(api, "_predict_cache_lock", _predict_cache_lock)

    status_attached = False
    view_finished = False

    def run_pipeline(job: PredictionJob) -> dict:
        job.start()
        try:
            result = function(ticker, days, job=job)
        except Exception:
            job.finish(False)
            raise
        job.finish(True)
        return result

    try:
        future, coalesced, job = coordinator.submit_with_state(key, run_pipeline)
        if request_id is not None:
            status_attached = registry.attach(request_id, job, coalesced)

        def cache_completed(completed: Future) -> None:
            try:
                result = completed.result()
            except Exception:
                return
            with predict_cache_lock:
                predict_cache[key] = result

        future.add_done_callback(cache_completed)
        result = await asyncio.wait_for(
            asyncio.shield(asyncio.wrap_future(future)),
            timeout=settings.prediction_timeout_seconds,
        )
        if status_attached and request_id is not None:
            registry.finish_view(request_id, True)
            view_finished = True
        return _with_execution_metadata(result, request_started, job, coalesced)
    except asyncio.CancelledError:
        if status_attached and request_id is not None:
            registry.finish_view(request_id, False)
            view_finished = True
        raise
    except (ServiceBusyError, TrainingCapacityError) as err:
        if status_attached and request_id is not None:
            registry.finish_view(request_id, False)
            view_finished = True
        raise HTTPException(
            status_code=503, detail="Prediction capacity is temporarily full."
        ) from err
    except ArtifactValidationError as err:
        if status_attached and request_id is not None:
            registry.finish_view(request_id, False)
            view_finished = True
        raise HTTPException(
            status_code=503,
            detail=(
                "The server-side learned model is disabled; browser training or a retryable data failure is required. "
                "Compatibility requests return an explicitly labelled baseline when data is available."
            ),
        ) from err
    except MarketContextUnavailable as err:
        if status_attached and request_id is not None:
            registry.finish_view(request_id, False)
            view_finished = True
        raise HTTPException(status_code=503, detail=str(err)) from err
    except UnknownTickerError as err:
        if status_attached and request_id is not None:
            registry.finish_view(request_id, False)
            view_finished = True
        raise HTTPException(status_code=404, detail=str(err)) from err
    except (ValueError, MarketDataUnavailable) as err:
        if status_attached and request_id is not None:
            registry.finish_view(request_id, False)
            view_finished = True
        raise HTTPException(status_code=422, detail=str(err)) from err
    except Exception as err:
        if status_attached and request_id is not None and not view_finished:
            registry.finish_view(request_id, False)
            view_finished = True
        logger.exception("Prediction failed for %s", ticker)
        raise HTTPException(
            status_code=500,
            detail="Forecast generation failed. Please try again later.",
        ) from err


FORECAST_UNAVAILABLE_RESPONSE = {
    503: {
        "description": (
            "The server-side learned model is disabled; browser training or a retryable data failure is required. "
            "Compatibility requests return an explicitly labelled baseline when data is available."
        )
    }
}


@router.get(
    "/api/v1/predict",
    response_model=PriceForecastResponse,
    responses=FORECAST_UNAVAILABLE_RESPONSE,
)
@limiter.limit("5/minute")
async def predict(
    request: Request,
    ticker: str = "AAPL",
    days: int = Query(default=DEFAULT_FORECAST_DAYS, ge=1, le=MAX_FORECAST_DAYS),
    request_id: str | None = Header(default=None, alias="X-Prediction-Request-ID"),
):
    import api

    request_started = time.perf_counter()
    request_id = _parse_request_id(request_id)
    ticker = validate_ticker(ticker)
    predict_cache = getattr(api, "_predict_cache", _predict_cache)
    predict_cache_lock = getattr(api, "_predict_cache_lock", _predict_cache_lock)
    status_registry = getattr(api, "_status_registry", _status_registry)

    cache_key = f"{ticker}_{days}"
    cached = await _get_fresh_cached_response(cache_key, ticker, "lstm")
    if cached:
        if request_id is not None:
            status_registry.cache_hit(request_id, cache_key)
        return _with_execution_metadata(
            cached, request_started, None, False, response_cache_hit=True
        )

    pipeline_fn = getattr(api, "_price_prediction_pipeline", _price_prediction_pipeline)
    data = await _await_prediction(
        cache_key, pipeline_fn, ticker, days, request_started, request_id
    )
    with predict_cache_lock:
        predict_cache[cache_key] = data
    return data


@router.get(
    "/api/v1/predict/direction",
    response_model=DirectionForecastResponse,
    responses=FORECAST_UNAVAILABLE_RESPONSE,
)
@limiter.limit("5/minute")
async def predict_direction_endpoint(
    request: Request,
    ticker: str = "AAPL",
    days: int = Query(default=DEFAULT_FORECAST_DAYS, ge=1, le=MAX_FORECAST_DAYS),
    request_id: str | None = Header(default=None, alias="X-Prediction-Request-ID"),
):
    import api

    request_started = time.perf_counter()
    request_id = _parse_request_id(request_id)
    ticker = validate_ticker(ticker)
    predict_cache = getattr(api, "_predict_cache", _predict_cache)
    predict_cache_lock = getattr(api, "_predict_cache_lock", _predict_cache_lock)
    status_registry = getattr(api, "_status_registry", _status_registry)

    cache_key = f"dir_{ticker}_{days}"
    cached = await _get_fresh_cached_response(cache_key, ticker, "bilstm_attention_direction")
    if cached:
        if request_id is not None:
            status_registry.cache_hit(request_id, cache_key)
        return _with_execution_metadata(
            cached, request_started, None, False, response_cache_hit=True
        )

    pipeline_fn = getattr(api, "_direction_prediction_pipeline", _direction_prediction_pipeline)
    data = await _await_prediction(
        cache_key, pipeline_fn, ticker, days, request_started, request_id
    )
    with predict_cache_lock:
        predict_cache[cache_key] = data
    return data


@router.get("/api/v1/prediction-status/{request_id}", response_model=PredictionStatusResponse)
@limiter.limit("60/minute")
async def prediction_status(request: Request, request_id: str):
    """Return short-lived, in-process progress for a client request ID."""
    import api

    status_registry = getattr(api, "_status_registry", _status_registry)
    try:
        parsed_id = _parse_request_id(request_id)
    except HTTPException:
        return JSONResponse(
            status_code=404,
            content={"detail": "Prediction status is unavailable."},
            headers={"Cache-Control": "no-store"},
        )
    status = status_registry.get(parsed_id or "")
    if status is None:
        return JSONResponse(
            status_code=404,
            content={"detail": "Prediction status is unavailable."},
            headers={"Cache-Control": "no-store"},
        )
    return JSONResponse(content=status, headers={"Cache-Control": "no-store"})
