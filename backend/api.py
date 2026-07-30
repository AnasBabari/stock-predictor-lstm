"""StockLSTM API — FastAPI backend for stock price prediction.

Fixes applied:
    1.1 CORS restricted to explicit origins
    1.4 Ticker input validation (regex, path-traversal safe)
    1.6 Internal errors sanitised — generic messages to client
    2.2 Single yfinance download per predict (dates from pipeline)
    2.3 Bounded TTL cache via cachetools
    2.5 Rate limiting via slowapi
    2.6 /health endpoint
    2.7 Structured logging
"""

import asyncio
import copy
import ipaddress
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Annotated, Any, Literal

import numpy as np
import sklearn  # type: ignore[import-untyped]
import tensorflow as tf  # type: ignore[import-untyped]
import yfinance as yf  # type: ignore[import-untyped]
from cachetools import TTLCache
from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded

from calendars import future_trading_dates
from config import (
    APP_VERSION,
    DEFAULT_FORECAST_DAYS,
    FEATURES,
    MAX_FORECAST_DAYS,
    MODEL_DIR,
    SCHEMA_VERSION,
    WINDOW_SIZE,
    settings,
)
from data_pipeline import fetch_data
from features.market import MarketContextUnavailable
from model import (
    ArtifactValidationError,
    TrainingCapacityError,
    get_manifest,
    load_cross_validation,
    load_fresh_artifact,
    load_metadata,
    load_metrics,
    load_validation_results,
    predict_direction,
    predict_future,
)
from news_features import get_live_financial_sentiment as get_financial_sentiment
from services.baselines import base_rate_direction_forecast, persistence_price_forecast

# ── Logging (2.7) ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-20s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ──────────────────────────────────────────────────────────────
app = FastAPI(title="StockLSTM API", version=APP_VERSION)


_trusted_proxy_ips = frozenset(settings.trusted_proxy_ips)


def _normalise_ip(value: str) -> str | None:
    try:
        return str(ipaddress.ip_address(value.strip()))
    except ValueError:
        return None


def rate_limit_identity(request: Request) -> str:
    """Trust forwarding data only when the direct peer is explicitly configured."""
    peer = request.client.host if request.client is not None else "unknown"
    normalised_peer = _normalise_ip(peer)
    if normalised_peer is None or normalised_peer not in _trusted_proxy_ips:
        return normalised_peer or peer

    forwarded = request.headers.get("x-forwarded-for")
    if not forwarded:
        return normalised_peer
    hops = [_normalise_ip(value) for value in forwarded.split(",")]
    if any(hop is None for hop in hops):
        return normalised_peer
    for hop in reversed(hops):
        if hop is not None and hop not in _trusted_proxy_ips:
            return hop
    return normalised_peer


# Rate limiter (2.5)
limiter = Limiter(key_func=rate_limit_identity)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    headers = (
        {"Cache-Control": "no-store"}
        if request.url.path.startswith("/api/v1/prediction-status/")
        else None
    )
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please wait before trying again."},
        headers=headers,
    )


# CORS (1.1) — explicit origins, no credentials
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["Content-Type", "X-Prediction-Request-ID"],
)

# ── Bounded caches (2.3) ────────────────────────────────────────────
_predict_cache_lock = threading.Lock()
_predict_cache: TTLCache = TTLCache(
    maxsize=settings.cache_max_size,
    ttl=settings.cache_ttl,
)
_info_cache: TTLCache = TTLCache(
    maxsize=settings.cache_max_size,
    ttl=settings.info_cache_ttl,
)
_info_cache_lock = threading.Lock()


class ServiceBusyError(RuntimeError):
    """The bounded prediction executor has no queue capacity."""


TIMING_FIELDS = (
    "queue_wait",
    "market_data",
    "feature_preparation",
    "artifact_load_validation",
    "training",
    "inference",
    "total",
)
EXECUTION_MODES = (
    "response_cache_hit",
    "artifact_loaded",
    "baseline_fallback",
    "trained",
    "coalesced",
)
ARTIFACT_STATES = ("fresh", "missing", "stale", "incompatible")
ARTIFACT_ACTIONS = ("loaded", "retrained", "not_applicable")
STATUS_STAGES = (
    "queued",
    "downloading_market_data",
    "preparing_features",
    "checking_artifact",
    "training",
    "generating_forecast",
    "completed",
    "failed",
)


class PredictionStatusResponse(BaseModel):
    status: Literal["queued", "running", "completed", "failed"]
    stage: Literal[
        "queued",
        "downloading_market_data",
        "preparing_features",
        "checking_artifact",
        "training",
        "generating_forecast",
        "completed",
        "failed",
    ]
    coalesced: bool


NonNegativeSeconds = Annotated[float, Field(ge=0)]
Probability = Annotated[float, Field(ge=0, le=1)]


class PredictionTimings(BaseModel):
    queue_wait: NonNegativeSeconds | None
    market_data: NonNegativeSeconds | None
    feature_preparation: NonNegativeSeconds | None
    artifact_load_validation: NonNegativeSeconds | None
    training: NonNegativeSeconds | None
    inference: NonNegativeSeconds | None
    total: NonNegativeSeconds


class PredictionExecution(BaseModel):
    mode: Literal[
        "response_cache_hit",
        "artifact_loaded",
        "baseline_fallback",
        "trained",
        "coalesced",
    ]
    coalesced: bool


class ForecastMetadata(BaseModel):
    """Stable telemetry contract plus permissive legacy runtime diagnostics."""

    model_config = ConfigDict(extra="allow")

    timings_seconds: PredictionTimings
    execution: PredictionExecution
    artifact_state_before: Literal["fresh", "missing", "stale", "incompatible"] | None
    artifact_action: Literal["loaded", "retrained", "not_applicable"]


class ForecastResponse(BaseModel):
    """Stable forecast envelope; metrics remain model-specific legacy data."""

    model_config = ConfigDict(extra="allow")

    ticker: str
    forecast_days: int
    future_dates: list[str]
    metrics: dict[str, Any]
    metadata: ForecastMetadata


class PriceForecastResponse(ForecastResponse):
    historical_dates: list[str]
    historical_prices: list[float]
    predicted_prices: list[float]


class AttentionWeight(BaseModel):
    index: int
    date: str
    weight: float


class DirectionForecastResponse(ForecastResponse):
    directions: list[Literal["Up", "Down"]]
    probabilities: list[Probability]
    attention_weights: list[AttentionWeight]
    sentiment: dict[str, Any]


class ModelPerformanceResponse(BaseModel):
    ticker: str
    forecast_type: Literal["price", "direction"]
    engine: dict[str, Any]
    metrics: dict[str, Any]
    benchmark: dict[str, Any]


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
                "expires_at": time.monotonic() + self.ttl_seconds if terminal else float("inf"),
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
_upstream_lock = threading.Lock()
_upstream_state = {
    "status": "unknown",
    "circuit": "unknown",
    "last_error": None,
    "checked_at_epoch": None,
    "consecutive_failures": 0,
}

VALID_MODEL_TYPES = {
    "lstm",
    "gru",
    "attention",
    "bilstm_attention_regression",
    "bilstm_attention_direction",
}


# ── Helpers ──────────────────────────────────────────────────────────
def get_runtime_metadata(ticker: str, model_type: str = "lstm") -> dict:
    """Gather dynamic runtime environment and model metadata."""
    git_commit = os.environ.get("GIT_COMMIT")
    if not git_commit:
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()[:7]
        except Exception:
            git_commit = "unknown"

    saved_meta = load_metadata(ticker, model_type)

    return {
        "model_version": APP_VERSION,
        "schema_version": saved_meta.get("schema_version", SCHEMA_VERSION),
        "git_commit": git_commit,
        "tensorflow_version": tf.__version__,
        "python_version": sys.version.split()[0],
        "sklearn_version": sklearn.__version__,
        "window_size": WINDOW_SIZE,
        "feature_count": saved_meta.get("feature_count", len(FEATURES)),
        "architecture": (
            "bidirectional_lstm_with_attention"
            if model_type == "bilstm_attention_direction"
            else "bidirectional_lstm_with_attention_regression"
            if model_type == "bilstm_attention_regression"
            else "attention_lstm"
            if model_type == "attention"
            else "gru"
            if model_type == "gru"
            else "lstm"
        ),
        "output_width": saved_meta.get("output_width"),
        "metric_source": saved_meta.get("metric_source", "unavailable"),
        "seed": saved_meta.get("seed"),
        "deterministic": saved_meta.get("deterministic"),
    }


def validate_ticker(ticker: str) -> str:
    """Sanitise and validate a ticker symbol (1.4)."""
    ticker = ticker.strip().upper()
    if not re.fullmatch(r"[A-Z0-9.\-]{1,12}", ticker):
        raise HTTPException(status_code=400, detail="Invalid ticker symbol.")
    return ticker


def validate_model_type(model_type: str) -> str:
    """Validate model type parameter to prevent path traversal."""
    model_type = model_type.strip().lower()
    if model_type not in VALID_MODEL_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model type. Must be one of: {sorted(VALID_MODEL_TYPES)}",
        )
    return model_type


def _record_upstream(status: str, error: str | None = None) -> None:
    with _upstream_lock:
        failures = (
            min(int(_upstream_state["consecutive_failures"] or 0) + 1, 1000)
            if status == "unavailable"
            else 0
        )
        _upstream_state.update(
            {
                "status": status,
                "circuit": "open" if status == "unavailable" else "closed",
                "last_error": error,
                "checked_at_epoch": int(time.time()),
                "consecutive_failures": failures,
            }
        )


def _fetch_snapshot(ticker: str):
    with _upstream_lock:
        checked_at = int(_upstream_state["checked_at_epoch"] or 0)
        cooldown_remaining = settings.upstream_circuit_cooldown_seconds - (time.time() - checked_at)
        if _upstream_state["circuit"] == "open" and cooldown_remaining > 0:
            raise MarketContextUnavailable("Market data circuit is temporarily open; retry later.")
        if _upstream_state["circuit"] == "open":
            _upstream_state["circuit"] = "half_open"
        elif _upstream_state["circuit"] == "half_open":
            raise MarketContextUnavailable("Market data circuit recovery probe is in progress.")
    try:
        snapshot = fetch_data(ticker)
        _record_upstream("available")
        feature_df, prices, dates, metadata = snapshot
        return feature_df.copy(deep=True), prices.copy(), dates.copy(), dict(metadata)
    except Exception as err:
        _record_upstream("unavailable", type(err).__name__)
        raise


def _validated_future_dates(ticker: str, last_date, days: int) -> tuple[list[str], str]:
    dates, calendar_id = future_trading_dates(ticker, last_date, days)
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
    job = job or PredictionJob(f"price_{ticker}_{days}")
    model = model_scaler = None
    try:
        model, model_scaler = _measure(
            job,
            "artifact_load_validation",
            "checking_artifact",
            lambda: load_fresh_artifact(ticker, "lstm", MAX_FORECAST_DAYS),
        )
        job.set_artifact("fresh", "loaded")
    except ArtifactValidationError:
        job.set_artifact("missing", "not_applicable")
    feature_df, closing_prices, historical_dates, feature_metadata = _measure(
        job, "market_data", "downloading_market_data", lambda: _fetch_snapshot(ticker)
    )
    if model is None:
        predictions, metrics = persistence_price_forecast(closing_prices, days)
        future_dates, calendar_id = _validated_future_dates(ticker, historical_dates[-1], days)
        engine = {
            "family": "persistence",
            "role": "baseline_fallback",
            "baseline_fallback": True,
        }
    else:
        predictions, future_dates, calendar_id = _measure(
            job,
            "inference",
            "generating_forecast",
            lambda: (
                predict_future(model, feature_df, model_scaler, days=days),
                *_validated_future_dates(ticker, historical_dates[-1], days),
            ),
        )
        metrics = load_metrics(ticker, "lstm")
        engine = {
            "family": "lstm",
            "role": "learned_candidate",
            "baseline_fallback": False,
        }
    if len(predictions) != days:
        raise RuntimeError("Price model returned an incompatible forecast horizon.")

    runtime = get_runtime_metadata(ticker, "lstm")
    runtime.update(
        {
            "calendar": calendar_id,
            "data_snapshot": feature_metadata,
            "data_quality": feature_metadata.get("market_context", {}),
            "engine": engine,
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
    job = job or PredictionJob(f"direction_{ticker}_{days}")
    model_type = "bilstm_attention_direction"
    model = model_scaler = None
    try:
        model, model_scaler = _measure(
            job,
            "artifact_load_validation",
            "checking_artifact",
            lambda: load_fresh_artifact(ticker, model_type, MAX_FORECAST_DAYS),
        )
        job.set_artifact("fresh", "loaded")
    except ArtifactValidationError:
        job.set_artifact("missing", "not_applicable")
    feature_df, closing_prices, historical_dates, feature_metadata = _measure(
        job, "market_data", "downloading_market_data", lambda: _fetch_snapshot(ticker)
    )
    if model is None:
        directions, probabilities, metrics = base_rate_direction_forecast(closing_prices, days)
        attention_weights = []
        future_dates, calendar_id = _validated_future_dates(ticker, historical_dates[-1], days)
        engine = {
            "family": "recent_base_rate",
            "role": "baseline_fallback",
            "baseline_fallback": True,
        }
    else:
        directions, probabilities, attention_weights, future_dates, calendar_id = _measure(
            job,
            "inference",
            "generating_forecast",
            lambda: (
                *predict_direction(model, feature_df, model_scaler, days=days),
                *_validated_future_dates(ticker, historical_dates[-1], days),
            ),
        )
        metrics = load_metrics(ticker, model_type=model_type)
        engine = {
            "family": model_type,
            "role": "learned_candidate",
            "baseline_fallback": False,
        }
    if not (len(directions) == len(probabilities) == len(future_dates) == days) or (
        model is not None and len(attention_weights) != WINDOW_SIZE
    ):
        raise RuntimeError("Direction model returned an incompatible response shape.")
    if not all(
        np.isfinite(probability) and 0.0 <= probability <= 1.0 for probability in probabilities
    ):
        raise RuntimeError("Direction model returned invalid probabilities.")

    past_dates = (
        historical_dates[-len(attention_weights) :].strftime("%Y-%m-%d").tolist()
        if attention_weights
        else []
    )
    formatted_attention = [
        {"index": index, "date": date, "weight": float(weight)}
        for index, (date, weight) in enumerate(zip(past_dates, attention_weights, strict=True))
    ]
    sentiment_data = get_financial_sentiment(ticker)
    runtime = get_runtime_metadata(ticker, model_type)
    runtime.update(
        {
            "calendar": calendar_id,
            "data_snapshot": feature_metadata,
            "data_quality": feature_metadata.get("market_context", {}),
            "engine": engine,
        }
    )
    return {
        "ticker": ticker,
        "forecast_days": days,
        "future_dates": future_dates,
        "directions": directions,
        "probabilities": probabilities,
        "attention_weights": formatted_attention,
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
    """Return a cached response only if its underlying serving artifact is still fresh."""
    with _predict_cache_lock:
        cached = _predict_cache.get(cache_key)
    if cached is None:
        return None
    if cached.get("metadata", {}).get("engine", {}).get("baseline_fallback"):
        return cached

    try:
        await asyncio.to_thread(load_fresh_artifact, ticker, model_type, MAX_FORECAST_DAYS)
    except ArtifactValidationError as err:
        with _predict_cache_lock:
            _predict_cache.pop(cache_key, None)
        logger.info(
            "Evicted response cache for unavailable forecast artifact %s/%s", ticker, model_type
        )
        raise HTTPException(
            status_code=503,
            detail="Forecast model is not currently available for this ticker.",
        ) from err
    return cached


async def _await_prediction(
    key: str,
    function,
    ticker: str,
    days: int,
    request_started: float,
    request_id: str | None = None,
) -> dict:
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
        future, coalesced, job = _work_coordinator.submit_with_state(key, run_pipeline)
        if request_id is not None:
            status_attached = _status_registry.attach(request_id, job, coalesced)

        def cache_completed(completed: Future) -> None:
            try:
                result = completed.result()
            except Exception:
                return
            with _predict_cache_lock:
                _predict_cache[key] = result

        future.add_done_callback(cache_completed)
        result = await asyncio.wait_for(
            asyncio.shield(asyncio.wrap_future(future)),
            timeout=settings.prediction_timeout_seconds,
        )
        if status_attached and request_id is not None:
            _status_registry.finish_view(request_id, True)
            view_finished = True
        return _with_execution_metadata(result, request_started, job, coalesced)
    except asyncio.CancelledError:
        if status_attached and request_id is not None:
            _status_registry.finish_view(request_id, False)
            view_finished = True
        raise
    except (ServiceBusyError, TrainingCapacityError) as err:
        if status_attached and request_id is not None:
            _status_registry.finish_view(request_id, False)
            view_finished = True
        raise HTTPException(
            status_code=503, detail="Prediction capacity is temporarily full."
        ) from err
    except ArtifactValidationError as err:
        if status_attached and request_id is not None:
            _status_registry.finish_view(request_id, False)
            view_finished = True
        logger.info("Fresh forecast artifact unavailable for %s", ticker)
        raise HTTPException(
            status_code=503,
            detail="Forecast model is not currently available for this ticker.",
        ) from err
    except TimeoutError as err:
        if status_attached and request_id is not None:
            _status_registry.finish_view(request_id, False)
            view_finished = True
        raise HTTPException(
            status_code=503, detail="Prediction timed out; the shared job may still complete."
        ) from err
    except MarketContextUnavailable as err:
        if status_attached and request_id is not None:
            _status_registry.finish_view(request_id, False)
            view_finished = True
        raise HTTPException(status_code=503, detail=str(err)) from err
    except ValueError as err:
        if status_attached and request_id is not None:
            _status_registry.finish_view(request_id, False)
            view_finished = True
        safe_messages = ("Not enough historical data", "Not enough data for")
        detail = (
            str(err) if str(err).startswith(safe_messages) else "Invalid input data for prediction."
        )
        raise HTTPException(status_code=400, detail=detail) from err
    except Exception as err:
        if status_attached and request_id is not None:
            _status_registry.finish_view(request_id, False)
            view_finished = True
        logger.exception("Prediction pipeline failed for %s", ticker)
        raise HTTPException(
            status_code=500, detail="Prediction failed. Please try again later."
        ) from err
    finally:
        if status_attached and request_id is not None and not view_finished:
            _status_registry.finish_view(request_id, False)


# ── Endpoints ────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """O(1) Liveness probe."""
    return {"status": "ok", "version": APP_VERSION}


@app.get("/ready")
def ready():
    """Readiness checks storage admission and bounded upstream circuit state."""
    model_dir_ok = False
    free_mb = 0
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=MODEL_DIR, prefix=".ready-", delete=True):
            pass
        free_mb = shutil.disk_usage(MODEL_DIR).free // (1024 * 1024)
        model_dir_ok = free_mb >= settings.model_min_free_mb
    except OSError:
        logger.warning("Model storage readiness check failed", exc_info=True)
    with _upstream_lock:
        upstream = dict(_upstream_state)
    checked_at = upstream["checked_at_epoch"] or 0
    circuit_blocked = (
        upstream["circuit"] == "open"
        and time.time() - checked_at < settings.upstream_circuit_cooldown_seconds
    )
    if upstream["circuit"] == "open" and not circuit_blocked:
        upstream["circuit"] = "half_open"
    is_ready = model_dir_ok and not circuit_blocked
    content = {
        "status": "ready" if is_ready else "degraded",
        "version": APP_VERSION,
        "dependencies": {
            "market_data": upstream,
            "model_storage": {"writable": model_dir_ok, "free_mb": free_mb},
        },
    }
    return JSONResponse(status_code=200 if is_ready else 503, content=content)


@app.get("/models")
def list_models():
    """Return manifest of trained/cached models."""
    return {"version": APP_VERSION, "manifest": get_manifest()}


@app.get("/api/v1/search")
@limiter.limit("30/minute")
def search(
    request: Request,
    query: str = Query(..., min_length=1, max_length=100),
):
    exact_symbol = query.strip().upper()
    fallback = []
    if re.fullmatch(r"[A-Z0-9.\-]{1,12}", exact_symbol):
        fallback.append({"ticker": exact_symbol, "name": exact_symbol, "type": "SYMBOL"})
    if fallback and query.strip() == exact_symbol:
        return {"results": fallback}
    try:
        results = yf.Search(query, max_results=8)
        suggestions = []
        for r in results.quotes:
            if r.get("quoteType") in ("EQUITY", "ETF"):
                suggestions.append(
                    {
                        "ticker": r.get("symbol", ""),
                        "name": r.get("longname") or r.get("shortname", ""),
                        "type": r.get("quoteType", ""),
                    }
                )
        seen = {item["ticker"] for item in suggestions}
        return {"results": suggestions + [item for item in fallback if item["ticker"] not in seen]}
    except Exception as err:
        if fallback:
            logger.warning("Autocomplete upstream unavailable; returning exact symbol fallback")
            return {"results": fallback, "degraded": True}
        logger.exception("Error in /api/v1/search")
        raise HTTPException(
            status_code=500,
            detail="Search failed. Please try again later.",
        ) from err


@app.get("/api/v1/info")
@limiter.limit("20/minute")
def stock_info(request: Request, ticker: str = "AAPL"):
    """Return rich metadata for a ticker."""
    ticker = validate_ticker(ticker)

    with _info_cache_lock:
        cached = _info_cache.get(ticker)
    if cached:
        return cached

    try:
        info = yf.Ticker(ticker).info
        data = {
            "ticker": ticker,
            "name": info.get("longName") or info.get("shortName", ticker),
            "exchange": info.get("exchange", "—"),
            "currency": info.get("currency", "USD"),
            "marketCap": info.get("marketCap"),
            "peRatio": info.get("trailingPE"),
            "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
            "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
            "avgVolume": info.get("averageVolume"),
            "dayHigh": info.get("dayHigh"),
            "dayLow": info.get("dayLow"),
            "previousClose": info.get("previousClose"),
            "sector": info.get("sector", "—"),
            "industry": info.get("industry", "—"),
        }
        with _info_cache_lock:
            _info_cache[ticker] = data
        return data
    except Exception as err:
        logger.exception("Error fetching info for %s", ticker)
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch stock info. Please try again later.",
        ) from err


FORECAST_UNAVAILABLE_RESPONSE = {
    503: {
        "description": (
            "No fresh, pre-trained model artifact is available for the requested ticker, "
            "or prediction capacity is temporarily full."
        )
    }
}


@app.get(
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
    request_started = time.perf_counter()
    request_id = _parse_request_id(request_id)
    ticker = validate_ticker(ticker)

    cache_key = f"{ticker}_{days}"
    cached = await _get_fresh_cached_response(cache_key, ticker, "lstm")
    if cached:
        if request_id is not None:
            _status_registry.cache_hit(request_id, cache_key)
        return _with_execution_metadata(
            cached, request_started, None, False, response_cache_hit=True
        )

    data = await _await_prediction(
        cache_key, _price_prediction_pipeline, ticker, days, request_started, request_id
    )
    with _predict_cache_lock:
        _predict_cache[cache_key] = data
    return data


@app.get(
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
    request_started = time.perf_counter()
    request_id = _parse_request_id(request_id)
    ticker = validate_ticker(ticker)

    cache_key = f"dir_{ticker}_{days}"
    cached = await _get_fresh_cached_response(cache_key, ticker, "bilstm_attention_direction")
    if cached:
        if request_id is not None:
            _status_registry.cache_hit(request_id, cache_key)
        return _with_execution_metadata(
            cached, request_started, None, False, response_cache_hit=True
        )

    data = await _await_prediction(
        cache_key, _direction_prediction_pipeline, ticker, days, request_started, request_id
    )
    with _predict_cache_lock:
        _predict_cache[cache_key] = data
    return data


@app.get("/api/v1/prediction-status/{request_id}", response_model=PredictionStatusResponse)
@limiter.limit("60/minute")
def prediction_status(request: Request, request_id: str):
    """Return short-lived, in-process progress for a client request ID."""
    try:
        parsed_id = _parse_request_id(request_id)
    except HTTPException:
        return JSONResponse(
            status_code=404,
            content={"detail": "Prediction status is unavailable."},
            headers={"Cache-Control": "no-store"},
        )
    status = _status_registry.get(parsed_id or "")
    if status is None:
        return JSONResponse(
            status_code=404,
            content={"detail": "Prediction status is unavailable."},
            headers={"Cache-Control": "no-store"},
        )
    return JSONResponse(content=status, headers={"Cache-Control": "no-store"})


@app.get("/api/v1/diagnostics/{ticker}")
@limiter.limit("10/minute")
async def diagnostics(
    request: Request,
    ticker: str,
    model_type: str = Query(default="bilstm_attention_direction"),
):
    """
    Return walk-forward validation diagnostics for a trained ticker model.

    Includes per-fold residuals, actuals, predictions, and cross-validation summary.
    """
    ticker = validate_ticker(ticker)
    model_type = validate_model_type(model_type)

    try:
        cv_summary = load_cross_validation(ticker, model_type)
        fold_results = load_validation_results(ticker, model_type)
        metadata = load_metadata(ticker, model_type)

        if not cv_summary and not fold_results:
            raise HTTPException(
                status_code=404,
                detail=f"No diagnostics found for {ticker}/{model_type}. Train the model first.",
            )

        return {
            "ticker": ticker,
            "model_type": model_type,
            "cross_validation": cv_summary,
            "fold_results": fold_results,
            "model_metadata": {
                "schema_version": metadata.get("schema_version"),
                "feature_count": metadata.get("feature_count"),
                "dataset_fingerprint": metadata.get("dataset_fingerprint"),
                "training_duration_seconds": metadata.get("training_duration_seconds"),
                "created_at": metadata.get("created_at"),
                "validation_method": metadata.get("validation_method"),
                "validation_folds": metadata.get("validation_folds"),
            },
        }

    except HTTPException:
        raise
    except Exception as err:
        logger.exception("Error fetching diagnostics for %s", ticker)
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch diagnostics. Please try again later.",
        ) from err


@app.get(
    "/api/v1/model-performance/{ticker}",
    response_model=ModelPerformanceResponse,
)
@limiter.limit("20/minute")
def model_performance(
    request: Request,
    ticker: str,
    forecast_type: Literal["price", "direction"] = Query(default="price"),
):
    """Disclose evidence for the engine selected by public serving."""

    ticker = validate_ticker(ticker)
    model_type = "lstm" if forecast_type == "price" else "bilstm_attention_direction"
    try:
        load_fresh_artifact(ticker, model_type, MAX_FORECAST_DAYS)
        metadata = load_metadata(ticker, model_type)
        metrics = load_metrics(ticker, model_type)
        engine = {
            "family": model_type,
            "role": "learned_candidate",
            "baseline_fallback": False,
            "artifact_version": metadata.get("version_id"),
        }
        benchmark = {
            "snapshot": metadata.get("data_snapshot", {}),
            "validation_method": metadata.get("validation_method"),
            "validation_folds": metadata.get("validation_folds"),
            "metric_source": metadata.get("metric_source"),
        }
    except ArtifactValidationError:
        family = "persistence" if forecast_type == "price" else "recent_base_rate"
        engine = {
            "family": family,
            "role": "baseline_fallback",
            "baseline_fallback": True,
            "artifact_version": None,
        }
        metrics = {
            "metric_source": "baseline_definition",
            "relative_mae": 1.0 if forecast_type == "price" else None,
            "relative_rmse": 1.0 if forecast_type == "price" else None,
            "detail": "No learned candidate currently has qualifying fresh evidence.",
        }
        benchmark = {
            "snapshot": None,
            "validation_method": None,
            "validation_folds": None,
            "metric_source": "baseline_definition",
        }
    return {
        "ticker": ticker,
        "forecast_type": forecast_type,
        "engine": engine,
        "metrics": metrics,
        "benchmark": benchmark,
    }
