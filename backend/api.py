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
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Literal

import numpy as np
import yfinance as yf  # type: ignore[import-untyped]
from cachetools import TTLCache
from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded

from calendars import future_trading_dates
from config import (
    APP_VERSION,
    DEFAULT_FORECAST_DAYS,
    FEATURES,
    MAX_FORECAST_DAYS,
    SCHEMA_VERSION,
    WINDOW_SIZE,
    settings,
)
from data_pipeline import fetch_data
from features.market import MarketContextUnavailable
from news_features import get_live_financial_sentiment as get_financial_sentiment
from server_models.api import router as server_forecasts_router
from server_models.response_models import (
    TIMING_FIELDS,
    DirectionForecastResponse,
    PredictionStatusResponse,
    PriceForecastResponse,
)
from services.baselines import base_rate_direction_forecast, persistence_price_forecast
from services.training_data import build_training_snapshot


class ArtifactValidationError(RuntimeError):
    """Compatibility exception retained while server artifacts are disabled."""


class TrainingCapacityError(RuntimeError):
    """Compatibility exception retained for legacy clients."""


def load_fresh_artifact(*_args, **_kwargs):
    """Server-side artifact loading is intentionally disabled in production."""
    raise ArtifactValidationError("Server-side model artifacts are disabled.")


def load_metadata(*_args, **_kwargs) -> dict:
    return {}


def load_metrics(*_args, **_kwargs) -> dict:
    return {}


def load_cross_validation(*_args, **_kwargs) -> dict:
    return {}


def load_validation_results(*_args, **_kwargs) -> list:
    return []


def get_manifest() -> dict:
    return {}


def predict_future(*_args, **_kwargs):
    raise ArtifactValidationError("Server-side model artifacts are disabled.")


def predict_direction(*_args, **_kwargs):
    raise ArtifactValidationError("Server-side model artifacts are disabled.")


# ── Logging (2.7) ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-20s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ──────────────────────────────────────────────────────────────
app = FastAPI(title="StockLSTM API", version=APP_VERSION)


def _deployment_environment() -> str:
    if os.getenv("IS_PULL_REQUEST", "").strip().lower() in {"1", "true", "yes"}:
        return "preview"
    return (
        settings.deployment_environment
        or os.getenv("RENDER_SERVICE_TYPE")
        or os.getenv("VERCEL_ENV")
        or os.getenv("ENVIRONMENT")
        or "local"
    )


def _deployment_commit() -> str | None:
    value = (
        settings.deployment_commit
        or os.getenv("RENDER_GIT_COMMIT")
        or os.getenv("VERCEL_GIT_COMMIT_SHA")
        or os.getenv("GITHUB_SHA")
    )
    if not value or not re.fullmatch(r"[0-9a-fA-F]{7,40}", value):
        return None
    return value[:12].lower()


def _deployment_identity() -> dict[str, Any]:
    environment = _deployment_environment()
    provider = settings.deployment_provider or ("render" if os.getenv("RENDER") else "unknown")
    return {
        "provider": provider,
        "environment": environment,
        "commit": _deployment_commit(),
        "preview": environment.lower() in {"preview", "pr", "pull_request"},
    }


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
_preview_cors_regex = (
    settings.preview_cors_origin_regex
    if _deployment_identity()["preview"] and settings.preview_cors_origin_regex
    else None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_origin_regex=_preview_cors_regex,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["Content-Type", "X-Prediction-Request-ID", "X-Client-Class"],
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


# Forecast response models and telemetry constants live in
# `server_models.response_models` so the server-forecast serving layer can
# return payloads typed by the exact same contracts (see `api.py` imports).


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
    """Return runtime metadata for a browser-trained or baseline forecast."""
    git_commit = os.environ.get("GIT_COMMIT")
    if not git_commit:
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()[:7]
        except Exception:
            git_commit = "unknown"

    return {
        "model_version": APP_VERSION,
        "schema_version": SCHEMA_VERSION,
        "git_commit": git_commit,
        "python_version": sys.version.split()[0],
        "window_size": WINDOW_SIZE,
        "feature_count": len(FEATURES),
        "architecture": "browser_trained_compact_lstm",
        "output_width": MAX_FORECAST_DAYS,
        "metric_source": "baseline_definition",
        "browser_training": True,
        "forecast_type": "direction" if "direction" in model_type else "price",
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
    """Return an explicit server baseline; learned inference runs in the browser."""
    job = job or PredictionJob(f"price_{ticker}_{days}")
    job.set_artifact("missing", "not_applicable")
    feature_df, closing_prices, historical_dates, feature_metadata = _measure(
        job, "market_data", "downloading_market_data", lambda: _fetch_snapshot(ticker)
    )
    predictions, metrics = persistence_price_forecast(closing_prices, days)
    future_dates, calendar_id = _validated_future_dates(ticker, historical_dates[-1], days)
    runtime = get_runtime_metadata(ticker, "lstm")
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
    """Return an explicit server direction baseline; learned inference is client-side."""
    job = job or PredictionJob(f"direction_{ticker}_{days}")
    job.set_artifact("missing", "not_applicable")
    feature_df, closing_prices, historical_dates, feature_metadata = _measure(
        job, "market_data", "downloading_market_data", lambda: _fetch_snapshot(ticker)
    )
    directions, probabilities, metrics = base_rate_direction_forecast(closing_prices, days)
    future_dates, calendar_id = _validated_future_dates(ticker, historical_dates[-1], days)
    sentiment_data = get_financial_sentiment(ticker)
    runtime = get_runtime_metadata(ticker, "bilstm_attention_direction")
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
    """Return a bounded baseline response cache entry.

    Learned model freshness is a browser concern now.  The server cache contains
    only market-data-derived baseline responses and never stores model weights.
    """
    with _predict_cache_lock:
        return _predict_cache.get(cache_key)


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
@app.get("/")
def root():
    """Return discoverable service metadata for the deployment root."""
    return {
        "name": app.title,
        "status": "online",
        "version": APP_VERSION,
        "deployment": _deployment_identity(),
        "docs": "/docs",
        "redoc": "/redoc",
        "openapi": "/openapi.json",
        "health": "/health",
        "readiness": "/ready",
    }


@app.get("/health")
def health():
    """O(1) Liveness probe."""
    return {"status": "ok", "version": APP_VERSION, "deployment": _deployment_identity()}


@app.get("/ready")
def ready():
    """Readiness checks the market-data dependency and, in server-pretrained
    deployments, the server-forecast infrastructure. No browser model disk is
    required for browser training."""
    with _upstream_lock:
        upstream = dict(_upstream_state)
    checked_at = upstream["checked_at_epoch"] or 0
    circuit_blocked = (
        upstream["circuit"] == "open"
        and time.time() - checked_at < settings.upstream_circuit_cooldown_seconds
    )
    if upstream["circuit"] == "open" and not circuit_blocked:
        upstream["circuit"] = "half_open"
    is_ready = not circuit_blocked
    dependencies = {
        "market_data": upstream,
        "model_storage": {
            "required": False,
            "writable": None,
            "detail": "Browser-trained models are trained and cached in each user's browser.",
        },
    }

    # Mode-aware server-forecast readiness: in server_pretrained deployments the
    # serving stack is a hard dependency (no silent browser fallback); in hybrid
    # deployments it is reported but not required.
    if settings.server_forecast_serving_enabled:
        from server_models.api import server_forecast_readiness

        readiness = server_forecast_readiness()
        server_ready = readiness.configured
        dependencies["server_forecasts"] = {
            "configured": readiness.configured,
            "status": "ready" if readiness.configured else readiness.reason,
            "required": settings.training_mode == "server_pretrained",
        }
        if settings.training_mode == "server_pretrained":
            dependencies["model_storage"] = {
                "required": True,
                "writable": readiness.configured,
                "detail": "Server artifacts are stored in the registry and object store.",
            }
            is_ready = is_ready and server_ready
    content = {
        "status": "ready" if is_ready else "degraded",
        "version": APP_VERSION,
        "deployment": _deployment_identity(),
        "dependencies": dependencies,
    }
    return JSONResponse(status_code=200 if is_ready else 503, content=content)


@app.get("/models")
def list_models():
    """Describe browser-trained model availability and, when enabled, the
    server-forecast serving stack. The server_models block reflects the real
    deployment mode instead of always claiming server artifacts are disabled."""
    from server_models.api import MESSAGES, server_forecast_readiness

    server_models = {
        "status": "disabled",
        "reason": "Server forecast serving is disabled.",
        "training_mode": settings.training_mode,
    }
    if settings.server_forecast_serving_enabled:
        readiness = server_forecast_readiness()
        if readiness.configured:
            status = "configured"
            reason = "Server-pretrained forecast serving is configured."
        else:
            status = readiness.reason or "unconfigured"
            reason = MESSAGES.get(status, "Server forecast serving is not fully configured.")
        server_models = {
            "status": status,
            "reason": reason,
            "training_mode": settings.training_mode,
        }
    return {
        "version": APP_VERSION,
        "manifest": {},
        "server_models": server_models,
        "browser_training": {
            "status": "available",
            "model_family": "compact_tfjs_lstm",
            "storage": "indexeddb",
            "cache_scope": "per_user_per_ticker",
            "supported_forecast_types": ["price", "direction"],
        },
        "model_storage": {
            "location": "registry" if settings.server_forecast_serving_enabled else "browser",
            "required": (
                settings.server_forecast_serving_enabled
                and settings.training_mode == "server_pretrained"
            ),
            "detail": (
                "Server artifacts are stored in the registry and object store."
                if settings.server_forecast_serving_enabled
                else "Models are trained and cached per user in the browser."
            ),
        },
        "availability": {
            "price": {
                "status": "browser_available",
                "engine": "compact_tfjs_lstm",
                "tickers": [],
            },
            "direction": {
                "status": "browser_available",
                "engine": "compact_tfjs_lstm",
                "tickers": [],
            },
        },
    }


@app.get("/api/v1/training-data")
@limiter.limit("10/minute")
async def training_data(request: Request, ticker: str = "AAPL"):
    """Return a bounded feature snapshot for browser-side training."""
    ticker = validate_ticker(ticker)
    try:
        return await asyncio.to_thread(build_training_snapshot, ticker)
    except MarketContextUnavailable as err:
        raise HTTPException(status_code=503, detail=str(err)) from err
    except ValueError as err:
        safe_messages = ("Not enough historical data", "Not enough feature rows")
        detail = (
            str(err) if str(err).startswith(safe_messages) else "Invalid input data for training."
        )
        raise HTTPException(status_code=400, detail=detail) from err
    except Exception as err:
        logger.exception("Training-data snapshot failed for %s", ticker)
        raise HTTPException(
            status_code=503,
            detail="Training data is temporarily unavailable. Please try again later.",
        ) from err


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
            "The server-side learned model is disabled; browser training or a retryable data failure is required. "
            "Compatibility requests return an explicitly labelled baseline when data is available."
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
    """Disclose the browser-training contract and any optional offline evidence."""

    ticker = validate_ticker(ticker)
    return {
        "ticker": ticker,
        "forecast_type": forecast_type,
        "engine": {
            "family": "compact_tfjs_lstm",
            "role": "browser_training_available",
            "baseline_fallback": False,
            "artifact_version": None,
        },
        "metrics": {
            "metric_source": "browser_purged_holdout",
            "detail": "Metrics are produced locally in the browser after training.",
        },
        "benchmark": {
            "snapshot": None,
            "validation_method": "browser_purged_holdout",
            "validation_folds": None,
            "metric_source": "browser_purged_holdout",
        },
    }


# Routes are registered unconditionally so the OpenAPI surface is stable across
# deployments. Runtime availability is gated inside the router by
# ``settings.server_forecast_serving_enabled`` (availability endpoint reports
# ``enabled: false`` and forecast requests fall back to browser training).
app.include_router(server_forecasts_router)
