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
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
import sklearn  # type: ignore[import-untyped]
import tensorflow as tf  # type: ignore[import-untyped]
import yfinance as yf  # type: ignore[import-untyped]
from cachetools import TTLCache
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

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
from data_pipeline import fetch_data, prepare_return_data, preprocess
from features.market import MarketContextUnavailable
from model import (
    TrainingCapacityError,
    get_manifest,
    load_cross_validation,
    load_metadata,
    load_metrics,
    load_or_train,
    load_validation_results,
    predict_direction,
    predict_future,
)
from news_aggregator import get_financial_sentiment

# ── Logging (2.7) ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-20s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ──────────────────────────────────────────────────────────────
app = FastAPI(title="StockLSTM API", version=APP_VERSION)


# Rate limiter (2.5)
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please wait before trying again."},
    )


# CORS (1.1) — explicit origins, no credentials
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
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


class WorkCoordinator:
    """Bounded executor with exact-request coalescing."""

    def __init__(self, workers: int, queue_size: int):
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="prediction")
        self._capacity = threading.BoundedSemaphore(workers + queue_size)
        self._lock = threading.Lock()
        self._inflight: dict[str, Future] = {}

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


_work_coordinator = WorkCoordinator(settings.prediction_workers, settings.prediction_queue_size)
_upstream_lock = threading.Lock()
_upstream_state = {
    "status": "unknown",
    "circuit": "unknown",
    "last_error": None,
    "checked_at_epoch": None,
    "consecutive_failures": 0,
}

VALID_MODEL_TYPES = {"lstm", "bilstm_attention_direction", "attention"}


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
            else "attention_lstm"
            if model_type == "attention"
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


def _price_prediction_pipeline(ticker: str, days: int) -> dict:
    feature_df, closing_prices, historical_dates, feature_metadata = _fetch_snapshot(ticker)
    X_train, X_test, y_train, y_test, scaler, _, _ = preprocess(
        feature_df, forecast_days=MAX_FORECAST_DAYS
    )
    model, model_scaler = load_or_train(
        ticker,
        X_train,
        y_train,
        X_test,
        y_test,
        scaler,
        "lstm",
        feature_df,
        feature_metadata,
    )
    predictions = predict_future(model, feature_df, model_scaler, days=days)
    future_dates, calendar_id = _validated_future_dates(ticker, historical_dates[-1], days)
    if len(predictions) != days:
        raise RuntimeError("Price model returned an incompatible forecast horizon.")

    runtime = get_runtime_metadata(ticker, "lstm")
    runtime.update(
        {
            "calendar": calendar_id,
            "data_snapshot": feature_metadata,
            "data_quality": feature_metadata.get("market_context", {}),
        }
    )
    return {
        "ticker": ticker,
        "historical_dates": historical_dates.strftime("%Y-%m-%d").tolist(),
        "historical_prices": np.asarray(closing_prices, dtype=float).flatten().tolist(),
        "future_dates": future_dates,
        "predicted_prices": [float(value) for value in predictions],
        "forecast_days": days,
        "metrics": load_metrics(ticker, "lstm"),
        "metadata": runtime,
    }


def _direction_prediction_pipeline(ticker: str, days: int) -> dict:
    feature_df, _, historical_dates, feature_metadata = _fetch_snapshot(ticker)
    # Direction artifacts always have a fixed maximum output width. Responses are sliced only
    # after the artifact signature has been validated by load_or_train.
    X_train, X_test, y_train, y_test, scaler, _, _ = prepare_return_data(
        feature_df, forecast_days=MAX_FORECAST_DAYS
    )
    model_type = "bilstm_attention_direction"
    model, model_scaler = load_or_train(
        ticker,
        X_train,
        y_train,
        X_test,
        y_test,
        scaler,
        model_type,
        feature_df,
        feature_metadata,
    )
    directions, probabilities, attention_weights = predict_direction(
        model, feature_df, model_scaler, days=days
    )
    future_dates, calendar_id = _validated_future_dates(ticker, historical_dates[-1], days)
    if not (
        len(directions) == len(probabilities) == len(future_dates) == days
        and len(attention_weights) == WINDOW_SIZE
    ):
        raise RuntimeError("Direction model returned an incompatible response shape.")
    if not all(
        np.isfinite(probability) and 0.0 <= probability <= 1.0 for probability in probabilities
    ):
        raise RuntimeError("Direction model returned invalid probabilities.")

    past_dates = historical_dates[-WINDOW_SIZE:].strftime("%Y-%m-%d").tolist()
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
        }
    )
    return {
        "ticker": ticker,
        "forecast_days": days,
        "future_dates": future_dates,
        "directions": directions,
        "probabilities": probabilities,
        "attention_weights": formatted_attention,
        "metrics": load_metrics(ticker, model_type=model_type),
        "sentiment": sentiment_data.get(
            "sentiment",
            {
                "score": 0.0,
                "status": "fallback",
                "provider": "yfinance",
                "method": "vader_financial",
            },
        ),
        "metadata": runtime,
    }


async def _await_prediction(key: str, function, ticker: str, days: int) -> dict:
    try:
        future = _work_coordinator.submit(key, function, ticker, days)

        def cache_completed(completed: Future) -> None:
            try:
                result = completed.result()
            except Exception:
                return
            with _predict_cache_lock:
                _predict_cache[key] = result

        future.add_done_callback(cache_completed)
        return await asyncio.wait_for(
            asyncio.shield(asyncio.wrap_future(future)),
            timeout=settings.prediction_timeout_seconds,
        )
    except (ServiceBusyError, TrainingCapacityError) as err:
        raise HTTPException(
            status_code=503, detail="Prediction capacity is temporarily full."
        ) from err
    except TimeoutError as err:
        raise HTTPException(
            status_code=503, detail="Prediction timed out; the shared job may still complete."
        ) from err
    except MarketContextUnavailable as err:
        raise HTTPException(status_code=503, detail=str(err)) from err
    except ValueError as err:
        safe_messages = ("Not enough historical data", "Not enough data for")
        detail = (
            str(err) if str(err).startswith(safe_messages) else "Invalid input data for prediction."
        )
        raise HTTPException(status_code=400, detail=detail) from err
    except Exception as err:
        logger.exception("Prediction pipeline failed for %s", ticker)
        raise HTTPException(
            status_code=500, detail="Prediction failed. Please try again later."
        ) from err


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


@app.get("/api/v1/predict")
@limiter.limit("5/minute")
async def predict(
    request: Request,
    ticker: str = "AAPL",
    days: int = Query(default=DEFAULT_FORECAST_DAYS, ge=1, le=MAX_FORECAST_DAYS),
):
    ticker = validate_ticker(ticker)

    cache_key = f"{ticker}_{days}"
    with _predict_cache_lock:
        cached = _predict_cache.get(cache_key)
    if cached:
        return cached

    data = await _await_prediction(cache_key, _price_prediction_pipeline, ticker, days)
    with _predict_cache_lock:
        _predict_cache[cache_key] = data
    return data


@app.get("/api/v1/predict/direction")
@limiter.limit("5/minute")
async def predict_direction_endpoint(
    request: Request,
    ticker: str = "AAPL",
    days: int = Query(default=DEFAULT_FORECAST_DAYS, ge=1, le=MAX_FORECAST_DAYS),
):
    ticker = validate_ticker(ticker)

    cache_key = f"dir_{ticker}_{days}"
    with _predict_cache_lock:
        cached = _predict_cache.get(cache_key)
    if cached:
        return cached

    data = await _await_prediction(cache_key, _direction_prediction_pipeline, ticker, days)
    with _predict_cache_lock:
        _predict_cache[cache_key] = data
    return data


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
