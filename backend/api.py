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

import logging
import os
import re
import subprocess
import sys
import threading
from datetime import timedelta

import numpy as np
import pandas_market_calendars as mcal
import sklearn  # type: ignore[import-untyped]
import tensorflow as tf  # type: ignore[import-untyped]
import yfinance as yf  # type: ignore[import-untyped]
from cachetools import TTLCache
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

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
from data_pipeline import fetch_data, get_pipeline, prepare_return_data, preprocess
from model import (
    evaluate_model,
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
    ttl=settings.cache_ttl,
)

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
        "architecture": "attention_lstm" if model_type == "attention" else "lstm",
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


# ── Endpoints ────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """O(1) Liveness probe."""
    return {"status": "ok", "version": APP_VERSION}


@app.get("/ready")
def ready():
    """O(1) Readiness probe."""
    model_dir_ok = os.path.exists(MODEL_DIR) and os.access(MODEL_DIR, os.R_OK)
    return {
        "status": "ready" if model_dir_ok else "degraded",
        "version": APP_VERSION,
        "dependencies": {
            "yfinance": True,
            "model_storage": model_dir_ok,
        },
    }


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
        return {"results": suggestions}
    except Exception as err:
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

    try:
        pipeline_data, closing_prices, historical_dates, feature_metadata = get_pipeline(ticker)
        X_train, X_test, y_train, y_test, scaler, _, _ = pipeline_data
        feature_df, _, _, _ = fetch_data(ticker)

        # Model load or train with atomic fail-safe (passes feature_df for walk-forward)
        model, model_scaler = await run_in_threadpool(
            load_or_train, ticker, X_train, y_train, X_test, y_test, scaler, "lstm", feature_df
        )

        if model_scaler is not scaler:
            X_train, X_test, y_train, y_test, model_scaler, _, _ = preprocess(
                feature_df, scaler=model_scaler
            )

        metrics = evaluate_model(model, X_test, y_test, model_scaler)
        predictions = predict_future(model, feature_df, model_scaler, days=days)

        hist_dates = historical_dates.strftime("%Y-%m-%d").tolist()

        cur = historical_dates[-1]
        nyse = mcal.get_calendar("NYSE")
        schedule = nyse.schedule(
            start_date=cur + timedelta(days=1), end_date=cur + timedelta(days=days * 3 + 10)
        )
        future_dates = [d.strftime("%Y-%m-%d") for d in schedule.index if d > cur][:days]

        historical_prices = np.asarray(closing_prices, dtype=float).flatten().tolist()

        data = {
            "ticker": ticker,
            "historical_dates": hist_dates,
            "historical_prices": historical_prices,
            "future_dates": future_dates,
            "predicted_prices": [float(p) for p in predictions],
            "forecast_days": days,
            "metrics": metrics,
            "metadata": get_runtime_metadata(ticker, "lstm"),
        }

        with _predict_cache_lock:
            _predict_cache[cache_key] = data
        return data

    except ValueError as err:
        logger.exception("ValueError while predicting %s", ticker)
        safe_msgs = ["Not enough historical data", "Not enough data for"]
        detail = (
            str(err)
            if any(m in str(err) for m in safe_msgs)
            else "Invalid input data for prediction."
        )
        raise HTTPException(
            status_code=400,
            detail=detail,
        ) from err

    except Exception as err:
        logger.exception("Error predicting %s", ticker)
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Please try again later.",
        ) from err


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

    try:
        feature_df, closing_prices, historical_dates, feature_metadata = fetch_data(ticker)

        X_train, X_test, y_train, y_test, scaler, _, _ = prepare_return_data(
            feature_df, forecast_days=days
        )

        model_type = "bilstm_attention_direction"
        model, model_scaler = await run_in_threadpool(
            load_or_train, ticker, X_train, y_train, X_test, y_test, scaler, model_type, feature_df
        )

        if model_scaler is not scaler:
            X_train, X_test, y_train, y_test, model_scaler, _, _ = prepare_return_data(
                feature_df, forecast_days=days, scaler=model_scaler
            )

        directions, probabilities, attention_weights = predict_direction(
            model, feature_df, model_scaler, days=days
        )

        metrics = load_metrics(ticker, model_type=model_type)

        cur = historical_dates[-1]
        nyse = mcal.get_calendar("NYSE")
        schedule = nyse.schedule(
            start_date=cur + timedelta(days=1), end_date=cur + timedelta(days=days * 3 + 10)
        )
        future_dates = [d.strftime("%Y-%m-%d") for d in schedule.index if d > cur][:days]

        past_dates = historical_dates[-WINDOW_SIZE:].strftime("%Y-%m-%d").tolist()
        formatted_attention = [
            {"index": idx, "date": d, "weight": float(w)}
            for idx, (d, w) in enumerate(zip(past_dates, attention_weights, strict=False))
        ]

        sentiment_data = get_financial_sentiment(ticker)

        data = {
            "ticker": ticker,
            "forecast_days": days,
            "future_dates": future_dates,
            "directions": directions,
            "probabilities": probabilities,
            "attention_weights": formatted_attention,
            "metrics": metrics,
            "sentiment": sentiment_data.get(
                "sentiment",
                {
                    "score": 0.0,
                    "status": "fallback",
                    "provider": "yfinance",
                    "method": "vader_financial",
                },
            ),
            "metadata": get_runtime_metadata(ticker, model_type),
        }

        with _predict_cache_lock:
            _predict_cache[cache_key] = data
        return data

    except ValueError as err:
        logger.exception("ValueError while predicting direction for %s", ticker)
        safe_msgs = ["Not enough historical data", "Not enough data for"]
        detail = (
            str(err)
            if any(m in str(err) for m in safe_msgs)
            else "Invalid input data for prediction."
        )
        raise HTTPException(status_code=400, detail=detail) from err
    except Exception as err:
        logger.exception("Error predicting direction %s", ticker)
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Please try again later.",
        ) from err


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
