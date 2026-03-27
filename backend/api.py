"""StockLSTM API — FastAPI backend for stock price prediction.

Fixes applied:
    1.1  CORS restricted to explicit origins
    1.4  Ticker input validation (regex, path-traversal safe)
    1.6  Internal errors sanitised — generic messages to client
    2.2  Single yfinance download per predict (dates from pipeline)
    2.3  Bounded TTL cache via cachetools
    2.5  Rate limiting via slowapi
    2.6  /health endpoint
    2.7  Structured logging
"""

import logging
import re
from datetime import timedelta

from cachetools import TTLCache
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
import yfinance as yf  # type: ignore[import-untyped]

from config import DEFAULT_FORECAST_DAYS, MAX_FORECAST_DAYS, settings
from data_pipeline import get_pipeline
from model import load_or_train, predict_future, evaluate_model

# ── Logging (2.7) ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-20s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ──────────────────────────────────────────────────────────────
app = FastAPI(title="StockLSTM API", version="3.0")

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
_predict_cache: TTLCache = TTLCache(
    maxsize=settings.cache_max_size, ttl=settings.cache_ttl,
)
_info_cache: TTLCache = TTLCache(
    maxsize=settings.cache_max_size, ttl=settings.cache_ttl,
)


# ── Helpers ──────────────────────────────────────────────────────────
def validate_ticker(ticker: str) -> str:
    """Sanitise and validate a ticker symbol (1.4)."""
    ticker = ticker.strip().upper()
    if not re.fullmatch(r"[A-Z0-9.\-]{1,12}", ticker):
        raise HTTPException(status_code=400, detail="Invalid ticker symbol.")
    return ticker


# ── Endpoints ────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Liveness probe (2.6)."""
    return {"status": "ok"}


@app.get("/api/v1/search")
@limiter.limit("30/minute")
def search(request: Request, query: str):
    try:
        results = yf.Search(query, max_results=8)
        suggestions = []
        for r in results.quotes:
            if r.get("quoteType") in ("EQUITY", "ETF"):
                suggestions.append({
                    "ticker": r.get("symbol", ""),
                    "name": r.get("longname") or r.get("shortname", ""),
                    "type": r.get("quoteType", ""),
                })
        return {"results": suggestions}
    except Exception:
        logger.exception("Error in /api/v1/search")
        raise HTTPException(
            status_code=500,
            detail="Search failed. Please try again later.",
        )


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
    except Exception:
        logger.exception("Error fetching info for %s", ticker)
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch stock info. Please try again later.",
        )


@app.get("/api/v1/predict")
@limiter.limit("10/minute")
def predict(
    request: Request,
    ticker: str = "AAPL",
    days: int = Query(default=DEFAULT_FORECAST_DAYS, ge=1, le=MAX_FORECAST_DAYS),
):
    ticker = validate_ticker(ticker)

    # Cache lookup
    cache_key = f"{ticker}_{days}"
    cached = _predict_cache.get(cache_key)
    if cached:
        return cached

    try:
        # Single download provides prices + dates (2.2)
        pipeline_data, closing_prices, historical_dates = get_pipeline(ticker)
        X_train, X_test, y_train, y_test, scaler = pipeline_data

        # Model (with per-ticker lock)
        model = load_or_train(ticker, X_train, y_train, X_test, y_test)

        # Metrics (3.6 — MAPE, R², directional accuracy)
        metrics = evaluate_model(model, X_test, y_test, scaler)

        # Direct multi-step predictions (3.2)
        predictions = predict_future(model, closing_prices, scaler, days=days)

        # Dates — reuse the index from the same download (2.2)
        hist_dates = historical_dates.strftime("%Y-%m-%d").tolist()

        future_dates = []
        cur = historical_dates[-1]
        added = 0
        while added < days:
            cur += timedelta(days=1)
            if cur.weekday() < 5:
                future_dates.append(cur.strftime("%Y-%m-%d"))
                added += 1

        historical_prices = [float(p[0]) for p in closing_prices]

        data = {
            "ticker": ticker,
            "historical_dates": hist_dates,
            "historical_prices": historical_prices,
            "future_dates": future_dates,
            "predicted_prices": [float(p) for p in predictions],
            "forecast_days": days,
            "metrics": metrics,
        }

        _predict_cache[cache_key] = data
        return data

    except ValueError as e:
        # Expected errors like "Not enough historical data" are safe to return to client
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Error predicting %s", ticker)
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Please try again later.",
        )