#type: ignore
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
from datetime import timedelta
import time

from config import HISTORICAL_YEARS, DEFAULT_FORECAST_DAYS, MAX_FORECAST_DAYS
from data_pipeline import get_pipeline
from model import load_or_train, predict_future, evaluate_model

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="StockLSTM API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory caches ────────────────────────────────────────────────────────
_predict_cache: dict = {}
_info_cache: dict = {}
CACHE_TTL = 300  # seconds


# ── Endpoints ───────────────────────────────────────────────────────────────
@app.get("/api/v1/search")
def search(query: str):
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/info")
def stock_info(ticker: str = "AAPL"):
    """Return rich metadata for a ticker (market cap, 52-week range, etc.)."""
    ticker = ticker.upper()

    cached = _info_cache.get(ticker)
    if cached and (time.time() - cached["ts"]) < CACHE_TTL:
        return cached["data"]

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
        _info_cache[ticker] = {"data": data, "ts": time.time()}
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/predict")
def predict(
    ticker: str = "AAPL",
    days: int = Query(default=DEFAULT_FORECAST_DAYS, ge=1, le=MAX_FORECAST_DAYS),
):
    ticker = ticker.upper()

    # Cache lookup
    cache_key = f"{ticker}_{days}"
    cached = _predict_cache.get(cache_key)
    if cached and (time.time() - cached["ts"]) < CACHE_TTL:
        return cached["data"]

    try:
        # Pipeline
        pipeline_data, closing_prices = get_pipeline(ticker)
        X_train, X_test, y_train, y_test, scaler = pipeline_data

        # Model
        model = load_or_train(ticker, X_train, y_train)

        # Metrics
        metrics = evaluate_model(model, X_test, y_test, scaler)

        # Predictions
        predictions = predict_future(model, closing_prices, scaler, days=days)

        # Dates
        raw = yf.download(ticker, period=f"{HISTORICAL_YEARS}y", progress=False).dropna()
        historical_dates = raw.index.strftime("%Y-%m-%d").tolist()

        future_dates = []
        cur = raw.index[-1]
        added = 0
        while added < days:
            cur += timedelta(days=1)
            if cur.weekday() < 5:
                future_dates.append(cur.strftime("%Y-%m-%d"))
                added += 1

        historical_prices = [float(p[0]) for p in closing_prices]

        data = {
            "ticker": ticker,
            "historical_dates": historical_dates,
            "historical_prices": historical_prices,
            "future_dates": future_dates,
            "predicted_prices": [float(p) for p in predictions],
            "forecast_days": days,
            "metrics": metrics,
        }

        _predict_cache[cache_key] = {"data": data, "ts": time.time()}
        return data

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))