# Architecture

## Overview

StockLSTM is a decoupled full-stack application: a **FastAPI** backend handles data ingestion, feature engineering, model training/inference, and diagnostics; a **React + Vite** frontend renders interactive charts and sends REST requests. GitHub Actions enforces quality gates on every push and publishes Docker images to GHCR on merges to `main`.

---

## Frontend

- **React 18.3 + Vite 5.4** — component-driven SPA, lazy-loaded charts.
- **Chart.js** — line charts with dynamic timeframe selection (1W → 1Y).
- **Features**: ticker autocomplete, watchlist (localStorage), CSV/PNG/ZIP export, forecast-type toggle (price / direction).
- Connects to the backend via `VITE_API_BASE_URL`; defaults to `http://localhost:8000`.

---

## Backend

- **FastAPI 0.115+** — async endpoints, auto-generated OpenAPI docs at `/docs`.
- **Middleware**: explicit CORS origins, per-IP rate limiting (slowapi), input sanitisation (`[A-Z0-9.\-]{1,12}` regex on tickers).
- **Caching**: `cachetools.TTLCache` for predictions (300 s) and stock info (3600 s). Separate cache keys for price and direction endpoints.
- **Error handling**: internal stack traces are logged server-side only; clients receive sanitised messages.

---

## ML Pipeline

### Feature Engineering (22 features per time step)

| Group | Features |
|---|---|
| Base (OHLCV) | Open, High, Low, Close, Volume |
| Technical (9) | SMA_20, EMA_20, RSI_14, MACD, MACD_Signal, BB_Upper, BB_Lower, ATR_14, OBV |
| Market context (4) | SPY_Return_1D, QQQ_Return_1D, VIX_Return_1D, TNX_Return_1D |
| Calendar (4) | Month_Sin, Month_Cos, Day_Sin, Day_Cos |

Input window: **60 days**. Scaling: `MinMaxScaler` fitted exclusively on the training partition (no look-ahead bias).

### Models

**LSTM Regression** — predicts raw closing prices.
- 2× LSTM(64) + Dropout(0.2), `Dense(forecast_days)` linear output.
- Loss: MSE. Target: scaled closing price.

**Bi-LSTM + Attention** — predicts up/down direction with interpretability.
- `Bidirectional(LSTM(64, return_sequences=True))` → self-attention → `Dense(forecast_days, sigmoid)`.
- Loss: binary cross-entropy. Target: log-return direction (1 = up, 0 = down).
- Attention weights (60 values) exported per request for interpretability.

---

## Walk-Forward Validation

```python
ValidationConfig(method="expanding", folds=5, min_train_size=500, horizon=30, seed=42)
```

Five expanding folds. Each fold increases the training set by `horizon` days while keeping the test window fixed. Strategy (`expanding` / `rolling` / `anchored`) is config-only — no code changes needed to compare strategies.

Per-fold results and aggregated cross-validation summaries are persisted to `saved_models/` alongside each model.

---

## Model Caching

Each trained ticker produces six artefacts in `saved_models/`:

```
{ticker}_{model_type}_model.keras       ← Keras weights
{ticker}_{model_type}_scaler.joblib     ← Fitted MinMaxScaler
{ticker}_{model_type}_metrics.json      ← Evaluation metrics
{ticker}_{model_type}_metadata.json     ← Hyper-params, dataset fingerprint, training time
{ticker}_{model_type}_cv.json           ← Aggregate cross-validation summary
{ticker}_{model_type}_validation.json   ← Per-fold predictions and residuals
```

Models are considered stale after `MODEL_MAX_AGE_DAYS` (default 7) and auto-retrain on the next request.

---

## Key Design Decisions

1. **Two models, two endpoints** — Regression (MSE) and classification (BCE) are kept separate so each model is independently cacheable, replaceable, and optimised for its own loss.

2. **Scaler co-persistence** — Saving the fitted scaler alongside the model guarantees that inference preprocessing exactly matches training. Re-fitting on new data would introduce distribution shift.

3. **Configurable validation strategy** — `ValidationConfig` is a Pydantic model. Swapping `expanding` → `rolling` requires only a config change.

4. **NYSE calendar for future dates** — `pandas_market_calendars` generates only trading days for forecast horizons, excluding weekends and holidays.

5. **Look-ahead bias prevention** — Both preprocessing functions (`preprocess` and `prepare_return_data`) fit all transformations on the training partition only before applying them to test data.
