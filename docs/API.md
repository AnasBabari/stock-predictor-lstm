# API Reference

Base URL (local): `http://127.0.0.1:8000`

All endpoints return `application/json`. Rate limits are per source IP. Errors always include `{"detail": "..."}`.

---

## Health

### `GET /health`
Liveness probe.
```json
{"status": "ok", "version": "1.1.0"}
```

### `GET /ready`
Readiness probe.
```json
{"status": "ready", "version": "1.1.0", "dependencies": {"yfinance": true}}
```

### `GET /models`
Manifest of cached model artefacts.
```json
{"version": "1.1.0", "manifest": [{"ticker": "AAPL", "model_type": "lstm", "age_days": 2}]}
```

---

## Prediction

### `GET /api/v1/predict`

LSTM regression price forecast.

**Rate limit**: 5 / min · **Cache**: 300 s

| Parameter | Type | Default | Constraints |
|---|---|---|---|
| `ticker` | string | `AAPL` | 1–12 chars, `[A-Z0-9.\-]` |
| `days` | integer | `7` | 1–30 |

**Example**
```
GET /api/v1/predict?ticker=AAPL&days=7
```
```json
{
  "ticker": "AAPL",
  "historical_dates": ["2025-07-14", "..."],
  "historical_prices": [210.5, "..."],
  "future_dates": ["2025-07-21", "..."],
  "predicted_prices": [212.1, 213.4, "..."],
  "forecast_days": 7,
  "metrics": {"rmse": 4.82, "mae": 3.61, "mape": 1.74, "r2": 0.91, "direction_accuracy": 0.63},
  "metadata": {"model_version": "1.1.0", "architecture": "lstm", "window_size": 60}
}
```

---

### `GET /api/v1/predict/direction`

Bi-LSTM + Attention directional forecast with sentiment.

**Rate limit**: 5 / min · **Cache**: 300 s

| Parameter | Type | Default | Constraints |
|---|---|---|---|
| `ticker` | string | `AAPL` | 1–12 chars, `[A-Z0-9.\-]` |
| `days` | integer | `7` | 1–30 |

**Example**
```
GET /api/v1/predict/direction?ticker=AAPL&days=5
```
```json
{
  "ticker": "AAPL",
  "forecast_days": 5,
  "future_dates": ["2025-07-21", "..."],
  "directions": [1, 1, 0, 1, 0],
  "probabilities": [0.68, 0.71, 0.43, 0.59, 0.48],
  "attention_weights": [{"index": 0, "date": "2025-06-19", "weight": 0.012}, "..."],
  "metrics": {"precision": 0.61, "recall": 0.58, "naive_baseline": 0.54},
  "sentiment": {"score": 0.23, "status": "ok"},
  "metadata": {"architecture": "attention_lstm", "window_size": 60}
}
```

**Notes**
- `directions` — `1` = predicted up, `0` = predicted down.
- `probabilities` — raw sigmoid output; > 0.5 → up.
- `attention_weights` — 60 entries, one per day in the input window.
- `sentiment.score` — VADER compound score in [-1.0, 1.0].

---

## Diagnostics

### `GET /api/v1/diagnostics/{ticker}`

Walk-forward validation results for a trained model.

**Rate limit**: 10 / min

| Parameter | Where | Default | Description |
|---|---|---|---|
| `ticker` | path | — | Ticker symbol |
| `model_type` | query | `bilstm_attention_direction` | `lstm` or `bilstm_attention_direction` |

**Example**
```
GET /api/v1/diagnostics/AAPL
```
```json
{
  "ticker": "AAPL",
  "model_type": "bilstm_attention_direction",
  "cross_validation": {"mean_rmse": 5.21, "mean_direction_accuracy": 0.59, "folds": 5},
  "fold_results": [{"fold": 1, "rmse": 5.80, "direction_accuracy": 0.56}],
  "model_metadata": {"training_duration_seconds": 87.4, "validation_method": "expanding"}
}
```

Returns `404` if the model has not been trained yet.

---

## Discovery

### `GET /api/v1/search`

Ticker autocomplete. **Rate limit**: 30 / min

| Parameter | Type | Description |
|---|---|---|
| `query` | string | Partial ticker or company name |

```
GET /api/v1/search?query=apple
```
```json
{"results": [{"ticker": "AAPL", "name": "Apple Inc.", "type": "EQUITY"}]}
```

---

### `GET /api/v1/info`

Stock fundamentals. **Rate limit**: 20 / min · **Cache**: 3600 s

| Parameter | Type | Default |
|---|---|---|
| `ticker` | string | `AAPL` |

```
GET /api/v1/info?ticker=AAPL
```
```json
{
  "ticker": "AAPL", "name": "Apple Inc.", "marketCap": 3200000000000,
  "peRatio": 31.5, "fiftyTwoWeekHigh": 237.23, "fiftyTwoWeekLow": 164.08,
  "sector": "Technology", "industry": "Consumer Electronics"
}
```
