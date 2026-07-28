# API reference

Local backend: `http://127.0.0.1:8000`. Interactive OpenAPI: `/docs`; schema: `/openapi.json`.

All errors use `{"detail":"..."}`. `400` means invalid/insufficient instrument data, `422` means query-schema failure, `429` means per-IP rate limiting, and `503` means bounded capacity, timeout, readiness failure, or retryable benchmark unavailability.

## Probes and discovery

- `GET /health`: process liveness only.
- `GET /ready`: `200` only when model storage is writable/has its free-space floor and the last market-data dependency state is not unavailable; otherwise `503`.
- `GET /models`: model manifest.
- `GET /api/v1/search?query=AAPL`: up to eight Yahoo suggestions. A syntactically valid exact symbol remains available as a clearly generic `SYMBOL` fallback if autocomplete is down.
- `GET /api/v1/info?ticker=AAPL`: fundamentals; thread-safe 3600-second cache by default.
- `GET /api/v1/prediction-status/{request_id}`: short-lived progress for a request ID supplied to a forecast request. Rate limit: 60/minute/IP. Responses use `Cache-Control: no-store`.

## `GET /api/v1/predict`

Parameters: `ticker` (default `AAPL`, `[A-Z0-9.\-]{1,12}`), `days` (default 7, range 1–30). Rate limit: 5/minute/IP. Prediction cache: 300 seconds by default. Send an optional `X-Prediction-Request-ID` UUIDv4 header to enable short-lived status polling.

```json
{
  "ticker": "AAPL",
  "historical_dates": ["2026-07-24"],
  "historical_prices": [213.88],
  "future_dates": ["2026-07-27", "2026-07-28", "2026-07-29"],
  "predicted_prices": [214.1, 214.5, 213.9],
  "forecast_days": 3,
  "metrics": {
    "metric_source": "walk_forward_out_of_fold",
    "metric_scope": "all_forecast_horizons",
    "rmse": 4.82,
    "mae": 3.61,
    "mape": 1.74,
    "r2": 0.91,
    "directional_accuracy": 0.63
  },
  "metadata": {
    "architecture": "lstm",
    "output_width": 30,
    "calendar": "NYSE",
    "metric_source": "walk_forward_out_of_fold",
    "data_snapshot": {"snapshot_id": "sha256..."},
    "data_quality": {"schema_version": 2, "policy": "fail_closed", "status": "complete"},
    "timings_seconds": {"queue_wait": "<measured seconds or null>", "market_data": "<measured seconds or null>", "feature_preparation": "<measured seconds or null>", "artifact_load_validation": "<measured seconds or null>", "training": "<measured seconds or null>", "inference": "<measured seconds or null>", "total": "<measured seconds>"},
    "execution": {"mode": "artifact_loaded", "coalesced": false},
    "artifact_state_before": "fresh",
    "artifact_action": "loaded"
  }
}
```

Metrics can instead be `{"metric_source":"unavailable","detail":"..."}`. They are never computed on the final production model's training samples.

`timings_seconds` are measured for this request. Stages that did not run are `null`; a response-cache hit has `null` pipeline stages but a measured `total`. `artifact_state_before` is `fresh`, `missing`, `stale`, or `incompatible` when artifact validation ran (otherwise `null` for a response-cache hit). `artifact_action` is `loaded`, `retrained`, or `not_applicable`. `execution.mode` is `response_cache_hit`, `artifact_loaded`, `trained`, or `coalesced`.

## `GET /api/v1/predict/direction`

Same parameters/cache/rate limit. `directions` contains strings (`"Up"`/`"Down"`), and `probabilities` contains sigmoid probabilities in `[0,1]`. Both arrays and `future_dates` have exactly `forecast_days` entries. `attention_weights` has exactly 60 dated entries. The cached model always has `output_width: 30`, regardless of request order.

```json
{
  "ticker": "VOD.L",
  "forecast_days": 3,
  "future_dates": ["2026-07-27", "2026-07-28", "2026-07-29"],
  "directions": ["Up", "Down", "Up"],
  "probabilities": [0.68, 0.43, 0.59],
  "attention_weights": [{"index": 0, "date": "2026-05-01", "weight": 0.012}],
  "metrics": {"metric_source": "walk_forward_out_of_fold", "precision": 0.61, "recall": 0.58, "f1": 0.59},
  "sentiment": {"score": 0.23, "status": "live", "provider": "yfinance", "method": "vader_financial"},
  "metadata": {"architecture": "bidirectional_lstm_with_attention", "output_width": 30, "calendar": "LSE"}
}
```

Sentiment is untrusted, headline-only external data. Failures produce a documented `fallback` score of `0.0`; sentiment does not enter model features.

## `GET /api/v1/prediction-status/{request_id}`

Use the UUIDv4 value sent in `X-Prediction-Request-ID` on a pending forecast request. The response contains generic request lifecycle/status fields and the current shared stage (`queued`, `downloading_market_data`, `preparing_features`, `checking_artifact`, `training`, `generating_forecast`, `completed`, or `failed`), plus whether the caller joined matching in-flight work. Unknown, expired, or malformed IDs return a generic `404` response.

Status telemetry is in-process and short-lived: it is intended for request UX and diagnostics, not durable observability or a production benchmark.

## `GET /api/v1/diagnostics/{ticker}`

Query `model_type` is one of `lstm`, `attention`, or `bilstm_attention_direction`. Returns persisted fold boundaries, untouched-fold predictions/residuals, cross-validation aggregates, and model metadata. Returns `404` when no activated validation artifacts exist.
