# API reference

Local backend: `http://127.0.0.1:8000`. Interactive OpenAPI: `/docs`; schema: `/openapi.json`.

All errors use `{"detail":"..."}`. `400` means invalid/insufficient instrument data, `422` means query-schema failure, `429` means per-client rate limiting, and `503` means no fresh prepared artifact, bounded capacity, timeout, readiness failure, or retryable benchmark unavailability.

## Probes and discovery`r`n`r`n- `GET /`: service metadata and links to health, readiness, and OpenAPI documentation.

- `GET /health`: process liveness only.
- `GET /ready`: `200` only when model storage is writable/has its free-space floor and the last market-data dependency state is not unavailable; otherwise `503`.
- `GET /models`: model manifest.
- `GET /api/v1/search?query=AAPL`: up to eight Yahoo suggestions. A syntactically valid exact symbol remains available as a clearly generic `SYMBOL` fallback if autocomplete is down.
- `GET /api/v1/info?ticker=AAPL`: fundamentals; thread-safe 3600-second cache by default.
- `GET /api/v1/prediction-status/{request_id}`: short-lived progress for a request ID supplied to a forecast request. Rate limit: 60/minute/IP. Responses use `Cache-Control: no-store`.

## `GET /api/v1/predict`

Parameters: `ticker` (default `AAPL`, `[A-Z0-9.\-]{1,12}`), `days` (default 7, range 1–30). Rate limit: 5/minute/client. Prediction cache: 300 seconds by default; a cache hit revalidates its underlying artifact and is evicted if it is no longer fresh. Send an optional `X-Prediction-Request-ID` UUIDv4 header to enable short-lived status polling. Public requests load only fresh, validated 30-day artifacts and never initiate training. Missing, stale, or invalid artifacts may use a clearly labelled deterministic baseline after bounded market-data retrieval; upstream circuit failures return `503`.

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
    "metric_scope": "forecast_origin_horizon_pairs",
    "rmse": 4.82,
    "mae": 3.61,
    "mase": 0.84,
    "rmsse": 0.89,
    "relative_mae": 0.81,
    "relative_rmse": 0.86,
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

Metrics can instead be `{"metric_source":"unavailable","detail":"..."}`. They are never computed on the final production model's training samples. MASE/RMSSE use training-only naïve scales; `relative_mae` and `relative_rmse` divide candidate error by the no-change persistence error for the same forecast origins. Values below `1.0` indicate improvement over their respective baseline.

`timings_seconds` use caller-level semantics. Stages that did not run for that caller are `null`; a response-cache hit has `null` pipeline stages but a measured `total`. A coalesced caller receives its independently measured `total` while owner-executed pipeline stages remain `null`. `artifact_state_before` is `fresh`, `missing`, `stale`, or `incompatible` when artifact validation ran (otherwise `null` for a response-cache hit). `artifact_action` is `loaded`, `retrained`, or `not_applicable`. For a coalesced response, the artifact fields describe the shared job's validated outcome rather than a second artifact check by the joiner. `execution.mode` is `response_cache_hit`, `artifact_loaded`, `baseline_fallback`, `trained`, or `coalesced`. A baseline fallback is labelled in `metadata.engine` and never represented as learned-model evidence.

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
  "metrics": {
    "metric_source": "walk_forward_out_of_fold",
    "precision": 0.61,
    "recall": 0.58,
    "f1": 0.59,
    "balanced_accuracy": 0.60,
    "brier_score": 0.23,
    "log_loss": 0.65,
    "naive_baseline": 0.53
  },
  "sentiment": {
    "score": 0.23,
    "status": "live",
    "provider": "yfinance",
    "method": "vader_financial",
    "article_count": 8,
    "timestamped_article_count": 7,
    "freshest_article_at": "2026-07-26T14:32:00+00:00",
    "reason": null
  },
  "metadata": {"architecture": "bidirectional_lstm_with_attention", "output_width": 30, "calendar": "LSE"}
}
```

Sentiment is untrusted, headline-only external data. Failures produce a documented `fallback` score of `0.0`, `status: "fallback"`, zero coverage counts, and a generic `reason` such as `no_usable_news` or `upstream_error`. Live sentiment does not enter model features. Historical news can enter only an offline ablation, and only timestamped articles published before each session are eligible.

## `GET /api/v1/prediction-status/{request_id}`

Use the UUIDv4 value sent in `X-Prediction-Request-ID` on a pending forecast request. The response contains generic request lifecycle/status fields and the current shared stage (`queued`, `downloading_market_data`, `preparing_features`, `checking_artifact`, `training`, `generating_forecast`, `completed`, or `failed`), plus whether the caller joined matching in-flight work. `training` remains a reserved telemetry value for compatibility but is not entered by public requests. Unknown, expired, or malformed IDs return a generic `404` response.

Completed and failed status views are eligible to remain available for up to 10 minutes, but terminal views may be evicted earlier under registry capacity pressure. Status telemetry is short-lived and in-process: it is intended only for request UX and diagnostics, not durable storage, production observability, or a production benchmark.

## `GET /api/v1/diagnostics/{ticker}`

Query `model_type` is one of `lstm`, `gru`, `attention`, `bilstm_attention_regression`, or `bilstm_attention_direction`. Returns persisted fold boundaries, untouched-fold predictions/residuals, cross-validation aggregates, per-horizon metrics for regression artifacts, and model metadata. Returns `404` when no activated validation artifacts exist.

## `GET /api/v1/model-performance/{ticker}`

Query `forecast_type` is `price` or `direction`. Returns the currently selected engine family and role, its attached metrics, snapshot and validation provenance, or an explicit `baseline_definition` result when no learned candidate has qualifying fresh evidence.
