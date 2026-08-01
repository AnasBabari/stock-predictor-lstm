# API reference

Local backend: `http://127.0.0.1:8000`. Interactive OpenAPI: `/docs`; schema: `/openapi.json`.

All errors use `{"detail":"..."}`. `400` means invalid or insufficient instrument data, `422` means query-schema failure, `429` means per-client rate limiting, and `503` means retryable market-data or service failure. Server forecast compatibility responses can be successful baselines; their metadata never claims a learned server model.

## Probes and discovery

- `GET /`: service metadata and links to health, readiness, and OpenAPI documentation.
- `GET /health`: process liveness only.
- `GET /ready`: readiness for the market-data dependency. Model storage is not required; browser IndexedDB is outside the server readiness probe.
- `GET /models`: reports `server_models.status = disabled` and `browser_training.status = available` with `storage = indexeddb`.
- `GET /api/v1/search?query=AAPL`: up to eight Yahoo suggestions. A syntactically valid exact symbol remains available as a generic `SYMBOL` fallback if autocomplete is down.
- `GET /api/v1/info?ticker=AAPL`: fundamentals with a thread-safe cache.
- `GET /api/v1/prediction-status/{request_id}`: short-lived compatibility telemetry for a request ID. Responses use `Cache-Control: no-store`.

## `GET /api/v1/training-data`

Parameters: `ticker` (default `AAPL`, `[A-Z0-9.\-]{1,12}`). Rate limit: 10/minute/client. The server fetches and validates the bounded feature snapshot; it accepts no client-supplied feature matrix and writes no files.

The response preserves the 22-feature order used by the offline pipeline, includes a deterministic `snapshot_id`, 60-session `window_size`, 30-step `output_width`, close-column index, historical prices, and backend-generated future trading dates.

```json
{
  "ticker": "MSFT",
  "schema_version": 3,
  "snapshot_id": "sha256...",
  "generated_at": "2026-08-01T12:00:00Z",
  "feature_names": ["Open", "High", "Low", "Close", "Volume", "SMA_20", "EMA_20", "RSI_14", "MACD", "MACD_Signal", "BB_Upper", "BB_Lower", "ATR_14", "OBV", "SPY_Return_1D", "QQQ_Return_1D", "VIX_Return_1D", "TNX_Return_1D", "Month_Sin", "Month_Cos", "Day_Sin", "Day_Cos"],
  "window_size": 60,
  "output_width": 30,
  "close_index": 3,
  "dates": ["2026-07-29"],
  "features": [["finite numeric values ..."]],
  "historical_prices": ["finite positive closes ..."],
  "future_dates": ["2026-07-30"],
  "calendar": "NYSE",
  "data_snapshot": {"snapshot_id": "sha256..."}
}
```

Invalid tickers and insufficient data return sanitized `400` responses; upstream failures return `503`. Non-finite feature rows, invalid closes, and snapshots over the 2,000-row bound are rejected or trimmed before serialization.

## Compatibility forecast endpoints

### `GET /api/v1/predict`

Parameters: `ticker` (default `AAPL`, `[A-Z0-9.\-]{1,12}`), `days` (default 7, range 1–30). Rate limit: 5/minute/client. The endpoint is retained while the frontend migration stabilizes. It fetches market data and returns a persistence forecast, not a server-trained LSTM. The response engine is:

```json
{
  "family": "persistence",
  "role": "server_disabled_fallback",
  "baseline_fallback": true
}
```

`metrics.metric_source` is `baseline_definition`; it is not a walk-forward score. The bounded response cache contains only baseline responses and does not store model weights.

### `GET /api/v1/predict/direction`

The same parameters and rate limit return a recent-up-session base-rate forecast. `directions` and `probabilities` have exactly `forecast_days` entries; `attention_weights` is an empty list because the server direction model is disabled. Live headline sentiment may be returned as response context only and is never a browser-model feature.

```json
{
  "ticker": "MSFT",
  "forecast_days": 3,
  "future_dates": ["2026-08-03", "2026-08-04", "2026-08-05"],
  "directions": ["Up", "Down", "Up"],
  "probabilities": [0.55, 0.55, 0.55],
  "attention_weights": [],
  "metrics": {"metric_source": "baseline_definition", "naive_baseline": 0.55},
  "metadata": {"engine": {"family": "recent_base_rate", "role": "server_disabled_fallback", "baseline_fallback": true}}
}
```

## Browser model response contract

The Vercel frontend fetches `/api/v1/training-data`, then sends the snapshot to a TensorFlow.js Web Worker. The worker uses `model.fit()` with a compact LSTM, no shuffle, an 80% train split, and a `forecast_days - 1` purge. It reports progress after every epoch, supports cancellation, disposes tensors, and tries WebGL before CPU. The build-time VITE_BROWSER_TRAINING_ENABLED=false setting is an explicit rollback to the compatibility baseline.

A learned browser response is labelled `metadata.engine.role = browser_learned`, `metadata.browser_training = true`, and `metadata.metric_source = browser_purged_holdout`. A cached response uses `execution.mode = browser_artifact_loaded`; a newly trained response uses `execution.mode = browser_trained`. IndexedDB keys include the model version, schema, ticker, forecast type, snapshot, window, and output width. Models expire/evict according to the seven-day and six-entry per-browser policy, and the UI provides a clear-local-models control.

Price holdout metrics include MAE, MSE, RMSE, MAPE, R², relative MAE, and relative RMSE against persistence. Direction metrics include accuracy, precision, recall, F1, balanced accuracy, Brier score, and naive-baseline accuracy. These values are labelled `browser_purged_holdout`; they are not the offline five-fold walk-forward metrics.

Unsupported workers or failed browser training use an explicit `baseline_fallback` response. A flat persistence result must never be presented as an LSTM result.

## Compatibility telemetry

`GET /api/v1/prediction-status/{request_id}` accepts the UUIDv4 value sent in `X-Prediction-Request-ID` on a compatibility forecast request. Status stages remain `queued`, `downloading_market_data`, `preparing_features`, `checking_artifact`, `training`, `generating_forecast`, `completed`, or `failed`; `training` is reserved compatibility telemetry and is not entered by the public browser path. Unknown, expired, or malformed IDs return a generic `404`.

Forecast metadata includes `artifact_state_before` and `artifact_action` alongside the typed `timings_seconds` fields `queue_wait`, `market_data`, `feature_preparation`, `artifact_load_validation`, `training`, `inference`, and `total`. Stages that did not run are `null`, with a non-negative measured `total`. The typed execution modes remain `response_cache_hit`, `artifact_loaded`, `baseline_fallback`, `trained`, and `coalesced`; artifact states are `fresh`, `missing`, `stale`, and `incompatible`; artifact actions are `loaded`, `retrained`, and `not_applicable`. A response-cache hit has `null` pipeline stages and a measured total. This caller-level semantics is short-lived, in-process telemetry, not durable observability. Completed and failed views may remain for up to 10 minutes and can be evicted under registry capacity pressure.

## Offline evidence endpoints

- `GET /api/v1/diagnostics/{ticker}` exposes persisted walk-forward artifacts only when the opt-in offline trainer has produced them; production Render has no such files.
- `GET /api/v1/model-performance/{ticker}` discloses browser-training availability and any offline evidence. It must not imply that a server model is active.

Live sentiment is untrusted headline-only external data. Historical news can enter only an offline, timestamped, leakage-safe ablation and must pass the same purged holdout and promotion gates as other features.