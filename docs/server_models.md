# Legacy server-pretrained forecast bundles

This document describes the retained per-ticker compatibility path. It is not
the production contract: production uses the signed global-volatility ONNX
release documented in GLOBAL_MODELS.md. The routes and job remain isolated for
migration/rollback and are disabled by default; they must never be presented as
the global champion or as a browser-trained result.

## Configuration

All settings live in `backend/config.py` and can be set via environment
variables or `backend/.env`:

| Setting | Default | Meaning |
| --- | --- | --- |
| `SERVER_FORECAST_SERVING_ENABLED` | `false` | Serves bundles on request when a fresh promoted artifact exists. Routes are always registered but dormant while false. |
| `SERVER_TRAINING_ENABLED` | `false` | Reserved for future request-time training; the current job runs out-of-band. |
| `SERVER_FORECAST_ALLOWLIST` | empty | Comma-separated tickers that may be served (upper-cased). |
| `SERVER_FORECAST_MAX_AGE_HOURS` | `36` | Bundle freshness window. |
| `SERVER_FORECAST_CACHE_TTL` | `900` | Serving response cache seconds. |
| `SERVER_BUNDLE_RETENTION_DAYS` | `30` | Minimum age before a non-current, non-rollback bundle object may be pruned. Registry and audit rows are retained. |
| `SERVER_FORECAST_PRIVATE_KEY_PATH` | unset | PEM path for the training job's Ed25519 signer; required by the job (exit 2 otherwise). |
| `SERVER_FORECAST_PUBLIC_KEY_PATH` | unset | PEM public key used by serving for Ed25519 verification. Required for configured serving: there is no digest-only mode. A missing key is reported as `unconfigured`; a configured-but-broken key is `integrity_failure` (a 503 that allows no fallback). |
| `REGISTRY_DATABASE_URL` | unset | Postgres URL for the promotion registry. |
| `S3_ENDPOINT_URL`, `S3_BUCKET`, `S3_KEY_PREFIX` | unset, unset, `artifacts` | Object storage for bundle blobs. |
| `TRAINING_MODE` | `browser_only` | One of `browser_only`, `hybrid`, `server_pretrained`. Only `server_pretrained` turns expected absences into `503`s. |

## Artifact identity and versioning

Every bundle has an immutable `version_id`:

```
{ticker}-{forecast_type}-{utc-ts}-{gitsha12}-{snapshot-hash8}
```

for example `AAPL-price-20260102T220000Z-0123456789ab-00000aaa`. The forecast
type (`price` or `direction`) and the snapshot fingerprint (last 8 hex
characters of the deterministic `sha256:` snapshot id, zfilled, or `unknown`)
are part of the identity, so retrains on new data or a different target never
collide on one storage key. `unknown` is used when the git SHA or snapshot id is
unavailable.

## Bundle contract

`ServerForecastBundle` is the frozen payload written to storage and served
read-only:

- `origin_close` / `origin_date`: the last training close and its trading date.
- `future_dates`: exactly 30 calendar-generated future trading dates.
- `predicted_log_returns` / `predicted_prices`: exactly 30 entries; prices are
  strictly positive and derived from the latest close at training time.
- `historical_dates` / `historical_prices`: exactly `HISTORY_DISPLAY_WINDOW`
  (120) trailing sessions for the chart, so serving never touches upstream
  providers.
- `evidence`: per-horizon metrics with `metric_source` set to
  `server_purged_walk_forward`.

Reproducibility metadata mirrors the browser contract: Stationary Schema v4,
the 28 `FEATURES_V4` columns in exact order, `TARGET_MODE =
cumulative_log_return_v1`, a 60-step window, the train-only fitted robust
scaler's medians/IQRs, horizon list, Python and library versions, and the git
commit. The bundle itself is validated against those frozen values at serving
time; a violation fails closed with `503`.

## Training job

`backend/scripts/run_server_training.py` is the only producer. It loads the
Ed25519 signer from `SERVER_FORECAST_PRIVATE_KEY_PATH` (exits 2 if unset),
then either trains one ticker (`--ticker AAPL`) or drains the registry job
queue (`--once` for cron mode).

For each ticker `train_server_forecast`:

1. Fetches the same bounded, validated market snapshot as the browser path
   (schema v4, deterministic `snapshot_id`).
2. Fits the torch-free `elastic_net` family on the full 1-30-day horizon range.
3. Computes the walk-forward evidence and promotes a candidate only when pooled
   `relative_rmse < 0.98` and `relative_mae < 0.98`; otherwise the job records a
   rejected/no-candidate audit entry.
4. Builds the self-contained bundle. The embedded forecast comes from the last
   60 raw feature rows transformed by the train-fitted scaler (never from a
   stale dataset row), with `origin_date`/`origin_close` equal to the final
   training session.
5. Signs the bundle bytes with the private key, stores the blob immutably, and
   handoff order is: bundle first, then registry row (`candidate`), then
   `promote`.

Promotion (`PostgresRegistry.promote`) is a single transaction: it selects the
promotions row with `FOR UPDATE`, demotes the current champion to `candidate`,
marks the new row `promoted`, saves the previous `version_id`, and appends an
audit entry with both versions. Re-promoting the current champion is a no-op.
All storage writes are immutable: `put_bundle` uses a conditional
`IfNoneMatch="*"` write so concurrent trainers can never clobber a bundle, and
`exists()`/`ensure_bucket()` treat only a genuine `404/NoSuchKey` as "absent".
The trainer calls `ensure_bucket()` on startup so a missing MinIO bucket in
dev/CI is created once instead of failing every write.

Rollback (`PostgresRegistry.rollback`) runs in the same single transaction and
uses the same demote-before-promote ordering to respect the partial unique
index on promoted rows: it locks the promotions pointer with `FOR UPDATE`,
demotes the current champion, promotes the saved `previous_version`, clears
`previous_version = NULL`, appends an `artifact_rollback` audit entry, and
rolls back the whole transaction on any unexpected error.

## Bundle retention

Each training-job startup runs an idempotent retention sweep before training or
queue processing. `--gc-only` runs the sweep and exits, which is suitable for a
scheduled maintenance job. The sweep uses `server_artifacts.created_at` and the
configured `SERVER_BUNDLE_RETENTION_DAYS` window; it deletes only expired object
blobs and preserves every registry and audit row.

Candidate selection shares the registry's pointer-first lock order with
promotion and rollback. Both `current_version` and `previous_version` are
protected for the duration of the S3 deletes, so the live champion and rollback
target cannot be reclaimed. Deleted rows receive `bundle_pruned_at`; attempting
to promote one fails closed. S3 deletion is idempotent, allowing a sweep to
recover safely if object deletion succeeds but the database transaction later
rolls back. `/ready` and `/models` report the configured retention window when
server serving is enabled. The training/maintenance identity therefore needs
`DeleteObject` permission for the configured bundle prefix; the serving
identity remains read-only.

## Serving semantics

`GET /api/v1/server-forecasts/{ticker}?forecast_type=price|direction&days=N`

| Situation | Status |
| --- | --- |
| Fresh, compatible, signed bundle | `200` with the canonical forecast payload, `ETag` = `version_id`, cached 900 s |
| Serving disabled, ticker not allowlisted, or `forecast_type=direction` (unsupported) | `200 {available: false, reason, fallback: "browser_training"}` |
| Missing/stale/incompatible bundle, mode `browser_only` or `hybrid` | `200 {available: false, ...}` |
| Missing/stale/incompatible bundle, mode `server_pretrained` | `503 {detail: {code, message, fallback: null}}` |
| Missing serving config (no registry/S3), mode `server_pretrained` | `503 {code: "unconfigured", fallback: null}` |
| Public key configured but missing/unreadable/not Ed25519 (any mode) | `503 {code: "integrity_failure", fallback: null}` |
| Registry unavailable, bundle unreadable, digest mismatch, signature failure, contract violation, identity mismatch (any mode) | `503` (fail closed; `fallback: "browser_training"` in browser modes, `null` in `server_pretrained`) |

Every `503` body is `{available: false, code, message, fallback}` with a stable
`code` from a fixed vocabulary and a sanitized `message` — exception details are
logged server-side, never returned. In the browser training modes `503`s carry
`fallback: "browser_training"` so the UI may degrade; in `server_pretrained`
mode they carry `fallback: null` and the frontend must surface the error.

`direction` requests are intentionally not converted to probabilities: the
contract maps the UI's `trend` type to `forecast_type=direction` and falls back
to browser training with `reason: unsupported_forecast_type`.

The success payload is canonical and identical in shape to the client-trained
output: `available`, `ticker`, `forecast_days`, `future_dates`,
`predicted_prices`, `historical_dates`, `historical_prices`, `metrics`, and
`metadata` with `engine.role = "server_pretrained"`, `metric_source =
server_purged_walk_forward`, `origin.date`/`origin.close`, `authenticity =
ed25519_verified` (the only value; verification never degrades to a digest-only
mode), `trained_at`, `browser_training: false`, and `evidence`.

`GET /api/v1/server-forecasts/availability` reports the running mode, the
allowlist, and per-ticker freshness (fresh/stale/missing), cached 300 s.

## Frontend integration

`frontend/src/ml/serverModelClient.js` fetches bundles and passes them through
verbatim once validated: exact vector lengths, strictly positive finite prices,
non-empty history for `price`, a ticker matching the request, and strictly
increasing `future_dates`/`historical_dates`. The UI's `trend` type maps to
`forecast_type=direction` via `API_FORECAST_TYPES`. Fallback is opt-in and
server-directed: the client returns `null` (browser training) only when the
server's response says `fallback: "browser_training"` — a 200 absence, or a
503 in the browser training modes. A 200 invalid payload, a 503 that forbids a
fallback (`fallback: null`), or an unreadable error body throws, so the UI
surfaces the failure instead of silently training in the browser. A cancelled
request propagates `AbortError`. Network-level failures (no response at all)
keep the browser fallback in `hybrid`/`browser_only` deployments; deployments
that require server-pretrained forecasts pass the deployment mode
(`VITE_TRAINING_MODE`/`window.STOCKLSTM_TRAINING_MODE` set to
`server_pretrained`) so the same failure throws instead of silently switching
to browser training.

## Testing

- Unit/contract: `backend/tests/test_server_models.py` (version IDs, storage
  immutability, promotion SQL order), `test_server_retention.py` (expiry,
  pointer protection, tombstones, and retention lock order),
  `test_server_training.py` (training
  pipeline with a real Ed25519 signer), `test_server_forecast_api.py`
  (mode-aware 200/503 matrix, tamper and digest failures fail closed).
- In-process E2E: `test_server_forecast_e2e.py` (train -> promote -> serve ->
  replacement -> rollback).
- Postgres + MinIO integration: `test_server_models_integration.py`, skipped
  unless `SERVER_E2E_DATABASE_URL` and `SERVER_E2E_S3_BUCKET` are set; run in
  CI by `.github/workflows/server-models-e2e.yml`.
- Frontend: `frontend/src/ml/serverModelClient.test.js` (valid payload passthrough,
  identity/chronology rejection, hybrid 503 -> browser fallback, no-fallback 503 ->
  throw); browser contract `frontend/e2e/server-contract.spec.js` (server forecast
  used verbatim, fallback on absence/503, `trend -> direction` mapping, and a
  fail-closed `server_pretrained` 503 that surfaces an error and never trains).
  The e2e contract spec installs a stubbed Web Worker
  (`globalThis.__STOCKLSTM_WORKER_FACTORY__`) via `e2e/fixtures.js` so it never
  builds TF.js models; real browser training is covered by
  `frontend/e2e/browser-real-training.spec.js`.
