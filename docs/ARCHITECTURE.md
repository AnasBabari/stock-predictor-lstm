# Architecture

## Production request flow

The browser requests a signed global volatility forecast from FastAPI. FastAPI validates the ticker and horizon, verifies the immutable release once per process, builds a causal market snapshot, runs every CPU ONNX ensemble member, and returns dated quantile paths plus the exact evidence identity. Render has no TensorFlow import, Keras artifact directory, boot-time training, or model-weight upload.

\`\`\`mermaid
flowchart LR
    browser[React SPA / Vercel] --> api[FastAPI / Render]
    api --> release[Signed ONNX release]
    api --> market[Yahoo history + exchange calendar]
    market --> snapshot[Causal Deployable Schema v5 snapshot]
    release --> runtime[CPU ensemble runtime]
    snapshot --> runtime
    runtime --> cone[Certified volatility cone]
    cone --> browser
    api -->|missing, invalid, stale, uncertified| abstain[503 explicit abstention]
\`\`\`

The production route is \`GET /api/v2/forecast?ticker=MSFT&horizon=7\`. Supported horizons are \`{1, 3, 5, 7, 14, 30}\` trading sessions. A request is accepted only when the signed manifest, per-file SHA-256 checksums, Ed25519 signature, schema, feature order, ONNX I/O names, member set, model size, and requested horizon all verify.

The API returns a p05–p95 lognormal cone derived from certified cumulative variance. p50 is the unchanged-close location baseline; it is not a learned price path. The current release does not certify return location or direction, and the UI says so. A failed horizon is an abstention, not a baseline masquerading as a model.

## Data contract

\`backend/services/volatility_snapshot.py\` builds Deployable Schema v5 with 26 causal features. The 60-row input window, feature names, calendar dates, origin close, and snapshot fingerprint are bound to the runtime contract. Cross-sectional ranks, research-only regime labels, and future-filled values cannot enter the serving matrix.

\`GET /api/v1/training-data\` remains a bounded diagnostic/research snapshot. It rejects client matrices, non-finite values, invalid chronology, invalid tickers, and oversized responses. It is not used to train models in the Render request process.

## Offline research boundary

The RTX workstation runs the research harness in \`research/volatility_forecasting\`:

1. Build an immutable, license-acknowledged panel snapshot.
2. Derive causal market features and econometric baselines.
3. Evaluate persistence, shrunk mean, Ridge/ElasticNet, DLinear, residual TCN, and GARCH-LSTM challengers on calendar-aligned expanding folds with purge and embargo.
4. Run paired market-only versus market-plus-news ablations on exactly the same folds and origins.
5. Apply horizon-specific QLIKE, coverage, calibration, bootstrap, DM/Holm, fold-consistency, and seed-dispersion gates.
6. Consume one untouched holdout only after the methodology gate and winner decision pass.
7. Export ONNX members, verify CPU parity, sign the manifest, and mount the immutable release.

The final refit is never used to claim evaluation metrics. Locked evaluation metrics describe only untouched out-of-fold or certification observations.

## News boundary

Historical news is aligned by publication timestamp before each market origin. The GDELT builder stores raw-event checksums, daily aggregation statistics, topic exposures, coverage cutoffs, decay settings, and explicit archive-gap dates. A provider missing archive is an annotated gap, never an unobserved no-news day. News features remain an ablation input until they demonstrate incremental value against the market-only champion.

Live Yahoo headlines remain context-only in compatibility responses. They are not silently merged into the production feature matrix.

## Release and readiness

\`backend/release/bundle.py\` creates an immutable manifest with runtime schema, model id, member seeds/files, feature order, certified horizons, certification metrics, and checksums. \`VolatilityOnnxRuntime.from_release_bundle\` verifies the manifest and opens CPU sessions. Diskless hosts may bootstrap a deterministic immutable ZIP only after checking its configured SHA-256, bounded safe extraction, Ed25519 signature, and every member checksum. \`/models\` reports the verified model id and horizons; \`/ready\` can require the release with \`VOLATILITY_SERVING_REQUIRED=true\`.

The response cache is bounded and keyed by \`(signed_model_id, ticker, horizon)\`. A newly promoted release therefore cannot inherit an older model’s cached response. Cache entries are also rejected before use when the current release no longer certifies the horizon.

## Compatibility and security

The legacy \`/api/v1/predict\` routes are persistence/base-rate compatibility endpoints with \`server_disabled_fallback\` metadata. They never train. Trusted proxy addresses are exact configured peers/CIDRs; forwarded headers are replaced at Nginx. CORS is explicit, errors are sanitized, and no user identifiers or model weights are sent to Render.

The source tree temporarily retains the old TFJS methodology harness, but Vite includes it only when the explicit legacy test build sets `VITE_VOLATILITY_SERVING_ENABLED=false` and `VITE_BROWSER_TRAINING_ENABLED=true`. Normal Vercel production builds compile that path out, and `npm run check:production-bundle` rejects emitted training-worker or TFJS artifacts. Production must not advertise browser-trained learned forecasts. Final source deletion must remove its dependencies, profile selector, worker tests, and methodology gate together in one reviewed boundary.

## Deployment gate

Repository checks validate the release contract, resource budget, API docs, and browser integration. Provider smoke tests must verify \`/health\`, \`/ready\`, \`/models\`, and a seven-day \`/api/v2/forecast\` response against the same deployed commit. See [DEPLOYMENT_GATE.md](DEPLOYMENT_GATE.md).
