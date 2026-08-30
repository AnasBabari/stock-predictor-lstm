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
    runtime --> distribution[Certified return distribution or volatility cone]
    distribution --> browser
    api -->|missing, invalid, stale, uncertified| abstain[503 explicit abstention]
\`\`\`

The production route is \`GET /api/v2/forecast?ticker=MSFT&horizon=7\`. Supported horizons are \`{1, 3, 5, 7, 14, 30}\` trading sessions. A request is accepted only when the signed manifest, per-file SHA-256 checksums, Ed25519 signature, schema, feature order, ONNX I/O names, member set, model size, and requested horizon all verify.

The API returns p05–p95 price quantiles derived from the signed runtime. A
legacy volatility-only release exposes a zero-location lognormal cone whose p50
is the unchanged-close baseline. A V11.2 release can additionally certify a
Student-t return distribution: its terminal location and variance become the
learned median path and uncertainty band, while intermediate daily points are
labelled interpolations rather than independently certified horizons. Direction
remains uncertified unless its own head passes a separate gate. A failed
horizon is an abstention, not a baseline masquerading as a model.

## Data contract

\`backend/services/volatility_snapshot.py\` builds Deployable Schema v5 with 26 causal features. The 60-row input window, feature names, calendar dates, origin close, and snapshot fingerprint are bound to the runtime contract. Cross-sectional ranks, research-only regime labels, and future-filled values cannot enter the serving matrix.

\`GET /api/v1/training-data\` remains a bounded diagnostic/research snapshot. It rejects client matrices, non-finite values, invalid chronology, invalid tickers, and oversized responses. It is not used to train models in the Render request process.

## Data provenance and certification eligibility

Three data inputs exist and they have **different** evidentiary standing. Conflating them is the single easiest way to manufacture a false result.

| Input | Provenance | Certification eligible |
| --- | --- | --- |
| `data/fixtures/synthetic_csco_like_golden_v1.csv` | Deterministically generated, CC0-1.0, hash-pinned in source | **No** — software regression only |
| `research/ndx100` point-in-time universe | Secondary development reconstruction | **No** — not an attested constituent source |
| Local market panel caches | `yfinance` development download | **No** — redistribution and training rights not cleared |

The synthetic fixture exists so the CSCO code path stays deterministic in CI. It contains **no** market observation. Every CLI run against it prints `SYNTHETIC SOFTWARE REGRESSION — NOT MARKET PERFORMANCE` before printing any number, and its metrics must never be cited as forecasting skill. See [data/fixtures/README.md](../data/fixtures/README.md).

A certified market model additionally requires an attested point-in-time constituent source and a licensed panel that permits training and derived-model distribution. Until both are held, every v9 artifact carries `evidence_role=development_diagnostic_only` and `certification_eligible=false`, and the v9 protocol records this in `configs/volatility_v9_protocol.json` under `data_eligibility`.

## Offline research boundary

The RTX workstation runs the research harness in \`research/volatility_forecasting\`:

1. Build an immutable, license-acknowledged panel snapshot.
2. Derive causal market features and econometric baselines.
3. Evaluate persistence, shrunk mean, Ridge/ElasticNet, DLinear, residual TCN, and GARCH-LSTM challengers on calendar-aligned expanding folds with purge and embargo.
4. Run paired market-only versus market-plus-news ablations on exactly the same folds and origins.
5. Apply horizon-specific QLIKE, coverage, calibration, bootstrap, DM/Holm, fold-consistency, and seed-dispersion gates.
6. Consume one untouched holdout only after the methodology gate and winner decision pass.
7. Export ONNX members, verify CPU parity, sign the manifest, and mount the immutable release.

V11.2 is an additive numeric-only development protocol. It uses an exactly 64-security point-in-time universe, a session-grouped 70/15/15 split, independent per-horizon routing, and a fresh AES-GCM encrypted holdout. The development process receives only train/validation files; the decryption key and sealed payload are reserved for a separate one-shot certification command. Epoch zero is evaluated before neural updates so the residual learner can restore the HAR prior. See [VOLATILITY_V11_2.md](VOLATILITY_V11_2.md).

The final refit is never used to claim evaluation metrics. Locked evaluation
metrics describe only untouched out-of-fold or certification observations.
For V11.2, the signed serving payload preserves `metric_source=sealed_holdout_once`
and exposes the certified return-distribution head separately from the
volatility head.

## News boundary

Historical news is aligned by publication timestamp before each market origin. The GDELT builder stores raw-event checksums, daily aggregation statistics, topic exposures, coverage cutoffs, decay settings, and explicit archive-gap dates. A provider missing archive is an annotated gap, never an unobserved no-news day. News features remain an ablation input until they demonstrate incremental value against the market-only champion.

Live Yahoo headlines remain context-only in compatibility responses. They are not silently merged into the production feature matrix.

## Release and readiness

\`backend/release/bundle.py\` creates an immutable manifest with runtime schema, model id, member seeds/files, feature order, certified horizons, certification metrics, and checksums. \`VolatilityOnnxRuntime.from_release_bundle\` verifies the manifest and opens CPU sessions. Diskless hosts may bootstrap a deterministic immutable ZIP only after checking its configured SHA-256, bounded safe extraction, Ed25519 signature, and every member checksum. \`/models\` reports the verified model id and horizons; \`/ready\` can require the release with \`VOLATILITY_SERVING_REQUIRED=true\`.

The response cache is bounded and keyed by \`(signed_model_id, ticker, horizon)\`. A newly promoted release therefore cannot inherit an older model’s cached response. Cache entries are also rejected before use when the current release no longer certifies the horizon.

## Compatibility and security

The legacy \`/api/v1/predict\` routes are persistence/base-rate compatibility endpoints with \`server_disabled_fallback\` metadata. They never train. Trusted proxy addresses are exact configured peers/CIDRs; forwarded headers are replaced at Nginx. CORS is explicit, errors are sanitized, and no user identifiers or model weights are sent to Render.

Legacy browser training has been completely retired. The frontend purely interfaces with the verified server global volatility forecasting contract (`GET /api/v2/forecast`), and CI validates that the production bundle is TFJS-free. Production never advertises or performs browser-trained forecasts.

## Deployment gate

Repository checks validate the release contract, resource budget, API docs, and browser integration. Provider smoke tests must verify \`/health\`, \`/ready\`, \`/models\`, and a seven-day \`/api/v2/forecast\` response against the same deployed commit. See [DEPLOYMENT_GATE.md](DEPLOYMENT_GATE.md).
