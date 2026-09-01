# Architecture

## Production request flow

The browser requests a causal volatility forecast from FastAPI. FastAPI validates the ticker, horizon, and baseline, builds a bounded market snapshot from the latest available OHLCV, computes an auditable statistical forecast, and returns dated quantile paths plus the snapshot identity. This active path does not train, deserialize, or persist model weights, so it remains suitable for the lightweight Render deployment.

\`\`\`mermaid
flowchart LR
    browser[React SPA / Vercel] --> api[FastAPI / Render]
    api --> market[Yahoo history + exchange calendar]
    market --> snapshot[Causal Deployable Schema v5 snapshot]
    snapshot --> baseline[Causal volatility baseline]
    baseline --> distribution[raw Gaussian p05-p95 reference scenario]
    distribution --> browser
    api -->|invalid or unavailable data| error[Sanitized 4xx/503]
\`\`\`

The active route is \`GET /api/v1/volatility/forecast?ticker=MSFT&horizon=7&model=har_rv\`. Supported horizons are \`{1, 3, 5, 7, 14, 30}\` trading sessions and supported models are the causal persistence, rolling-mean, EWMA, and log-HAR baselines. The response reports the baseline explicitly, includes the target definition and snapshot metadata, and uses the latest close as the median price path. No claim of a learned or certified model is made.

The historical signed global-model route remains \`GET /api/v2/forecast\`. It is a separate compatibility and research interface, still fail-closed when no signed release is configured. It is not the active frontend path and must not be conflated with the transparent baseline route.

## Data contract

\`backend/services/volatility_snapshot.py\` builds Deployable Schema v5 with 26 causal features. The 60-row input window, feature names, calendar dates, origin close, and snapshot fingerprint are bound to the runtime contract. Cross-sectional ranks, research-only regime labels, and future-filled values cannot enter the serving matrix. The active endpoint consumes the same bounded snapshot and derives a realised-volatility target without using observations after the origin.

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

V11.2 is retained as an archived numeric-only research protocol. Its one-shot
reserve is permanently `INVALIDATED_OPENED`; preparation, certification,
release assembly, and runtime loading reject it. The remaining modules support
historical reproducibility only. See [VOLATILITY_V11_2.md](VOLATILITY_V11_2.md).

The simplified active product deliberately does not require a signed learned
release. Offline learned candidates can be compared with the baselines using
the same immutable snapshots and temporal splits. Promotion to a future learned
serving path is a separate decision that must preserve the metric-source and
provenance labels; it cannot silently replace the active baseline response.

The final refit is never used to claim evaluation metrics. Locked evaluation
metrics describe only untouched out-of-fold or certification observations.
The failed V11.2 report preserves `metric_source=sealed_holdout_once` as
historical evidence, but it cannot produce a signed serving payload.

## News boundary

Historical news is aligned by publication timestamp before each market origin. The GDELT builder stores raw-event checksums, daily aggregation statistics, topic exposures, coverage cutoffs, decay settings, and explicit archive-gap dates. A provider missing archive is an annotated gap, never an unobserved no-news day. News features remain an ablation input until they demonstrate incremental value against the market-only champion.

Live Yahoo headlines remain context-only in compatibility responses. They are not silently merged into the production feature matrix.

## Release and readiness

\`backend/release/bundle.py\` creates an immutable manifest with runtime schema, model id, member seeds/files, feature order, certified horizons, certification metrics, and checksums. \`VolatilityOnnxRuntime.from_release_bundle\` verifies the manifest and opens CPU sessions. Diskless hosts may bootstrap a deterministic immutable ZIP only after checking its configured SHA-256, bounded safe extraction, Ed25519 signature, and every member checksum. \`/models\` reports the verified model id and horizons; \`/ready\` can require the release with \`VOLATILITY_SERVING_REQUIRED=true\`.

The active route does not retain model or response artifacts on Render. The
frontend may reuse a result during the current page session, but a reload
fetches a new snapshot. If a server-side cache is added later, it must be keyed
by \`(snapshot_id, ticker, horizon, model)\` so a new snapshot cannot inherit
an older forecast. Legacy signed-release cache entries remain keyed by the
signed model id and are never used by the active endpoint.

## Compatibility and security

The legacy \`/api/v1/predict\` routes are persistence/base-rate compatibility endpoints with \`server_disabled_fallback\` metadata. They never train. Trusted proxy addresses are exact configured peers/CIDRs; forwarded headers are replaced at Nginx. CORS is explicit, errors are sanitized, and no user identifiers or model weights are sent to Render.

The frontend uses the active causal baseline contract (\`GET /api/v1/volatility/forecast\`). The legacy \`GET /api/v2/forecast\` path remains available only for callers that explicitly request the signed global-model interface. Browser training is not required on the active path, and no forecast is labelled as an LSTM, global model, or certified artifact unless the corresponding evidence is actually present.

## Deployment gate

Repository checks validate the active baseline contract, resource budget, API docs, and browser integration. Provider smoke tests must verify \`/health\`, \`/ready\`, \`/models\`, and a seven-day \`/api/v1/volatility/forecast\` response against the same deployed commit; the legacy \`/api/v2/forecast\` probe is separate and may correctly return an explicit no-certified-model response. See [DEPLOYMENT_GATE.md](DEPLOYMENT_GATE.md).
