# StockLSTM — global volatility forecasting

[![CI](https://github.com/AnasBabari/stock-predictor-lstm/actions/workflows/ci.yml/badge.svg)](https://github.com/AnasBabari/stock-predictor-lstm/actions) [![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/) [![FastAPI 0.115+](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/) [![React 18.3](https://img.shields.io/badge/React-18.3.1-61DAFB?logo=react&logoColor=black)](https://react.dev/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

StockLSTM is a React/FastAPI research application for market-volatility forecasting. The active product path is a lightweight, causal statistical service: Render validates market history, computes transparent baselines, and returns a dated volatility cone. It does not train models or write model artifacts. Offline research can benchmark Ridge, ElasticNet, boosting, and a compact PyTorch LSTM on the same leakage-safe target before any learned model is promoted.

> [!WARNING]
> Educational project only. Forecasts, indicators, and sentiment are not financial advice.

> [!NOTE]
> GitHub Actions runs locked dependency checks, Ruff, Mypy, Bandit, pip-audit, API-documentation drift checks, backend and frontend tests/builds, and Docker Compose smoke tests. The local verification workflow may intentionally skip Bandit, but the CI gate remains active.

> [!NOTE]
> Normal forecasts are explicitly labelled `baseline` and carry `metric_source = baseline_definition`. They are uncertainty estimates, not learned price-direction claims. The legacy signed-release route remains fail-closed and is not required by the active route.

## Live Demo

- Frontend: <https://stock-predictor-lstm-two.vercel.app>
- Backend API: <https://stock-predictor-lstm.onrender.com>
- Swagger UI: <https://stock-predictor-lstm.onrender.com/docs>
- OpenAPI schema: <https://stock-predictor-lstm.onrender.com/openapi.json>

## Product Tour

| Price forecast dashboard | Direction forecast |
| --- | --- |
| ![Price forecast dashboard with historical and predicted prices.](assets/dashboard.png) | ![Direction forecast dashboard with model analysis.](assets/prediction.png) |

- Search for a validated ticker and select a supported 1-, 3-, 5-, 7-, 14-, or 30-session horizon.
- The active path calls `GET /api/v1/volatility/forecast`; choose a supported horizon and one of the causal persistence, rolling-mean, EWMA, or log-HAR baselines.
- The evidence card reports the true metric source (`baseline_definition`), target definition, annualised volatility, snapshot identity, and feature schema. It does not label a volatility forecast as price-direction accuracy.
- The historical `/api/v2/forecast` route remains available for signed-release compatibility and returns a sanitized abstention when a release is missing, stale, incompatible, tampered with, or not certified for the requested horizon.
- Save results to browser-local watchlists/history and export PNG, CSV, or complete ZIP analyses.

## Architecture and data contract

The active serving snapshot uses **Deployable Schema v5**: 26 causal stationary features covering return structure, realised-volatility proxies, liquidity, and market microstructure. Every row uses information available at or before its origin. A 60-session window, exchange calendar, deterministic snapshot fingerprint, and exact feature ordering are returned with the baseline response. The simplified offline benchmark additionally exposes a 22-session single-symbol feature sequence for transparent model comparisons.

The historical research champion family is a residual temporal-convolutional ensemble with frozen econometric volatility baselines. It remains available for reproduction, but is not on the active request path. V11.2 experimentally added a Student-t return-location/variance distribution, but its one-shot reserve was opened and the candidate failed; that generation is permanently `INVALIDATED_OPENED`.

The GPU-trained v7 development candidate remains unsigned while its genuinely future reserve matures. When 252 target-complete origins after 2026-08-27 are available, `scripts/certify_prospective_volatility_candidate.py` verifies the frozen three-seed artifact, immutable panel prefix, NMM/MSFT transfer coverage, and every locked gate before creating a release-role candidate. Interrupted post-pass materialization can be recovered without reopening the reserve through `scripts/materialize_prospective_certification.py`; failed or partial evidence remains non-releasable.

`GET /api/v1/training-data?ticker=MSFT` remains a bounded browser/research snapshot. `GET /api/v1/volatility/forecast?ticker=MSFT&horizon=7&model=har_rv` is the active product interface; it returns a causal baseline cone, future trading dates, target/evidence metadata, and a deterministic `snapshot_id`. Neither endpoint trains a model or accepts client-supplied features. See [docs/VOLATILITY_FORECASTING.md](docs/VOLATILITY_FORECASTING.md).

Offline comparison metrics use identical snapshots and a chronological 70/15/15 split with an H-session embargo. Volatility is scored primarily with QLIKE plus MAE, MSE, RMSE, R², calibration, and interval coverage. The validation partition selects a model; the untouched test partition is reported once and never drives selection. A learned candidate is not labelled production-ready until it beats the matched baseline on the same target and snapshot.

> [!IMPORTANT]
> **How to read the numbers.** A baseline response is a live uncertainty estimate and reports `baseline_definition`, not an accuracy score. Offline benchmark reports state their snapshot, target, split, embargo, model, and metric source. Historical development and certification reports remain archival and are never presented as active product evidence.

The compatibility `/api/v1/predict` and `/api/v1/predict/direction` endpoints remain available for old clients. Their response metadata always identifies `server_disabled_fallback`; they are persistence/base-rate results and never learned forecasts. `/models` reports the active `volatility_forecasting.status = available` contract, while `server_models.status = disabled`, `browser_training.status = disabled`, and `model_storage.required = false` make the train-free serving boundary explicit.

## Legacy signed global-volatility serving

`GET /api/v2/forecast?ticker=MSFT&horizon=7` is the historical fail-closed interface for one verified global ONNX ensemble across supported tickers. It is retained for compatibility and archival research; the active frontend uses `/api/v1/volatility/forecast` instead. An eligible future release must still pass the externally evidenced gate in [docs/FREE_CERTIFICATION_STACK.md](docs/FREE_CERTIFICATION_STACK.md), retain genuine Cosign verification of the exact frozen evidence manifest, and carry the existing Ed25519 runtime-bundle signature.

The response contains dated p05–p95 paths, expected annualized volatility, snapshot id, model id, certified horizon evidence, and the signed `metric_source`. When `certified_heads.return_distribution` is true it also contains the Student-t expected cumulative return, return-distribution variance, and a learned p50 median path; otherwise p50 remains the disclosed zero-location assumption and the product does not display it as a model forecast. If the release or requested horizon is unavailable, the endpoint returns a structured 503 abstention. `/ready` can enforce this with `VOLATILITY_SERVING_REQUIRED=true`; `/models` reports the verified model id and certified horizons.

On an ephemeral host, `scripts/package_volatility_release.py` deterministically packages an already-signed release. Render can download that immutable HTTPS ZIP using `VOLATILITY_RELEASE_ARCHIVE_URL` plus its mandatory `VOLATILITY_RELEASE_ARCHIVE_SHA256`; the backend then verifies the archive digest, safe extraction paths, Ed25519 manifest, per-member checksums, and ONNX runtime contract before serving. A local `VOLATILITY_RELEASE_DIR` remains supported for Compose and disk-backed deployments. Neither source may be configured from failed or development-only evidence.

The legacy per-ticker server-bundle routes are disabled for production. Their optional offline training/registry code remains isolated for migration and retention work, but it is not reachable from the global production request path.

## News and event context

Live Yahoo Finance headlines remain context-only for the legacy v1 paths. Historical GDELT event data is available as an immutable, point-in-time research snapshot with topic exposures, decay/coverage metadata, checksums, and explicit provider archive gaps. Its matched ablation did not demonstrate incremental value, so news is excluded from v7. It may return only in a separately preregistered future cycle that demonstrates incremental QLIKE/coverage value on identical origins and then clears locked certification. For that future cycle, the certified-serving path (`/api/v2/forecast`) already supports a live news feature provider (`VOLATILITY_NEWS_PROVIDER_ENABLED=true`): a news-certified signed release is then served with a schema-exact, causally aggregated live news vector, and it still abstains (503) whenever the provider is disabled, the provider fails, or the certified schema demands features the live provider cannot honestly reproduce.


## Historical global-model pipeline

The historical global-model pipeline builds immutable snapshots, evaluates econometric and neural challengers on CUDA, opens a locked holdout only after the methodology gate, exports CPU-parity ONNX members, and signs only an overall passing release. See [docs/GLOBAL_MODELS.md](docs/GLOBAL_MODELS.md) for that archived contract. The active simplified path is documented in [docs/VOLATILITY_FORECASTING.md](docs/VOLATILITY_FORECASTING.md) and does not require certification bureaucracy for a baseline response.

### V11.2 numeric PIT64 development

> [!CAUTION]
> V11.2 is an archived research generation, not an active certification path. Its reserve was opened once and the frozen candidate failed. Dataset preparation, certification, release assembly, and runtime loading now reject V11.2 unconditionally. A future attempt must use a new protocol, new externally evidenced inputs, and a new externally controlled reserve.

The additive V11.2 protocol is a numeric-only, pre-holdout development cycle. It expands the panel to exactly 64 audited point-in-time securities, uses a session-grouped chronological 70/15/15 split, and selects a separately gated route for each 1-, 3-, 5-, and 7-session horizon. Epoch zero of the neural residual model is evaluated before any optimizer update and remains a valid HAR-equivalent fallback. Loss uncertainty is estimated by contiguous 20-session block bootstrap with Holm correction; stock-origin observations and unique sessions are reported separately. A learned route must also be non-worse on QLIKE and keep central 80% Student-t coverage within the preregistered 0.65–0.95 calibration band. Historical news is deliberately disabled in V11.2 and reserved for an independent V12 protocol.

The archived V11.2 workflow wrote train/validation files separately from an AES-256-GCM encrypted final holdout. See [docs/VOLATILITY_V11_2.md](docs/VOLATILITY_V11_2.md) for the historical protocol and the permanent retirement boundary.

## Volatility v9 (preregistered)

v9 freezes the research question **before** any v9 model is trained, so it cannot be revised after an inconvenient result appears. The machine-readable protocol is [`configs/volatility_v9_protocol.json`](configs/volatility_v9_protocol.json) and the reasoning is in [docs/VOLATILITY_V9_PREREGISTRATION.md](docs/VOLATILITY_V9_PREREGISTRATION.md). Primary metric is QLIKE (`qlike_losses(forecast, realized)`); a candidate must show skill at **every** required horizon (1, 3, 5, 7) — a losing horizon cannot be hidden inside an average.

> [!IMPORTANT]
> **v9 has no certifiable data yet.** The point-in-time universe is a development reconstruction and the market panel is a `yfinance` development cache. Neither is certification-eligible. Every v9 artifact produced while this is true carries `evidence_role=development_diagnostic_only` and `certification_eligible=false`. No v9 number may be cited as evidence of forecasting skill on real markets.

## Test data contains no market data

`data/fixtures/` holds a **synthetic, generated** regression fixture used to keep the CSCO code path deterministic in CI. It is not a market dataset, it was derived from no real prices, and it is dedicated to the public domain under CC0-1.0. See [data/fixtures/README.md](data/fixtures/README.md).

Every CLI run against it prints a banner stating that the run is a synthetic software regression and says nothing about forecasting skill. The fixture is verified by a SHA-256 pinned **in source code** before any parsing happens, so a modified fixture fails closed.

## Docker Compose

```bash
git clone https://github.com/AnasBabari/stock-predictor-lstm.git
cd stock-predictor-lstm
docker compose build
docker compose up
```

Open <http://localhost:5500>. Nginx serves the frontend on port 5500 and proxies `/api/` to the lightweight FastAPI service. Without a mounted signed release, `/api/v2/forecast` intentionally returns an explicit 503; mount a verified bundle and public key for serving tests.

> [!NOTE]
> **Single-process by design.** Rate-limit buckets, caches, and the work coordinator are process-local, and both deployment targets (Render, Compose) run exactly one API worker. Scaling to multiple replicas would require external coordination and is intentionally out of scope.

## Local development

Requires Python 3.11+ and Node.js `>=20.19.0 <21` or `>=22.12.0` with npm. The backend reads `.env` from its working directory; copy the example to `backend/.env`.

```powershell
cd backend
uv sync --project . --frozen --no-dev
uv run --project . uvicorn api:app --reload --port 8000
```

```bash
cd frontend
npm ci
npm run dev
```
