# StockLSTM — global volatility forecasting

[![CI](https://github.com/AnasBabari/stock-predictor-lstm/actions/workflows/ci.yml/badge.svg)](https://github.com/AnasBabari/stock-predictor-lstm/actions) [![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/) [![FastAPI 0.115+](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/) [![React 18.3](https://img.shields.io/badge/React-18.3.1-61DAFB?logo=react&logoColor=black)](https://react.dev/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

StockLSTM is a React/FastAPI research application for market-volatility forecasting. Its production architecture accepts only a signed, CPU-served global TCN ensemble trained and certified offline on the RTX workstation. No candidate is currently certified: the v6 result was strictly rejected and v7 is prospective development work. Render supplies validated market history and calendars; it does not train models or accept model weights. Vercel renders a forecast only when the release contract verifies.

> [!WARNING]
> Educational project only. Forecasts, indicators, and sentiment are not financial advice.

> [!NOTE]
> GitHub Actions runs locked dependency checks, Ruff, Mypy, Bandit, pip-audit, API-documentation drift checks, backend and frontend tests/builds, and Docker Compose smoke tests. The local verification workflow may intentionally skip Bandit, but the CI gate remains active.

> [!NOTE]
> A forecast is only displayed as a learned production result when the signed release, checksum, schema, horizon certification, and metric provenance all verify. Otherwise the API abstains with `503`; it does not silently substitute a flat line and call it a model.

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
- The production path calls `GET /api/v2/forecast`; when a verified release exists, the dashboard renders only its certified volatility cone. It does not draw the unchanged-close mathematical center as a learned price forecast.
- The evidence card reports the true metric source (`locked_purged_walk_forward`), QLIKE, coverage, release identity, snapshot identity, and certified horizon. It does not label a volatility forecast as price-direction accuracy.
- The API returns a sanitized abstention when a release is missing, stale, incompatible, tampered with, or not certified for the requested horizon.
- Save results to browser-local watchlists/history and export PNG, CSV, or complete ZIP analyses.

## Architecture and data contract

The serving snapshot uses **Deployable Schema v5**: 26 causal stationary features covering return structure, realized-volatility proxies, liquidity, and market microstructure. Every row uses information available at or before its origin. A 60-session window, exchange calendar, deterministic snapshot fingerprint, and exact feature ordering are bound into the signed release contract.

The research champion family is a residual temporal-convolutional ensemble with frozen econometric volatility baselines. It predicts conditional variance; the API can derive a transparent daily cone only from a future certified release. The v6 locked certification failed overall, so strict rejection prevents materializing or serving that candidate or any individually passing horizon. With no verified signed release, the production API abstains. Any p50 shown by a future volatility-only release remains an explicitly unchanged-close location baseline until a separately certified return-location head clears its own gates.

The GPU-trained v7 development candidate remains unsigned while its genuinely future reserve matures. When 252 target-complete origins after 2026-08-27 are available, `scripts/certify_prospective_volatility_candidate.py` verifies the frozen three-seed artifact, immutable panel prefix, NMM/MSFT transfer coverage, and every locked gate before creating a release-role candidate. Interrupted post-pass materialization can be recovered without reopening the reserve through `scripts/materialize_prospective_certification.py`; failed or partial evidence remains non-releasable.

`GET /api/v1/training-data?ticker=MSFT` remains a bounded diagnostic/research snapshot. It returns validated features, historical closes, strictly increasing backend-generated future trading dates, schema metadata, and a deterministic `snapshot_id`; it does not train a model or accept client-supplied features.

Production metrics are computed offline on identical market snapshots using calendar-aligned expanding purged walk-forward folds. Volatility is scored primarily with QLIKE, log-variance error, calibration, and interval coverage; price-location and direction heads are separate gates. A development candidate is not called certified until one untouched holdout and the signed-release verification pass. A failed locked result is final for that candidate and reserve: neither passing horizons nor old holdout rows are recycled.

> [!IMPORTANT]
> **How to read the numbers.** `locked_purged_walk_forward` describes untouched locked evidence for a selected volatility head only after overall certification passes. In such a release, p50 is an unchanged-close baseline and the uncertainty cone is learned. There is no current certified release; development reports, browser experiments, failed locked evidence, and final refits are never presented as production metrics.

The compatibility `/api/v1/predict` and `/api/v1/predict/direction` endpoints remain available for old clients. Their response metadata always identifies `server_disabled_fallback`; they are persistence/base-rate results and never learned forecasts. `/models` reports `server_models.status = disabled` and `browser_training.status = disabled` in the production contract.

## Signed global-volatility serving

`GET /api/v2/forecast?ticker=MSFT&horizon=7` is the fail-closed interface for one verified global ONNX ensemble across supported tickers. An eligible future release must be trained and certified offline on the RTX workstation, signed with Ed25519, and mounted as an immutable directory. It is never trained in a public request and no model weights are uploaded by the browser.

The response contains dated p05–p95 volatility paths, expected annualized volatility, snapshot id, model id, certified horizon evidence, and `metric_source = locked_purged_walk_forward`. The API retains p50 as the disclosed zero-location distribution assumption, but the product does not display it as a forecast line or a 0% model prediction. If the release or requested horizon is unavailable, the endpoint returns a structured 503 abstention. `/ready` can enforce this with `VOLATILITY_SERVING_REQUIRED=true`; `/models` reports the verified model id and certified horizons.

On an ephemeral host, `scripts/package_volatility_release.py` deterministically packages an already-signed release. Render can download that immutable HTTPS ZIP using `VOLATILITY_RELEASE_ARCHIVE_URL` plus its mandatory `VOLATILITY_RELEASE_ARCHIVE_SHA256`; the backend then verifies the archive digest, safe extraction paths, Ed25519 manifest, per-member checksums, and ONNX runtime contract before serving. A local `VOLATILITY_RELEASE_DIR` remains supported for Compose and disk-backed deployments. Neither source may be configured from failed or development-only evidence.

The legacy per-ticker server-bundle routes are disabled for production. Their optional offline training/registry code remains isolated for migration and retention work, but it is not reachable from the global production request path.

## News and event context

Live Yahoo Finance headlines remain context-only for the legacy v1 paths. Historical GDELT event data is available as an immutable, point-in-time research snapshot with topic exposures, decay/coverage metadata, checksums, and explicit provider archive gaps. Its matched ablation did not demonstrate incremental value, so news is excluded from v7. It may return only in a separately preregistered future cycle that demonstrates incremental QLIKE/coverage value on identical origins and then clears locked certification. For that future cycle, the certified-serving path (`/api/v2/forecast`) already supports a live news feature provider (`VOLATILITY_NEWS_PROVIDER_ENABLED=true`): a news-certified signed release is then served with a schema-exact, causally aggregated live news vector, and it still abstains (503) whenever the provider is disabled, the provider fails, or the certified schema demands features the live provider cannot honestly reproduce.


## Global model pipeline

The offline global-model pipeline builds immutable snapshots, evaluates econometric and neural challengers on CUDA, opens a locked holdout only after the methodology gate, exports CPU-parity ONNX members, and signs only an overall passing release. See [docs/GLOBAL_MODELS.md](docs/GLOBAL_MODELS.md) for the full contract and [docs/VOLATILITY_V7_PREREGISTRATION.md](docs/VOLATILITY_V7_PREREGISTRATION.md) for the fresh cycle created after v6 strict rejection. Legacy browser TFJS remains only in the isolated methodology build; the normal Vercel production bundle compiles it out and CI verifies that no worker or TFJS model code is shipped.

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
