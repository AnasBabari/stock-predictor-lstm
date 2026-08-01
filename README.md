# StockLSTM - Browser-trained stock forecasting

[![CI](https://github.com/AnasBabari/stock-predictor-lstm/actions/workflows/ci.yml/badge.svg)](https://github.com/AnasBabari/stock-predictor-lstm/actions) [![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/) [![FastAPI 0.115+](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/) [![React 18.3](https://img.shields.io/badge/React-18.3.1-61DAFB?logo=react&logoColor=black)](https://react.dev/) [![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-4.22-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/js) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

StockLSTM is a React and FastAPI stock-forecasting application. The FastAPI service retrieves Yahoo Finance history and returns a validated feature snapshot. A compact TensorFlow.js LSTM trains in each user's browser, runs in a Web Worker, and stores that user's model in IndexedDB. Render supplies data and calendar dates; it does not train models, load Keras artifacts, or retain model weights.

> [!WARNING]
> Educational project only. Forecasts, indicators, and sentiment are not financial advice.

> [!NOTE]
> GitHub Actions runs locked dependency checks, Ruff, Mypy, Bandit, pip-audit, API-documentation drift checks, backend and frontend tests/builds, and Docker Compose smoke tests. The local verification workflow may intentionally skip Bandit, but the CI gate remains active.

## Live Demo

- Frontend: <https://stock-predictor-lstm-two.vercel.app>
- Backend API: <https://stock-predictor-lstm.onrender.com>
- Swagger UI: <https://stock-predictor-lstm.onrender.com/docs>
- OpenAPI schema: <https://stock-predictor-lstm.onrender.com/openapi.json>

## Product Tour

| Price forecast dashboard | Direction forecast |
| --- | --- |
| ![Price forecast dashboard with historical and predicted prices.](assets/dashboard.png) | ![Direction forecast dashboard with model analysis.](assets/prediction.png) |

- Search for a ticker or enter an exact symbol, then choose a 1–30 trading-day horizon.
- The first forecast downloads a bounded feature snapshot and trains a local model. Progress, the worker backend, holdout metrics, and cache status are shown in the UI.
- Reloading the page can load the same ticker/type model from IndexedDB. Price and direction models are separate, per browser, per ticker, and never uploaded.
- If workers, WebGL/CPU TensorFlow.js, or local storage are unavailable, the UI explicitly labels the deterministic persistence/base-rate fallback.
- Save results to browser-local watchlists/history and export PNG, CSV, or complete ZIP analyses.

## Architecture and data contract

The production pipeline has **22 ordered features**: OHLCV, technical indicators, market-context returns, and cyclical calendar values. Browser training uses a 60-session input window, a 30-step model output, an 80% training split, a train-only min/max scaler, and a `forecast_days - 1` purge at the train/holdout boundary. The price target is scaled `Close`; the direction target is a positive future log return.

`GET /api/v1/training-data?ticker=MSFT` returns the validated feature matrix, historical closes, backend-generated future trading dates, feature schema version, and a deterministic `snapshot_id`. The frontend invalidates a cached model when that snapshot, schema, feature list, window, output width, or implementation version changes. The service bounds the snapshot to the configured historical period and 2,000 rows, returns finite numeric values only, and applies a dedicated 10-per-minute limit.

The compact browser model is:

```text
Input [60, 22] -> LSTM(32, sequences) -> Dropout(.20) -> LSTM(16)
                -> Dense(16, relu) -> Dense(30)
```

Price uses linear output/MSE; direction uses sigmoid output/binary cross-entropy. Training is capped at 12 epochs, batch size 32, no shuffle, validation early stopping after three unimproved epochs, and a cancel path that disposes tensors and terminates the worker request.

Metrics are labelled `browser_purged_holdout`: MAE, MSE, RMSE, MAPE, R², relative MAE/RMSE versus persistence, and direction accuracy, precision, recall, F1, balanced accuracy, Brier score, and naive-baseline accuracy. They are not the historical five-fold walk-forward metrics unless a separate offline benchmark is run.

The compatibility `/api/v1/predict` and `/api/v1/predict/direction` endpoints remain available during migration. Their response metadata always identifies `server_disabled_fallback`; they are not used for learned browser forecasts. `/models` reports `server_models.status = disabled` and `browser_training.status = available`.

## News and sentiment

Live Yahoo Finance headlines are still fetched for the direction response as context (`sentiment.status`/coverage metadata). They are deliberately not sent in the 22-feature browser matrix, so headline sentiment cannot silently become a learned input. Historical news experiments remain offline-only and must use timestamped articles, leakage-safe alignment, controlled ablations, and the same purged holdout/promotion gates as price features. This separation keeps the browser model reproducible and makes the current news limitation explicit instead of claiming that sentiment improves the forecast.

## Docker Compose

```bash
git clone https://github.com/AnasBabari/stock-predictor-lstm.git
cd stock-predictor-lstm
docker compose build
docker compose up
```

Open <http://localhost:5500>. Nginx serves the frontend on port 5500 and proxies `/api/` to the lightweight FastAPI service. The backend binds to loopback for local access and the internal Compose network for the proxy. No model volume is required. The same browser worker/IndexedDB flow is used locally and on Render.

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

Vite runs at <http://localhost:5500> and proxies `/api` to <http://127.0.0.1:8000>. CI-aligned production setup is `uv sync --project backend --frozen --no-dev`; it installs no TensorFlow package. The offline research/training environment is opt-in:

```bash
uv sync --project backend --frozen --group training --group dev
uv run --project backend python backend/pretrain.py --ticker AAPL --model-type lstm
```

Offline training may write local artifacts for research and benchmark work. It is not called by the production API, Render start command, or browser path.

## Reproducible offline benchmark

Run the benchmark separately from the public API:

```powershell
uv run --project backend python backend/benchmark.py `
  --ticker AAPL `
  --horizons 1,5,20 `
  --feature-sets price,ohlcv,ohlcv_market,ohlcv_technical_market `
  --output reports/aapl.json
```

Reports record the snapshot identifier, date boundaries, target type, purge gap, folds, seed, feature set, per-horizon metrics, pooled metrics, and promotion decision. They are evidence, not an automatic deployment action, and generated reports should not be committed.

## Configuration

See [.env.example](.env.example) and [backend/config.py](backend/config.py). Production settings cover data, CORS, rate limiting, request queues, upstream circuit protection, and trusted proxy addresses. Model-directory, artifact-age, quota, and pretraining settings are retained only for the opt-in offline trainer; they are not present in `render.yaml` and are not read by the production API.

`ALLOWED_ORIGINS` is a JSON-style list, for example `["http://localhost:5500"]`. `CORS_ORIGIN` appends the production frontend origin. `TRUSTED_PROXY_IPS` is an exact JSON IP list; leave it empty unless the direct peer is a controlled proxy. Browser model storage is separate from existing localStorage theme/watchlist/history data. The frontend build flag VITE_BROWSER_TRAINING_ENABLED=false provides an emergency rollback to the explicitly labelled compatibility baseline.

## API

[docs/API.md](docs/API.md) is the canonical reference.

| Endpoint | Purpose |
| --- | --- |
| `GET /health`, `GET /ready`, `GET /models` | Liveness, readiness, and browser/server model status. |
| `GET /api/v1/search`, `GET /api/v1/info` | Ticker discovery and fundamentals. |
| `GET /api/v1/training-data?ticker=MSFT` | Validated 22-feature snapshot for browser training. |
| `GET /api/v1/predict` | Compatibility persistence fallback during migration. |
| `GET /api/v1/predict/direction` | Compatibility direction base-rate fallback and context sentiment. |
| `GET /api/v1/diagnostics/{ticker}` | Offline artifact diagnostics, when explicitly available. |
| `GET /api/v1/model-performance/{ticker}` | Browser-training availability and offline evidence status. |

Prediction requests accept tickers matching `[A-Z0-9.\-]{1,12}` and 1–30 days. `429` indicates rate limiting; `503` indicates retryable market-data or service failure. A baseline response is successful but always labelled in metadata.

## Deployment, testing, and contribution

The frontend is hosted on Vercel; [frontend/vercel.json](frontend/vercel.json) provides SPA rewrites. Render uses [render.yaml](render.yaml) with the native Python environment, a TensorFlow-free `pip install -r requirements.txt` build, and a lightweight `python -m uvicorn` command. If the Render service was created manually, copy both commands into its Build Command and Start Command fields (a repository `render.yaml` change does not retroactively overwrite dashboard settings). No persistent disk or model directory is required. Set `CORS_ORIGIN` to the exact Vercel origin. Browser model files remain on each user's device.

```bash
uv run --project backend pytest backend/tests -q --cov=backend --cov-report=term-missing --cov-fail-under=70
uv run --project backend python scripts/check_api_docs.py
cd frontend
npm run test:run
npm run build
```

CI checks dependency locks, Ruff, Mypy, Bandit, pip-audit, API documentation, backend tests/coverage, frontend tests/build, policy checks, and Compose configuration/smoke tests. It does not publish containers or deploy. Do not commit `.env`, generated model artifacts, downloaded data, logs, coverage data, or editor files.

## License

MIT License - see [LICENSE](LICENSE) for details.