# StockLSTM - AI Stock Price Predictor

[![CI](https://github.com/AnasBabari/stock-predictor-lstm/actions/workflows/ci.yml/badge.svg)](https://github.com/AnasBabari/stock-predictor-lstm/actions) [![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/) [![FastAPI 0.115+](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/) [![React 18.3](https://img.shields.io/badge/React-18.3.1-61DAFB?logo=react&logoColor=black)](https://react.dev/) [![TensorFlow 2.16+](https://img.shields.io/badge/TensorFlow-2.16+-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Render](https://img.shields.io/badge/Render-46E3B7?logo=render&logoColor=white)](https://stock-predictor-lstm.onrender.com) [![Vercel](https://img.shields.io/badge/Vercel-000000?logo=vercel&logoColor=white)](https://stock-predictor-lstm-two.vercel.app) [![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)](https://github.com/AnasBabari/stock-predictor-lstm/pkgs/container/stock-predictor-lstm)

StockLSTM is a React and FastAPI stock-forecasting application. It retrieves Yahoo Finance history, engineers market features, and offers an interactive Chart.js dashboard plus a REST API. The default prepared artifacts use an LSTM for closing-price forecasts and a BiLSTM with temporal attention for up/down probabilities. GRU and BiLSTM-attention regression architectures are available as operator-selected comparison candidates.

> [!WARNING]
> Educational project only. Forecasts, indicators, and sentiment are not financial advice.

> [!NOTE]
> **Engineering Quality:** [GitHub Actions](https://github.com/AnasBabari/stock-predictor-lstm/actions) runs locked dependency checks, Ruff, Mypy, Bandit, pip-audit, API-documentation drift checks, backend and frontend tests/builds, and Docker Compose API smoke tests.

## Live Demo

- Frontend: https://stock-predictor-lstm-two.vercel.app
- Backend API: https://stock-predictor-lstm.onrender.com
- Swagger UI: https://stock-predictor-lstm.onrender.com/docs
- OpenAPI schema: https://stock-predictor-lstm.onrender.com/openapi.json

## Product Tour

| Price forecast dashboard | Direction forecast |
| --- | --- |
| ![Price forecast dashboard with historical and predicted prices.](assets/dashboard.png) | ![Direction forecast dashboard with model analysis.](assets/prediction.png) |

- Search for a ticker or enter an exact symbol, then choose a 1–30 trading-day horizon.
- Run a price forecast with historical/predicted prices, or a direction forecast with up/down probabilities.
- Attention weights and headline sentiment are supporting context, not financial advice; sentiment is not a model input.
- Save price results to a browser-local watchlist, revisit prediction history, and export PNG, CSV, or complete ZIP analyses.

Public requests never train models. They prefer fresh, operator-prepared artifacts and disclose the active engine. If an artifact is unavailable, the API may return a labelled persistence or base-rate baseline; it never silently presents a baseline as a learned model.

## Data, Models, and Runtime

The production feature pipeline has **22 ordered features**: 5 OHLCV fields, 9 technical values (SMA, EMA, RSI, MACD and signal, Bollinger bands, ATR, OBV), 4 market-context returns (SPY, QQQ, VIX, TNX), and 4 cyclic calendar values. Models receive a 60-session window. Artifacts have a 30-day maximum output width and return the requested 1-30-day slice.

Validation supports `expanding` and `rolling` walk-forward strategies; defaults use five expanding folds and publish untouched out-of-fold metrics when available. Forecasts are scored as explicit origin–horizon pairs, with per-horizon and pooled MAE, MSE, RMSE, MAPE, bias, R², directional accuracy, MASE, RMSSE, and error relative to persistence. Direction models also report balanced accuracy, Brier score, and log loss.

The offline experiment framework compares persistence, drift, ridge, and histogram-gradient-boosting baselines on identical purged folds. A candidate is not promoted unless it improves pooled MAE and RMSE over persistence by at least 5%, remains stable across folds, and satisfies scaled-error limits. See [model evaluation](docs/MODEL_EVALUATION.md) for the complete contract.

Exchange calendars support selected international suffixes and 24/7 crypto pairs; unknown dotted symbols report an NYSE fallback. Yahoo Finance headline sentiment uses VADER with explicit finance-phrase adjustments and observable coverage metadata. Live sentiment remains response context only. Historical sentiment features require timestamped articles, exclude future and untimestamped records, and must pass the same controlled ablation and promotion gate before entering a production model.

FastAPI validates input, rate-limits public predictions, caches responses, and keeps model training outside the public HTTP path. Versioned Keras artifacts, JSON scalers, metadata, and evaluation data are validated before activation. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for concurrency, locks, quotas, integrity, and readiness behavior.

## Docker Compose

```bash
git clone https://github.com/AnasBabari/stock-predictor-lstm.git
cd stock-predictor-lstm
docker compose build
docker compose run --rm backend python pretrain.py --ticker AAPL
docker compose up
```

The example explicitly approves `AAPL`; repeat `--ticker` for other symbols. No ticker is prepared by default. Open `http://localhost:5500`. Nginx serves the frontend on `5500`, proxies `/api/` to FastAPI, waits for backend readiness, and persists models in `model_cache`. Use `docker compose ps`, `docker compose logs -f`, and `docker compose down`; add `--volumes` only to intentionally remove cached models.

The default pretraining command prepares `lstm` and `bilstm_attention_direction`. Operators can prepare comparison candidates explicitly:

```bash
docker compose run --rm backend python pretrain.py --ticker AAPL --model-type gru
docker compose run --rm backend python pretrain.py --ticker AAPL --model-type bilstm_attention_regression
```

## Local Development

Requires Python 3.11+ and Node.js `>=20.19.0 <21` or `>=22.12.0` with npm. The backend reads `.env` from its working directory; copy the example to `backend/.env`.

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
Copy-Item ..\.env.example .env
pip install -r requirements.txt -r requirements-dev.txt
uvicorn api:app --reload --port 8000
```

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
cp ../.env.example .env
pip install -r requirements.txt -r requirements-dev.txt
uvicorn api:app --reload --port 8000
```

```bash
cd frontend
npm ci
npm run dev
```

Vite runs at `http://localhost:5500` and proxies `/api` to `http://127.0.0.1:8000`. Local Swagger and OpenAPI are `http://127.0.0.1:8000/docs` and `http://127.0.0.1:8000/openapi.json`. CI-aligned dependency setup is `uv sync --project backend --frozen`.

## Reproducible Model Benchmark

Run the offline benchmark separately from the public API:

```powershell
uv run --project backend python backend/benchmark.py `
  --ticker AAPL `
  --horizons 1,5,20 `
  --feature-sets price,ohlcv,ohlcv_market,ohlcv_technical_market `
  --output reports/aapl.json
```

The JSON report records the market-data snapshot identifier, date boundaries, target type, purge gap, fold indices, per-horizon metrics, pooled metrics, and promotion decision for each model and feature group. Generated reports are operator artifacts and should not be committed.

The benchmark evaluates deterministic baselines and can consume verified frozen snapshots plus operator-selected neural candidates. It is evidence, not an automatic deployment action. Promotion is an explicit lifecycle command: `uv run --project backend python backend/promote.py --registry <dir> --source <candidate> --manifest <manifest.json>`; add `--private-key` and `--public-key` when signed evidence is required. Reports should always name the snapshot, date range, folds, seed, and feature set used for MAE/RMSE comparisons.
## Configuration

See [.env.example](.env.example) for the full list and `backend/config.py` for defaults. Key groups are data/model (`HISTORICAL_YEARS`, `WINDOW_SIZE`, `LSTM_UNITS`, `EPOCHS`, `BATCH_SIZE`, `TRAIN_SPLIT`), forecast/artifact (`MODEL_DIR`, `MODEL_MAX_AGE_DAYS`, `DEFAULT_FORECAST_DAYS`, `MAX_FORECAST_DAYS`), cache, capacity, and storage settings. Defaults include a 60-session window, 7-day default forecast, 30-day maximum, 7-day artifact age, and `saved_models` directory.

`ALLOWED_ORIGINS` is a JSON-style list, for example `["http://localhost:5500"]`. `CORS_ORIGIN` appends one production frontend origin; Compose overrides local origins for `5500`. Nested validation uses `VALIDATION__...` settings for method, folds, minimum training size, horizon, gap, seed, and deterministic mode. Production should explicitly configure CORS and persistent model storage. `TRUSTED_PROXY_IPS` is an exact JSON IP list; leave it empty unless the direct peer is a controlled proxy.

## API

[docs/API.md](docs/API.md) is the canonical reference.

| Endpoint | Purpose |
| --- | --- |
| `GET /health`, `GET /ready`, `GET /models` | Liveness, readiness, and active artifact manifest. |
| `GET /api/v1/search`, `GET /api/v1/info` | Ticker discovery and fundamentals. |
| `GET /api/v1/predict` | LSTM price forecast and metrics. |
| `GET /api/v1/predict/direction` | Direction, probability, attention, sentiment, and metrics. |
| `GET /api/v1/diagnostics/{ticker}` | Persisted walk-forward diagnostics. |`n| `GET /api/v1/model-performance/{ticker}` | Active engine and persisted performance evidence. |

Prediction requests accept tickers matching `[A-Z0-9.\-]{1,12}` and 1-30 days. `429` indicates rate limiting; `503` can indicate an unavailable prepared artifact, capacity, timeout, readiness, or retryable market-data failure.

## Deployment, Testing, and Contribution

The frontend is hosted on Vercel at https://stock-predictor-lstm-two.vercel.app; [frontend/vercel.json](frontend/vercel.json) provides SPA rewrites. The Render API at https://stock-predictor-lstm.onrender.com uses [render.yaml](render.yaml) with Python 3.11.9 and persistent `/app/saved_models`; set `CORS_ORIGIN` to the exact public frontend origin. Render runs `backend/render_start.py` before Uvicorn starts. It prepares missing artifacts on the mounted persistent disk for the approved `RENDER_PRETRAIN_TICKERS` universe (default `AAPL,MSFT,TSLA`) and then starts the API. Add or change approved tickers in Render configuration when expanding the hosted universe; do not move training into public request handling.

```bash
uv run --project backend pytest backend/tests -q --cov=backend --cov-report=term-missing --cov-fail-under=70
uv run --project backend python scripts/check_api_docs.py
cd frontend
npm run test:run
npm run build
```

CI checks dependency locks, Ruff, Mypy, Bandit, pip-audit, API documentation, backend tests/coverage, frontend tests/build, policy checks, and a Compose build, health, and frontend-to-backend API smoke test. It does not publish containers or deploy. Do not commit `.env`, generated models, downloaded data, logs, coverage data, or editor files.

Before serving forecasts, run `uv run --project backend python backend/pretrain.py --ticker SYMBOL` against persistent model storage and verify `/models`. The command exits non-zero if preparation fails; it never reports a stale fallback artifact as ready. Schedule the same explicit command at least daily to stay within the default seven-day artifact age. For a `503`, verify the requested artifact is fresh, then check storage and upstream status. For local API failures, ensure FastAPI is on `8000`, Vite is on `5500`, and frontend calls use `/api`.

## License

MIT License - see [LICENSE](LICENSE) for details.
