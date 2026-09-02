# Stock Volatility Forecasting System

[![CI](https://github.com/AnasBabari/stock-predictor-lstm/actions/workflows/ci.yml/badge.svg)](https://github.com/AnasBabari/stock-predictor-lstm/actions)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18.3-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-grade, empirical equity volatility forecasting platform. The system forecasts future realized volatility over the benchmark-backed horizons ($H \in \{1, 5, 10, 20\}$ sessions), projects Gaussian model-implied price ranges, and maintains an **immutable forward forecast ledger** that records predictions at the forecast origin and settles against actual market outcomes.

---

## Key Features

- **Causal Realized Volatility Targets:** Forecasts forward annualized close-to-close realized volatility $RV(t, H) = \sqrt{\frac{252}{H}\sum_{k=1}^H r_{t+k}^2}$ without future lookahead bias.
- **Empirically Selected Model Policy:** Production serving uses models selected strictly on validation QLIKE loss:
  - **1-Day Horizon:** Causal `GARCH(1,1) MLE` with numerical parameter optimization.
  - **Multi-Day Horizons (5d, 10d, 20d):** `Rolling Mean (60d)` sample standard deviation.
  - **Research & ML Baselines:** PyTorch `SOFTPLUS_VOLATILITY` LSTM, `HAR-RV`, `EWMA (0.94)`, and regularized regressors (`ElasticNet`, `Ridge`).
- **Resilient Market Data Boundary:** Render uses authenticated Alpaca daily bars with explicit adjusted-price semantics, bounded requests, session-aware ephemeral caching, and safe `503` handling. Yahoo is opt-in for local development only.
- **Immutable Forecast Ledger & Track Record:** Public forecasts are read-only previews. A separately authenticated, paced collector is the only API client permitted to create genuine live records. Every live forecast has a deterministic SHA-256 fingerprint covering provider, code, data date, model, origin price, prediction, and scenarios. Settled records cannot be mutated or overwritten. Local/test runs use SQLite; production requires PostgreSQL and fails closed when it is unavailable.
- **Exchange Calendar Synchronization:** Forecast origins and news cutoffs follow the official NYSE trading calendar via `pandas_market_calendars`, failing closed on weekends and holidays.
- **Modern Interactive UI:** React 18 dashboard with volatility scenario-range visualization, live scorecard (MAE, RMSE, QLIKE), and transparent model evaluation rationale.

The public volatility contract accepts only 1, 5, 10, and 20 trading sessions. `model=auto` follows the frozen validation policy (`GARCH(1,1)` at 1 session and `Rolling Mean (60d)` at 5, 10, and 20). The displayed p05–p95 band is a Gaussian model-implied scenario under zero drift; its midpoint is the unchanged latest close, not a point-price prediction, and the nominal 90% coverage is not an empirical confidence interval.

---

## Empirical Benchmark Highlights

Across a diverse benchmark universe of **44 liquid equities and ETFs** across 8 market sectors spanning **2015–2026** (70% train / 15% validation / 15% out-of-sample test with $H$-session purged boundary embargoes):

| Horizon | Selected Model | Test MAE | Test RMSE | Test QLIKE | Bootstrap 95% CI |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **1-Day** | `GARCH(1,1) MLE` | 22.21% | 31.38% | 1.8581 | Baseline Winner |
| **5-Day** | `Rolling Mean (60d)` | 9.42% | 13.08% | 0.7206 | Baseline Winner |
| **10-Day** | `Rolling Mean (60d)` | 8.11% | 11.20% | 0.5707 | Baseline Winner |
| **20-Day** | `Rolling Mean (60d)` | 7.39% | 10.15% | 0.5776 | Baseline Winner |

### Ablation Study & Stopping Rule
- **OHLC Range Features:** Parkinson, Garman-Klass, and Rogers-Satchell intraday range estimators consistently stabilized neural model optimization.
- **Market Context:** Adding leave-self-out SPY and QQQ returns reduced multi-day test error across sector equities.
- **News Sentiment Stopping Rule:** Adding point-in-time financial news sentiment features (VADER lexicon, negative intensity, volume z-scores) failed to yield a statistically significant out-of-sample gain across the 44-asset universe (95% bootstrap CIs crossed zero). Following empirical stopping rules, news features remain offline research-only and are omitted from production forecasting.

---

## Repository Structure

```text
├── backend/                  # FastAPI service & core business logic
│   ├── api.py                # FastAPI application entry point
│   ├── routes/               # Modular REST endpoints (volatility, market, health)
│   ├── services/             # Forecast ledger, snapshot builder, live volatility
│   ├── features/             # Causal market context and feature builders
│   ├── market_data/          # Alpaca/Yahoo adapters, normalization, cache, readiness
│   ├── data_pipeline.py      # Market data ingestion & validation
│   ├── calendars.py          # NYSE session calendar utilities
│   └── tests/                # Comprehensive pytest suite
├── frontend/                 # React 18 + Vite + Tailwind UI
│   ├── src/                  # Components, charts, and API hooks
│   └── tests/                # Vitest unit & integration test suites
├── research/                 # Empirical research & reproducible benchmarks
│   ├── volatility_forecasting/
│   │   └── simple_pipeline.py# Canonical GARCH MLE, HAR, PyTorch LSTM, metrics
│   └── tests/                # Research test suites
├── scripts/                  # Empirical evaluation & E2E verification
│   ├── run_comprehensive_empirical_study.py # 44-asset benchmark runner
│   ├── verify_forecast_ledger_e2e.py        # 8-point ledger integrity simulator
│   ├── migrate_forecast_ledger.py           # explicit SQLite → PostgreSQL migration
│   └── export_forecast_ledger.py            # deterministic live/replay export
├── reports/                  # Versioned benchmark summaries & JSON artifacts
├── docs/                     # Detailed technical documentation
│   ├── methodology.md        # Mathematical targets, metrics, and protocols
│   ├── architecture.md       # Service design, data flow, and immutability
│   └── api.md                # REST API endpoint reference
└── README.md
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+ and npm

### 1. Backend Setup
```powershell
# Navigate to backend directory
cd backend

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt -r requirements-dev.txt

# Start FastAPI development server
uvicorn api:app --reload --port 8000
```
API will be live at `http://127.0.0.1:8000` (interactive OpenAPI docs at `http://127.0.0.1:8000/docs`).

Local development defaults to Yahoo. Production must set `MARKET_DATA_PROVIDER=alpaca`,
`ALPACA_API_KEY_ID`, and `ALPACA_API_SECRET_KEY`; credentials are never sent to the browser.
Before collecting genuine live forecasts, also set `FORECAST_LEDGER_DATABASE_URL`
to a managed PostgreSQL database, `FORECAST_LEDGER_DATABASE_REQUIRED=true`, and
`FORECAST_COLLECTOR_TOKEN` to a high-entropy server-side secret. The token must
never be exposed through a `VITE_*` variable or browser code.
See [Production Market Data](docs/market-data.md) for cache, readiness, ledger migration,
and error semantics.

### 2. Frontend Setup
```powershell
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start Vite development server
npm run dev
```
UI will be live at `http://localhost:5173`.

### 3. Run Tests & Verification
```powershell
# Run backend and research unit tests
pytest -c backend/pyproject.toml backend/tests/ research/tests/

# Run frontend tests
cd frontend && npm test -- --run

# Run full ledger lifecycle simulation
python scripts/verify_forecast_ledger_e2e.py
```

---

## Documentation

- [Methodology & Mathematical Targets](docs/methodology.md)
- [System Architecture & Immutability](docs/architecture.md)
- [REST API Reference](docs/api.md)
- [Production Market Data](docs/market-data.md)
- [Secure Live Forecast Operations](docs/live-forecast-operations.md)
- [Benchmark Reports](reports/)

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
