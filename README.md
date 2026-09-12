# Signal Seven

[![CI](https://github.com/AnasBabari/stock-predictor-lstm/actions/workflows/ci.yml/badge.svg)](https://github.com/AnasBabari/stock-predictor-lstm/actions)

Signal Seven is a causal equity volatility forecasting engine. It estimates **how much stock prices will fluctuate—not whether they will rise or fall**.

The platform is powered by **G3**, a GPU-trained XGBoost panel model that predicts corrections to rolling volatility, alongside empirical intraday range estimators (Parkinson, Garman-Klass, Rogers-Satchell, and Yang-Zhang).

The frontend is a React app hosted on Vercel; the backend is a FastAPI service on Render. Opening the app wakes the backend automatically.

---

## What You Can Do

- **Search Supported Stocks**: 286 liquid US and UK equities across NASDAQ, NYSE, and LSE.
- **Interactive Price Chart**: Inspect historical price action with zoom, pan, crosshair, and time range controls.
- **Expected Volatility Cones**: View zero-drift expected price dispersion bands ($p_{05}$–$p_{95}$) projected across future trading sessions.
- **Multi-Horizon Volatility Outlook**: Compare annualized volatility forecasts for 5, 10, and 20 trading sessions with categorized risk levels.
- **Transparent Model Diagnostics**: Inspect comparisons against 60-session rolling volatility and see explicit fallback disclosures.

---

## How It Works

### Volatility Forecasting Engine (G3)

1. **Market Data**: The backend retrieves adjusted daily open, high, low, close, and volume (OHLCV) bars with strict calendar synchronization via `pandas_market_calendars`.
2. **Range & Structure Features**: The system computes intraday range estimators (Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang), multi-window realized volatilities ($RV_5, RV_{10}, RV_{20}, RV_{60}$), return asymmetry / leverage features, and Heterogeneous Autoregressive (HAR) components.
3. **Causal Inference**: The offline GPU-trained G3 model is served via lightweight ONNX runtime graphs (`g3_h5.onnx`, `g3_h10.onnx`, `g3_h20.onnx`), predicting a multiplicative correction $\exp(\delta)$ to the rolling base volatility.
4. **Fallback Mechanism**: If the learned model cannot be evaluated, the endpoint falls back explicitly to causal statistical baselines (e.g., 60-day rolling standard deviation or GARCH(1,1)).

---

## Honest Limitations

- This is a risk modeling tool and research experiment, **not financial advice**.
- Volatility forecasts estimate price dispersion and uncertainty; they **do not establish a directional price forecast**.
- Expected range cones assume zero drift around the latest close and reflect model-implied scenarios, not guaranteed bounds.
- Historical benchmark outperformance on held-out panels does not guarantee identical future performance in anomalous market regimes.

---

## API

```http
GET /health
GET /api/v1/history?ticker=MSFT
GET /api/v1/volatility/forecast?ticker=MSFT&horizon=5&model=auto
GET /api/v1/volatility/horizons
GET /api/v1/volatility/ledger?ticker=MSFT
```

### Example Response: `/api/v1/volatility/forecast`

```json
{
  "ticker": "MSFT",
  "horizon": 5,
  "current_price": 450.25,
  "forecast": {
    "predicted_volatility": 0.214,
    "expected_annualized_volatility": 0.214,
    "model": "gpu_g3",
    "future_dates": ["2026-09-15", "2026-09-16", "2026-09-17", "2026-09-18", "2026-09-19"],
    "price_quantiles": {
      "p05": [442.1, 439.5, 437.2, 435.1, 433.0],
      "p50": [450.25, 450.25, 450.25, 450.25, 450.25],
      "p95": [458.4, 461.0, 463.3, 465.4, 467.5]
    }
  },
  "evidence": {
    "model_status": "gpu_promoted",
    "risk_level": "Moderate",
    "trailing_annualized_volatility_60d": 0.198
  }
}
```

---

## Local Development

### Prerequisites
- Python 3.11
- Node.js 22.12 or later

### Backend

```powershell
python -m venv .venv
.venv/Scripts/pip install -r backend/requirements.txt -r backend/requirements-dev.txt
cd backend
../.venv/Scripts/uvicorn api:app --reload --port 8000
```

### Frontend

```powershell
cd frontend
npm ci
$env:VITE_API_URL = "http://127.0.0.1:8000"
npm run dev
```

Open `http://localhost:5500`.

---

## Verification

```powershell
# Backend & Research tests
pytest backend/tests research/tests -q -ra
ruff check .
ruff format --check .

# Frontend tests & build
cd frontend
npm run test:run
npm run build
```

---

## License

MIT — see [LICENSE](LICENSE).
