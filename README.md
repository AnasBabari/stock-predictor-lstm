# Signal Seven

[![CI](https://github.com/AnasBabari/stock-predictor-lstm/actions/workflows/ci.yml/badge.svg)](https://github.com/AnasBabari/stock-predictor-lstm/actions)

Signal Seven brings historical stock prices, learned forecasts, volatility estimates, and recent
financial news into one interface. Search for a supported US or UK stock, explore its history,
and view an estimate for the next seven trading sessions.

A separate volatility outlook estimates **how much prices could fluctuate—not whether they
will rise or fall**.

The frontend is a React app hosted on Vercel; the backend is a FastAPI service on Render.
Opening the app wakes the backend automatically. There is no second link to open.

## What you can do

- Search supported NASDAQ, NYSE, and London-listed stocks, with exchange-aware currencies.
- Explore prices with zoom, pan, a crosshair, and time ranges based on available history.
- View a seven-session learned price estimate, visually separated from actual prices.
- Inspect a 5-, 10-, or 20-session volatility outlook and its model/fallback disclosure.
- Read recent financial headlines as separate market context.
- Inspect historical evaluation results and the source/date of the underlying data.

The research universe has expanded from the original five-stock benchmark to **286 stocks:
195 US and 91 UK**. Research coverage is not a promise that every ticker will always have
usable live data or a successful forecast.

## How it works

### Seven-session price forecasts

1. The backend retrieves adjusted daily open, high, low, close, and volume data, using completed
   market sessions for forecasting.
2. It builds historical return, volatility, trend, range, and volume features.
3. Data is split chronologically into 70% training, 15% validation, and 15% test partitions.
   Seven-session targets crossing a partition boundary are purged.
4. Ridge and Random Forest candidates compete on validation error. An available compatible
   pretrained LSTM checkpoint can also participate.
5. For locally fitted models, the selected configuration is evaluated on the later test
   partition. A production model is then fitted using all resolved historical targets.
6. Predicted cumulative log returns are converted into seven future price estimates.

The learned endpoint remains `/api/v1/forecast`. A no-change forecast is an evaluation baseline,
not a hardcoded replacement for the learned price path.

### Volatility forecasts

The separate volatility endpoint includes **G3**, a GPU-trained XGBoost model that learns a
correction to rolling volatility. Training happens offline; the backend serves exported ONNX
models without request-time G3 training or a production GPU.

The recorded study used a 286-stock panel, six calendar-year evaluation folds, and a separately
scored historical test partition. G3 was selected for integration on that evidence. Detailed
results and limitations are preserved in the [research provenance record](artifacts/PROVENANCE_FINAL.md).

If G3 cannot be loaded or used, the volatility endpoint explicitly reports a fallback to
rolling volatility. This is separate from the learned price endpoint's behavior.

### Reusing expensive work

For US stocks, chart history warms the same daily market-data cache used by the price forecast.
Repeated requests can reuse a forecast for unchanged data rather than repeat training.
Same-worker requests are coordinated to avoid duplicate training.

With `FORECAST_MODEL_CACHE_DIR` configured, fitted model/preprocessor artifacts can survive a
process restart and be loaded for inference. Changes to data, model configuration, implementation,
runtime versions, or checkpoints invalidate reuse. Invalid artifacts fall through to training.

**A local disk cache is not durable storage across Render instance replacement or redeploys.**
UK chart history currently follows a separate Yahoo path. See the
[artifact cache design and security notes](docs/FORECAST_ARTIFACT_CACHE.md).

## Honest limitations

- This is an experiment, not financial advice or a guarantee of profitable trading.
- Price forecasts and volatility forecasts answer different questions. A volatility scenario
  range does not establish a directional price forecast.
- Price uncertainty bands come from validation residuals; they are not guaranteed confidence
  intervals. Gaussian volatility scenarios also depend on modelling assumptions.
- Historical evaluation is not a live track record. A pretrained checkpoint's training history
  must be audited before treating its displayed historical metrics as out-of-sample evidence.
- Recent news is context only in the public price forecast. A positive MSFT news pilot did not
  generalize convincingly across the fixed 25-stock replication; the
  [negative result is preserved](artifacts/news_replication25_v1/STUDY_SUMMARY.md).
- G3's historical volatility results do not demonstrate that the price predictor beats its
  baseline, or that either model will retain its performance in future markets.
- Surviving-stock selection, provider coverage, and historical news revisions limit the research.
  Results do not automatically transfer to penny stocks, illiquid securities, or unseen markets.

## One-link startup

The app calls `/health` when the page opens and requests chart history for the selected stock.
A sleeping Render service starts in the background while the interface displays its loading
state. Daily history also warms the US forecast data cache.

Cold starts and the first model fit can still take time. Repeated requests can reuse cached work;
the interface keeps unavailable data and degraded results explicit. Intraday history is best-effort.

## API

```http
GET /health
GET /api/v1/history?ticker=MSFT
GET /api/v1/forecast?ticker=MSFT&days=7
GET /api/v1/volatility/forecast?ticker=MSFT&horizon=5&model=gpu_g3
GET /api/v1/news?ticker=MSFT
```

Price responses include recent historical closes, seven future trading dates, predicted prices
and uncertainty paths, model selection/evaluation details, and market-data provenance. Volatility
responses identify the model used and any fallback. Forecast routes expose server-side timing
information through `Server-Timing` headers.

## Local development

Use Python 3.11 and Node.js 22.12 or later within the Node 22 series. Run the backend and frontend
in separate terminals.

### Backend

From the repository root:

```powershell
python -m venv backend/.venv
backend/.venv/Scripts/python.exe -m pip install -r backend/requirements.txt -r backend/requirements-dev.txt
cd backend
.venv/Scripts/python.exe -m uvicorn api:app --reload --port 8000
```

Local development defaults to Yahoo. Production can select Alpaca with server-only variables:

```text
MARKET_DATA_PROVIDER=alpaca
ALPACA_API_KEY_ID=<your-key>
ALPACA_API_SECRET_KEY=<your-secret>
```

ONNX Runtime is included in backend requirements for G3 inference. PyTorch is a separate
dependency for LSTM use and research tests. Never put provider credentials in a `VITE_*`
variable or commit them to the repository.

### Frontend

From the repository root, in the second terminal:

```powershell
cd frontend
npm ci
$env:VITE_API_URL = "http://127.0.0.1:8000"
npm run dev
```

Open `http://localhost:5500`. Setting the URL explicitly keeps local requests on your local
backend; the development proxy otherwise defaults to the hosted Render service.

For production, set `VITE_API_URL` to the backend's public URL **before building** the frontend.

## Verification

From the repository root:

```powershell
backend/.venv/Scripts/python.exe -m pip install "torch==2.11.0" --index-url https://download.pytorch.org/whl/cpu
backend/.venv/Scripts/python.exe -m pytest -c backend/pyproject.toml backend/tests research/tests -q -ra
backend/.venv/Scripts/ruff.exe check .
backend/.venv/Scripts/ruff.exe format --check .

cd frontend
npm run test:run
npm run build
```

Use a separate CUDA-enabled research environment for GPU training; the command above is for
CPU test execution. Tests cover temporal purging, completed-session data, provider errors,
history-to-forecast cache reuse, artifact equivalence, chart behavior, and G3 inference/fallbacks.
The [CI workflow](.github/workflows/ci.yml) is the source of truth for runner dependencies.

## Research and next steps

- [OHLCV study summary](artifacts/STUDY_SUMMARY.md): preserved negative price-signal experiments.
- [25-stock news replication](artifacts/news_replication25_v1/STUDY_SUMMARY.md): the MSFT pilot
  did not establish a general news advantage.
- [GPU volatility provenance](artifacts/PROVENANCE_FINAL.md): G3 evaluation and integration record.
- [Artifact cache design](docs/FORECAST_ARTIFACT_CACHE.md): reuse guarantees and operational limits.

Remaining engineering work includes shared durable artifact storage, cross-worker training
coordination, more complete latency telemetry, and unifying the UK/US cache interface. New
research must remain separate from scored studies; adding a feature is not evidence that it helps.

## License

MIT — see [LICENSE](LICENSE).
