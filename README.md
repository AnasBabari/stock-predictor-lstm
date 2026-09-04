# Signal Seven

[![CI](https://github.com/AnasBabari/stock-predictor-lstm/actions/workflows/ci.yml/badge.svg)](https://github.com/AnasBabari/stock-predictor-lstm/actions)

Signal Seven is a deliberately small stock-forecasting experiment. A user selects one of five
liquid US equities and receives a learned estimate for the next seven trading sessions, together
with an uncertainty range and an honest historical backtest.

The first supported universe is:

```text
AAPL · GOOGL · MSFT · NVDA · TSLA
```

The public app has one job: make this workflow understandable and reliable before adding penny
stocks, more markets, or more complicated models.

## How it works

1. The backend downloads adjusted, completed daily OHLCV bars.
2. It builds stationary return, volatility, trend, range, and volume features.
3. Observations are split chronologically into 70% training, 15% validation, and 15% test data.
4. Seven-session targets crossing a split boundary are purged.
5. Ridge and Random Forest models compete on validation MAE.
6. The selected learned model is refitted on training + validation and scored once on the later
   untouched test block.
7. A deployment model is fitted on all resolved historical targets and predicts cumulative log
   returns for days 1–7. Those returns are converted back to prices.

The no-change forecast is shown only as a historical comparison. It never replaces the learned
forecast line returned to the user.

## Honest limitations

- This is an experiment, not financial advice.
- Seven daily closing prices are estimates, not certainties.
- The residual band is calibrated from validation errors; it is not a guaranteed confidence band.
- Recent Alpaca headlines are displayed as context only. They are not fed into the first model
  because matching timestamped historical news has not yet been evaluated with the same
  chronological split. This prevents a live sentiment score from being presented as a trained
  signal.
- Results for five large, liquid equities do not automatically transfer to penny stocks.

## One-link startup

The React app calls the backend `/health` endpoint as soon as the Vercel page opens. A sleeping
Render free service therefore starts in the background; the page displays startup attempts and
enables the same forecast flow once Render responds. Users do not need to open Render manually.

## API

```http
GET /health
GET /api/v1/forecast?ticker=MSFT&days=7
GET /api/v1/news?ticker=MSFT
```

The forecast response includes:

- the latest 90 historical closes;
- seven future trading dates;
- seven learned price estimates;
- validation-calibrated lower and upper paths;
- selected model and candidate validation scores;
- untouched test MAE, RMSE, direction accuracy, and MAE relative to persistence;
- data provider, completed-session date, and calendar provenance.

## Local development

### Backend

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
python -m uvicorn api:app --reload --port 8000
```

Local development defaults to Yahoo data. Production uses Alpaca through server-only environment
variables:

```text
MARKET_DATA_PROVIDER=alpaca
ALPACA_API_KEY_ID=...
ALPACA_API_SECRET_KEY=...
```

Credentials are never included in the browser bundle.

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

For a production frontend build, set `VITE_API_URL` to the Render service URL. Local Vite uses its
development proxy for `/api` requests.

## Verification

```powershell
backend\.venv\Scripts\python.exe -m pytest -c backend/pyproject.toml backend/tests research/tests -q
backend\.venv\Scripts\ruff.exe check .
backend\.venv\Scripts\ruff.exe format --check .

cd frontend
npm run test:run
npm run build
```

Focused tests cover chronological target purging, supported-ticker validation, learned response
shape, partial-session rejection, automatic Render wake-up, UI rendering, and news disclosure.

## Next evidence-driven step

After the five-ticker price-only benchmark is stable, acquire a timestamped historical news archive
for the same dates. Add only causal article-count, sentiment, and event features, rerun the exact
same splits, and retain news only if it improves the untouched test results broadly across tickers.

## License

MIT — see [LICENSE](LICENSE).
