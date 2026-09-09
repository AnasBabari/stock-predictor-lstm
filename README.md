# Signal Seven

[![CI](https://github.com/AnasBabari/stock-predictor-lstm/actions/workflows/ci.yml/badge.svg)](https://github.com/AnasBabari/stock-predictor-lstm/actions)

Signal Seven is a deliberately small stock-forecasting experiment. A user selects a liquid US/UK equity and receives a learned estimate for the next seven trading sessions, together with an uncertainty range and an honest historical backtest. The frontend is a static Vercel app; the backend is a FastAPI service on Render that serves only completed-session data.

**Current supported universe:** 286 stocks across LSE/NASDAQ/NYSE (195 US, 91 UK) for volatility research; the original 5-ticker price benchmark (`AAPL · GOOGL · MSFT · NVDA · TSLA`) remains frozen for provenance.

## What shipped in the study branch (`study/gpu-panel-volatility-v1`, now merged to `main`)

**Volatility research is now the promoted path.** Four price-direction hypotheses were frozen as *null* (see `artifacts/STUDY_SUMMARY.md` and `v1.0-ohlcv-negative-study` tag), then the volatility-structure ladder produced one genuine winner:

- **G3 — global GPU XGBoost rolling-volatility correction** (`Vhat = B·exp(δ)`, QLIKE objective, `base_margin=log(rolling)`). Validated on 286 stocks / ~466k origins and **6 calendar-year rolling folds (2019–2024)** with HAC inference, then **burned once on the untouched test set** (`91k/90k/88k` origins, ΔQLIKE +0.239/+0.238/+0.237, `p<1e-11`). Packaged as 3×237KB ONNX (opset 15, parity ≤1.43e-05) in `backend/volatility_models/`.

**Production integration:**

- `GET /api/v1/history?ticker=MSFT` — single round-trip daily closes (downsampled to 1500) + 5-min intraday (best-effort, 120s TTL). **Warms the shared market-data cache** so the forecast that follows is a cache hit. Provider routing: US → shared cache, `.L` → Yahoo direct. Downsampling, tight error mapping (404/503/422), no large binaries in git.
- `GET /api/v1/volatility/forecast?ticker=MSFT&horizon=5&model=gpu_g3` — ONNX inference, `gpu_promoted` evidence, explicit `fallback_used` → `rolling_mean`, risk framing (`_trailing_risk_context`), `Server-Timing` header.
- **Trading212-style `PriceChart`** — 5D default (previous week), `24H/5D/1M/6M/1Y/5Y/MAX` availability-gated (e.g. 2024 IPO hides `5Y`), scroll-zoom / drag-pan / double-click reset, crosshair, last-price tag, forecast-region shading, **dashed estimate with a visible break from actuals**, `historical_provenance: 'synthetic'` guard (never draws fabricated history).
- **Volatility Outlook card** — `Enhanced volatility forecast` (no `G3`/`XGBoost`/`HAR` in user copy), risk pill, 7-day combined outlook, `How this forecast works` disclosure, advanced diagnostics.

## How it works

**Price path (per-request, 7 sessions):**
1. Download adjusted, completed daily OHLCV bars (8y, `yfinance` → `Alpaca` fallback, 6h forecast cache + on-disk artifact cache).
2. Build stationary return/volatility/trend/range/volume features.
3. Chronological 70/15/15 split with 7-session purge; Ridge vs Random Forest vs `gpu_lstm` compete on validation MAE; winner refitted and scored once on the untouched test block; deployment fits on all resolved targets. `Server-Timing: data, train_or_cache, total`.

**Volatility path (promoted, no request-time training):**
1. `GET /history` warms the OHLCV cache; `GET /volatility/forecast?model=gpu_g3` loads the frozen ONNX, builds the 22-column G3 feature block (range estimators, asymmetry, HAR components, vol-of-vol), adds `log(rolling)` outside the graph, `exp(clip)` → always-positive variance.
2. On any G3 load/inference failure, degrades **explicitly** to `rolling_mean` (`model_status: baseline`, `fallback_used` set).

The no-change forecast is shown only as a historical comparison. It never replaces the learned forecast.

## Honest limitations

- This is an experiment, not financial advice.
- Seven prices and 5/10/20-session volatilities are estimates, not certainties.
- The residual band is validation-calibrated; not a guaranteed confidence band. The `expected range` around the current price is a 1σ Gaussian reference, not a direction forecast.
- Recent Alpaca headlines are context only — not fed into the price model in this branch. The 72.5% headline revision rate is handled by a 6-hour gate; SPY macro proved independent (`r=0.003`) and null.
- Results on large caps do not transfer automatically to penny stocks (e.g. `IGC` ~$0.30).

## One-link startup

The React app calls `/health` on open. A sleeping Render free instance boots in background; the UI shows `Starting forecast service…` and enables the same flow once online. First paint also calls `GET /history` for `MSFT` to warm the cache. Users never open Render manually.

## API

```http
GET /health
GET /api/v1/history?ticker=MSFT
GET /api/v1/forecast?ticker=MSFT&days=7
GET /api/v1/volatility/forecast?ticker=MSFT&horizon=5&model=gpu_g3
GET /api/v1/news?ticker=MSFT
```

Forecast responses include: latest 90 closes, 7 future dates, 7 price estimates, validation-calibrated band, model selection, untouched-test MAE/RMSE/direction, and `data_provider`/`data_as_of`/`market_data_cache` provenance. Volatility responses include `model_status: gpu_promoted|baseline`, `fallback_used`, `risk_level`, and `forecast_fingerprint`.

## Local development

### Backend
```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
# ONNX Runtime is required for G3 tests (installed via requirements.txt)
python -m uvicorn api:app --reload --port 8000
```
Local defaults to Yahoo; production uses Alpaca via `MARKET_DATA_PROVIDER=alpaca` and `ALPACA_API_KEY_ID/SECRET` (server-only, never in the bundle).

### Frontend
```powershell
cd frontend
npm install
npm run dev
```
`VITE_API_URL` → Render URL for production; Vite proxies `/api` locally.

## Verification

```powershell
backend\.venv\Scripts\python.exe -m pytest -c backend/pyproject.toml backend/tests research/tests -q
backend\.venv\Scripts\ruff.exe check .
backend\.venv\Scripts\ruff.exe format --check .

cd frontend
npm run test:run
npm run build
```

Focused tests cover chronological purging, ticker validation, learned response shape, cache-warming invariant (`history` → `forecast` = 1 upstream fetch), ONNX parity, `gpu_promoted` status mapping, client hardening (`window` → `setTimeout`, `historical_provenance` guard), and the `G3` fallback/explicit-risk contract.

## Provenance

- `v1.0-ohlcv-negative-study` tag + `artifacts/STUDY_SUMMARY.md`: 4 frozen nulls.
- `study/gpu-panel-volatility-v1` → `main` (merge `98abacc`): `vol_structure_*`, `gpu_vol_panel_v1`, `gpu_rolling_origin_v1` (6 folds + one-shot `test_report.json` with refusals), `backend/volatility_models/*.onnx` (parity ≤1.43e-05), `artifacts/PROVENANCE*.md`.
- No secrets in repo or env (`.env` placeholders only); `FORECAST_COLLECTOR_TOKEN` handling hardened with `AUTH_MISMATCH_HINT`; large `*.parquet`/`*.jsonl`/`*.ubj` ignored except curated `data/news/alpaca/*.jsonl`.

## License

MIT — see [LICENSE](LICENSE).
