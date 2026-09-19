# Architecture

Signal Seven is a two-tier web service: a FastAPI backend that owns
market data, model training, and serving; and a React frontend that
reads from it. The two communicate over JSON.

## High-level layout

```
┌──────────────────────────────┐      ┌──────────────────────────────┐
│  React (Vite + chart.js)      │      │  FastAPI (uvicorn)            │
│  frontend/src/                │      │  backend/                     │
│                               │      │                               │
│  App.jsx                      │      │  api.py (lifespan + router)   │
│   ├─ SearchCard               │      │   ├─ routes/                  │
│   ├─ PriceChart (Trading212)  │  HTTP │   │  ├─ health                 │
│   ├─ SimpleForecastChart      │ ◄────► │   │  ├─ market (history/...)  │
│   └─ VolatilityOutlook        │  JSON │   │  ├─ simple_forecast       │
│                               │      │   │  └─ volatility (G3)      │
│  hooks:                       │      │   ├─ services/                │
│   ├─ usePriceHistory          │      │   │  ├─ simple_forecast.py    │
│   ├─ useVolatilityOutlook     │      │   │  ├─ g3_volatility.py      │
│   └─ useForecast              │      │   │  ├─ forecast_warmup.py    │
│                               │      │   │  ├─ forecast_artifacts.py │
│                               │      │   │  ├─ forecast_ledger.py   │
│                               │      │   │  ├─ volatility_snapshot   │
│                               │      │   │  └─ live_volatility.py    │
│                               │      │   ├─ market_data/             │
│                               │      │   │  ├─ yahoo.py, alpaca.py   │
│                               │      │   │  ├─ service.py (router)   │
│                               │      │   │  └─ cache.py (TTLCache)  │
│                               │      │   └─ data_pipeline.py         │
│                               │      │       └─ MarketDataService    │
└──────────────────────────────┘      └──────────────────────────────┘
                                                  │
                                                  │  on cold cache
                                                  ▼
                                    ┌──────────────────────────────┐
                                    │  yfinance / Alpaca (upstream)  │
                                    └──────────────────────────────┘
```

## Module map

`backend/api.py` is the FastAPI app. Its `lifespan` starts a single
background warm-up thread (see *Warm-up and cache strategy* below)
before the first request. Four route packages are mounted:

- `backend/routes/health.py` — `/`, `/health`, `/ready`, `/models`.
- `backend/routes/market.py` — `/api/v1/history`, `/api/v1/search`,
  `/api/v1/info`. Reads market data through the shared service.
- `backend/routes/simple_forecast.py` — `/api/v1/forecast`,
  `/api/v1/news`. Trains a 7-day price model on demand (cached per
  `(symbol, data_as_of, model, feature version, config, implementation,
  runtime)`).
- `backend/routes/volatility.py` — `/api/v1/volatility/forecast`,
  `/api/v1/volatility/ledger`, `/api/v1/volatility/collect` (authed),
  `/api/v1/volatility/export-ledger` (authed),
  `/api/v1/volatility/score-ledger` (authed). G3 model serving.

`backend/services/` holds the working logic. Most public functions are
*modules of functions*, not classes; the snapshot service and the
G3 session store are the only two that hold state:

- `volatility_snapshot.py` — the content-keyed snapshot cache described
  below.
- `g3_volatility.py` — the ONNX session cache, one session per
  horizon, loaded once per process.

`backend/market_data/` is the provider layer. `service.py` routes
each ticker to the right provider: US tickers go through the
`MarketDataService` (which composes yfinance / Alpaca and applies a
TTLCache plus a circuit breaker), LSE tickers go straight to yfinance.
`cache.py` is the in-memory TTLCache; `normalization.py` enforces a
single OHLCV schema across providers.

`backend/data_pipeline.py` is the lower-level data service: it owns
the circuit breaker, the per-ticker negative cache for missing
tickers, and a tiny adapter for ad-hoc callers.

`backend/calendars.py` resolves a ticker to its exchange calendar
(US NYSE/Nasdaq vs LSE) and produces future trading dates.

## Request flow

### Price forecast (`/api/v1/forecast`)

```
GET /api/v1/forecast?ticker=MSFT&days=7
  └─> simple_forecast._download_ohlcv(symbol)
        └─> market_data_service.fetch_daily_bars(symbol, years=8)
              └─> (yfinance / Alpaca, circuit-breakered, TTL-cached)
  └─> services.simple_forecast.train_and_forecast(symbol, frame)
        ├─> build_features()       (pandas features)
        ├─> chronological_masks()  (train / val / test)
        ├─> _candidate_models()    (ridge, random_forest, ...)
        ├─> maybe _load_gpu_lstm_model()  (optional)
        ├─> select winner by val MAE
        ├─> refit on full dev data
        └─> _infer_artifact() / _save_artifact()
              └─> signed, versioned JSON via forecast_artifacts.py
```

The trained `forecast_artifacts.py` payload is stored both in an
in-memory TTLCache (six hours) and, when `forecast_model_cache_dir` is
set, on disk. The cache key is content-hashed so a new data date
naturally invalidates yesterday's prediction.

### Volatility forecast (`/api/v1/volatility/forecast`)

```
GET /api/v1/volatility/forecast?ticker=MSFT&horizon=5&model=gpu_g3
  └─> _execute_forecast(symbol, horizon, model)
        └─> _prepare_forecast(symbol, horizon, model)
              ├─> build_volatility_inference_snapshot(ticker)
              │     ├─> _download_ohlcv(symbol)               (~0.2s warm)
              │     ├─> content fingerprint                 (cache key)
              │     └─> if miss:  derive snapshot           (~2.3s cold)
              │           ├─> build_features_v5(frame)
              │           ├─> realized_variance_proxies(frame)
              │           ├─> causal_log_har_forecasts(proxy, ...)
              │           └─> _garch11_cumulative_variance_path(...)
              └─> build_live_volatility_forecast(snapshot, model, ...)
                    └─> G3 ONNX session.run([window])          (cached)
                    └─> replace graph intercept with log(rolling_base)
                    └─> clip complete log variance, then exp
```

The G3 model itself is the tracked ONNX release in
`backend/volatility_models/` (`g3_h5.onnx`, `g3_h10.onnx`,
`g3_h20.onnx`, plus meta.json files with feature lists, parity
metrics, training rows, and the frozen inference formula). It is loaded
once per process into a per-horizon session cache.

### History (`/api/v1/history`)

```
GET /api/v1/history?ticker=MSFT
  └─> _fetch_history_daily(symbol)
        ├─> LSE:  YahooProvider().fetch_daily_bars(symbol, years=10)
        └─> US:   market_data_service.fetch_daily_bars(symbol, years=8)
  └─> downsample to MAX_HISTORY_BARS (1500) most-recent sessions
  └─> (intraday 24H added only when a live intraday source is configured)
```

## Warm-up and cache strategy

Three independent caches keep the request path off the heavy work:

1. **Market-data TTLCache** (per provider, in-process). Default TTL
   is 5 minutes for history fetches. The `data_pipeline` circuit
   breaker protects the upstream from a stampede of retries.

2. **Forecast artifact cache** (`forecast_artifacts.py` +
   `_cache` in `simple_forecast.py`). Two tiers:
   - In-memory TTLCache, six hours.
   - Disk cache at `forecast_model_cache_dir`, hashed by the
     full feature/data/model/implementation/runtime identity.
   Cache key intentionally includes the last data date so a new
   trading day invalidates yesterday's prediction.

3. **Volatility snapshot cache** (`volatility_snapshot.py`). Keyed
   on the SHA-256 of `pd.util.hash_pandas_object` of the downloaded
   frame plus its `data_as_of`, `data_provider` and
   `market_data_cache` attrs. Striped build locks give single-flight
   semantics: concurrent requests for the same inputs share one
   build; different inputs run in parallel. LRU at 64 entries; the
   snapshot is a frozen dataclass and no caller mutates it, so the
   same instance is safe to share.

The first visitor's `causal_log_har_forecasts` call is the largest
single cost (~2.3s of GIL-bound scalar work). The startup warm-up
(`backend/services/forecast_warmup.py`) runs in a daemon thread
triggered by `api.py`'s lifespan. For a bounded ticker list (default
8, configurable up to 64), it:

- Eagerly `import torch` so the request path never pays it.
- Calls `train_and_forecast(ticker, frame)` for the price model.
- Calls `build_volatility_inference_snapshot(ticker)` for the
  volatility model.

The warm-up is suppressed when `"pytest" in sys.modules` (so tests
never run it) and otherwise idempotent. Each step is logged at
INFO with a duration; failures are logged and swallowed — a broken
warm-up costs latency, never correctness, because every endpoint
still trains or builds on demand.

## Security model

- **Rate limiting** — `slowapi` per-IP limits on every public
  endpoint. `forecast` is 12/minute, `volatility/forecast` 30/minute,
  history 30/minute, search 30/minute. Limiter exception → JSONResponse
  with status 429.
- **CORS** — `allowed_origins` is allowlisted by environment;
  previews honour a separate regex. `X-Content-Type-Options:
  nosniff` and `Referrer-Policy: strict-origin-when-cross-origin` on
  every response.
- **Signed volatility artifacts** — production releases carry an
  ECDSA signature (`volatility_release_private_key_path`); the
  serving layer refuses anything that fails signature, size, or
  hash validation.
- **Mutable ledger** is write-only via authed `/api/v1/volatility/collect`.
  Public reads return `ledger_write: "disabled_public_preview"`.
- **Secrets** are read from environment / `.env`; `settings`
  (pydantic-settings) is the only consumer. No secrets in code.

## Deployment

`render.yaml` is the production manifest. The build step uses `uv`
to resolve a frozen lockfile; the start command is
`uvicorn api:app` on Render's native runtime. The frontend is
hosted statically on Vercel and hits the Render URL via
`VITE_API_URL`. The backend uses the durable cache directory only
when the filesystem is genuinely persistent (local dev, Docker, paid
Render), because Render's free tier wipes on sleep.

## Where the G3 model came from

`backend/volatility_models/g3_h{5,10,20}.onnx` is the deployed
artefact. The ONNX graph implements exactly the inference rule in
`docs/HISTORICAL_NEWS_VOLATILITY.md` and
`artifacts/gpu_rolling_origin_v1/DECISION.md`: a global XGBoost
rolling-volatility correction with a QLIKE objective, the 22-feature
panel set, and `base_margin = log(rolling)`. The release metadata records
`metric_source: validation_panel`; it is not an untouched-test certification.
The boot steps were:

1. `scripts/run_gpu_rolling_origin.py` (research evaluation and
   replication across chronological folds).
2. `scripts/freeze_g3_panel_v1.py` (per-horizon QLIKE XGBoost freeze
   pinned to the validation-panel training partition; manifest at
   `artifacts/g3_panel_v1/manifest.json`).
3. `scripts/package_g3_models.py` (hash-verifies those frozen boosters,
   exports without retraining to ONNX opset 15, and verifies final-variance
   parity under varied base margins before writing any artifact).

The same freeze manifests the ONNX↔XGBoost parity
(`parity_max_relative_variance_error` in each `meta.json`).
