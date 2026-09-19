# API Reference

All endpoints are served by the FastAPI backend. The base URL in
production is the Render service root (for example
`https://stock-predictor-lstm.onrender.com`); that is the only value
the frontend needs in `VITE_API_URL`.

Every response is JSON. Arrays of daily bars are always ordered
oldest → newest. Unless stated otherwise, responses are cached
per-process for a few minutes (see
[architecture.md](architecture.md#warm-up-and-cache-strategy)).

> Response shapes below were captured from a running server against
> live market data. Full JSON examples are in
> [`docs/api-samples/`](api-samples/).

## Conventions

| aspect | detail |
|---|---|
| Ticker format | 1–12 chars, `A–Z`, `0–9`, `.`, `-`, `_`. LSE tickers use the `.L` suffix (`SHEL.L`). |
| Dates | ISO `YYYY-MM-DD` (daily), ISO 8601 with offset (intraday). |
| Times | `origin_date` / `as_of` refer to the last **completed** session, never an in-progress one. |
| Rate limiting | Per-IP, SlowAPI. Exceeded → `429` JSON. |
| Errors | `{"detail": "<human-readable>"}`, except where noted. |

## Health & readiness

### `GET /health`

Liveness probe. Always fast — it never touches market data or a model.

```json
{"status":"ok","version":"1.1.0","deployment":{"provider":"unknown","environment":"local","commit":null,"preview":false}}
```

### `GET /ready`

Readiness probe. Reports the upstream market-data dependency (circuit
breaker state, consecutive failures, last error) so you can tell
"not ready" from "upstream is down".

### `GET /models`

Catalogue of currently available models and their status. Useful for
diagnosing whether the G3 volatility release loaded or fell back.

### `GET /`

Root banner + version.

## Market data

### `GET /api/v1/history`

Price history for the chart. One round trip backs every range tab and
zoom level; the client derives 5D…MAX from the daily series.

| parameter | type | default | notes |
|---|---|---|---|
| `ticker` | string | `AAPL` | validated against the supported universe |
| — | — | — | rate limit 30/minute |

Response:

| field | type | notes |
|---|---|---|
| `ticker` | string | normalised, upper-case |
| `as_of` | string | last completed session |
| `provider` | string | which market-data provider answered |
| `market_data_cache` | string | `hit` / `miss`, useful when diagnosing latency |
| `first_date` | string \| null | oldest date in the returned series |
| `daily` | array of `{d, c}` | date + close, capped at the most recent **1500** sessions |
| `intraday` | array \| null | 24H bars; `null` until a live intraday source is configured |
| `intraday_session` | string \| null | |

### `GET /api/v1/search`

Symbol autocomplete.

| parameter | type | constraints |
|---|---|---|
| `query` | string | 1–100 chars |

Returns `{"results":[{"ticker","name","type"}, ...]}`. When the
upstream search is unreachable it degrades to an exact-symbol
fallback and adds `"degraded": true` rather than failing.

### `GET /api/v1/info`

Company profile.

| parameter | type | default |
|---|---|---|
| `ticker` | string | `AAPL` |

```json
{"ticker":"MSFT","name":"Microsoft Corporation","exchange":"NMS","currency":"USD",
 "marketCap":3650769387520,"peRatio":27.39,"fiftyTwoWeekHigh":553.72,
 "fiftyTwoWeekLow":349.2,"avgVolume":36277025,"dayHigh":494.39,"dayLow":489.8,
 "previousClose":493.95,"sector":"Technology","industry":"Software - Infrastructure"}
```

## Price forecast

### `GET /api/v1/forecast`

A learned seven-session price estimate. **Rate limited to 12/minute**
(it is the most expensive endpoint).

| parameter | type | default | notes |
|---|---|---|---|
| `ticker` | string | `MSFT` | must be in the supported universe |
| `days` | int | `7` | 1–7; the horizon is fixed at 7, other values are rejected with `400` |
| `model` | string | `auto` | `auto`, or a named candidate |

Response:

| field | type | notes |
|---|---|---|
| `ticker`, `ticker_name` | string | |
| `exchange_mic`, `exchange_name` | string | e.g. `XNAS` / `NASDAQ` |
| `currency`, `currency_symbol` | string | `USD` / `$`, `GBp` / `p` for LSE |
| `forecast_days` | int | always 7 |
| `data_as_of` | string | last completed session used |
| `current_price` | float | |
| `future_dates` | string[7] | next 7 **trading** days (exchange calendar aware) |
| `predicted_prices` | float[7] | central estimate |
| `lower_prices`, `upper_prices` | float[7] | residual-quantile band |
| `historical_dates`, `historical_prices` | arrays | recent actuals for context |
| `historical_error_band` | object | `{lower_prices, upper_prices}` |
| `model` | object | `{name, kind, feature_version, target, selection, candidate_validation_mae}` |
| `backtest` | object | see below |
| `provenance` | object | `{data_provider, market_data_cache, calendar, completed_daily_bars_only}` |
| `news` | object | headlines, explicitly **non-model** context |

`backtest` is a chronological holdout, not a fit:

```
split, test_start, test_end, test_samples, metric_source,
mae_percent, rmse_percent, direction_accuracy,
persistence_mae_percent, relative_mae_vs_persistence, per_horizon
```

`relative_mae_vs_persistence` is the honest headline: below `1.0`
means the model beat a naive "tomorrow = today" on that ticker.

## Volatility (G3)

Volatility endpoints estimate **how much** a price may move, never
which direction. See [methodology.md](methodology.md).

### `GET /api/v1/volatility/forecast`

| parameter | type | default | notes |
|---|---|---|---|
| `ticker` | string | `AAPL` | |
| `horizon` | int | `5` | must be one of **5, 10, 20** |
| `model` | string | `auto` | `gpu_g3` selects the learned validation-panel model explicitly |

Response (abridged):

| field | type | notes |
|---|---|---|
| `ticker`, `as_of`, `horizon` | | |
| `current_price` | float | |
| `historical_dates`, `historical_prices` | arrays | last 90 sessions |
| `forecast` | object | see below |
| `evidence` | object | ~35 keys of provenance (below) |

`forecast`:

```
future_dates, price_quantiles, expected_cumulative_variance_path,
expected_cumulative_variance, expected_annualized_volatility,
predicted_volatility, volatility_unit, model, requested_model, baseline
```

`evidence` carries the disclosure the UI shows under *How this
forecast works*: `model_status`, `model_family`, `model_name`,
`baseline` (true when it fell back), `model_version`,
`feature_set_version`, `model_policy_version`, `fallback_used`,
`test_evidence_qlike_vs_rolling` (null unless independently supplied), `risk_level`,
`risk_ratio_vs_trailing_60d`, `trailing_annualized_volatility_60d`.

When the requested learned model cannot be used, the response **says so**:
`baseline: true` plus a non-null `fallback_used`. It never silently
substitutes a different model.

The packaged G3 artifacts currently report `metric_source: validation_panel`.
That is historical validation evidence, not a claim of untouched production
test performance. A serving failure is disclosed with `model_status: baseline`
and a non-null `fallback_used`; the UI renders missing risk information as
`Unavailable`.

### `GET /api/v1/volatility/ledger`

Immutable forward ledger of past forecasts, so live performance can
be measured later. Read-only.

### Authenticated volatility endpoints

These require a collector credential and are not part of the public
surface:

- `POST /api/v1/volatility/collect` — record one forecast.
- `GET /api/v1/volatility/export-ledger` — export the ledger.
- `POST /api/v1/volatility/score-ledger` — score recorded forecasts
  against realised outcomes.

## News

### `GET /api/v1/news`

Recent headlines for a ticker, delivered as **context only**. They
are never an input to the price or volatility models.

## Error responses

| status | when |
|---|---|
| `400` | Bad parameter (e.g. `days != 7`, horizon not in 5/10/20) |
| `404` | Ticker not in the supported universe / no market data |
| `422` | Market history too short or not finite enough to form a forecast |
| `429` | Rate limit exceeded |
| `503` | Upstream market data unavailable (circuit open), or the ledger is unavailable |

A `503` from `/api/v1/volatility/forecast` returns a structured body
rather than a bare `detail`, so callers can distinguish
`MARKET_DATA_UNAVAILABLE` from `FORECAST_LEDGER_UNAVAILABLE`.

## Server timing

`/api/v1/forecast` returns a `Server-Timing` header:

```
Server-Timing: data;dur=12, train_or_cache;dur=42, total;dur=55
```

`train_or_cache` is the useful one: a low value means the artifact
cache served the request, a high one means the model was actually
fitted on this request.
