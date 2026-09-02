# Production Market Data

## Provider policy

The FastAPI service owns all market-data credentials. Production on Render uses
Alpaca's authenticated historical stock-bars endpoint. Requests use daily bars,
the configured `iex` feed, and `adjustment=all` so open, high, low, close, and
volume are obtained under one explicit adjustment policy. Yahoo is retained only
as an explicit local/development fallback and is disabled in `render.yaml`.

Configuration:

```text
MARKET_DATA_PROVIDER=alpaca
MARKET_DATA_YAHOO_FALLBACK_ENABLED=false
ALPACA_API_KEY_ID=<Render secret>
ALPACA_API_SECRET_KEY=<Render secret>
ALPACA_DATA_FEED=iex
ALPACA_ADJUSTMENT=all
```

The current Render free service has no persistent disk. Its normalized-bar cache
therefore lives under `/tmp`, can survive repeated requests within an instance,
and is lost when the instance is replaced or restarted. It is an availability
optimization, not durable storage.

The forward forecast ledger uses SQLite at `data/forecast_ledger.db` for local
development and deterministic tests. Production must set
`FORECAST_LEDGER_DATABASE_URL` (or the platform-standard `DATABASE_URL`) to a
managed PostgreSQL database and set `FORECAST_LEDGER_DATABASE_REQUIRED=true`.
When that flag is enabled, a missing/unreachable PostgreSQL store keeps `/ready`
at `503` and authenticated collector writes fail with a sanitized `503`; the API
never silently falls back to ephemeral SQLite. This prevents a Render instance
replacement from erasing genuine forward observations.

Migrate an existing local database explicitly, after taking a backup:

```text
python scripts/migrate_forecast_ledger.py \
  --sqlite-path data/forecast_ledger.db \
  --database-url "$FORECAST_LEDGER_DATABASE_URL"
```

The migration is idempotent, preserves ids/status/settlement values and
fingerprints, and aborts on an immutable logical-key conflict. It does not
invent live history. `scripts/export_forecast_ledger.py` can produce separate
deterministic JSON/CSV exports for `live` and `historical_replay` records before
or after migration.

## Cache and session freshness

Cache keys contain provider and uppercase symbol. A cache entry is usable only
when its last bar is at least the most recent completed NYSE regular session.
Before the current session closes, weekends, and holidays all resolve to the
previous completed session. Stale entries are never used to conceal an upstream
failure.

## HTTP semantics

- A provider-confirmed unknown symbol returns `404`.
- Rate limits, timeouts, provider `5xx` responses, rejected credentials, malformed
  provider payloads, and missing production credentials return `503`.
- Temporary failures use the stable, sanitized body:

```json
{
  "error": "MARKET_DATA_UNAVAILABLE",
  "message": "Current market data is temporarily unavailable. Please try again later."
}
```

Provider exception text and credentials are never returned to clients.

`/health` is an O(1) process liveness probe. `/ready` is stricter: it returns
`200` only after the service has current-session provider evidence or a fresh
cache entry, and returns `503` otherwise. Render uses `/health` for process
management so a temporary market-data outage does not create a restart loop.

## Provenance and safe smoke tests

Successful forecasts disclose `evidence.data_provider`, `evidence.data_as_of`,
`evidence.code_commit`, `evidence.snapshot_id`, and `evidence.market_data_cache`.
They also disclose the selected horizon, model-policy version, and the exact
`auto_model_policy` mapping. `data_provider` is also part of every new
forecast-ledger fingerprint; existing SQLite databases migrate atomically with
`unknown` recorded for legacy entries.

## Production horizon and model policy

The active endpoint and genuine live ledger are locked to **1, 5, 10, and 20
completed trading sessions**. Requests for 3, 7, 14, or 30 sessions are rejected
even though older browser and replay artifacts may contain those values. With
`model=auto`, the frozen `empirical_volatility_benchmark_v3` policy selects
`garch_11` for 1 session and `rolling_mean` for 5, 10, and 20 sessions. The
response records this policy so a horizon cannot silently inherit a legacy
7-session route.

The p05–p95 output is a **Gaussian model-implied price range**. It uses a
zero-drift log-return reference process to show conditional price dispersion;
the midpoint is not a price forecast and the nominal 90% level is not a
calibrated confidence interval.

Deployment verification and the browser may call:

```text
GET /api/v1/volatility/forecast?ticker=AAPL&horizon=5
```

This exercises acquisition, normalization, features, calendar generation, and
forecasting without creating a forward-ledger observation. The GET route has no
ledger-writing mode; supplying the removed `record_ledger` query key cannot turn
it into a mutation. Genuine observations are created only through the protected
`POST /api/v1/volatility/collect` route by the operational collector. Legacy
7-session ledger rows are retained for audit but cannot be created by the active
route.

`/health` is intentionally an O(1) liveness check. `/ready` includes a
`forecast_ledger` dependency object with `backend`, `durable`, and `required`
fields. Configure the PostgreSQL URL and verify that this object reports
`status=available` before starting a public collection run. On a Render free
service the application process can still be healthy while readiness remains
degraded until the external database is supplied.

## Secure live collection boundary

Set `FORECAST_COLLECTOR_TOKEN` only in Render and the trusted scheduler. Never
put it in Vercel, a `VITE_*` variable, frontend source, logs, or API responses.
The protected mutation routes are:

```text
POST /api/v1/volatility/collect?ticker=MSFT&horizon=5
POST /api/v1/volatility/score-ledger?ticker=MSFT
GET  /api/v1/volatility/export-ledger
Authorization: Bearer <collector token>
```

The public `GET /api/v1/volatility/ledger` remains readable and never generates
historical replay data as a side effect. Historical replay is an explicit
offline operation and remains distinct from the genuine live track.
