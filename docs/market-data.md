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

The forward forecast ledger is also SQLite-backed and defaults to the repository
`data/forecast_ledger.db` path. Its fingerprints and settlement updates are
immutable while the database exists, but the free Render filesystem can lose the
database on a restart or replacement. Before treating the live track record as a
permanent public audit trail, configure durable database/object storage or export
the ledger on a scheduled basis.

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

Deployment verification may call:

```text
GET /api/v1/volatility/forecast?ticker=AAPL&horizon=5&record_ledger=false
```

This exercises acquisition, normalization, features, calendar generation, and
forecasting without creating a forward-ledger observation. Genuine user-facing
forecast collection keeps the default `record_ledger=true` after selecting one
of the four supported horizons; legacy 7-session ledger rows are retained for
audit but cannot be created by the active route.
