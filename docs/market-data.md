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
and `evidence.market_data_cache`. `data_provider` is also part of every new
forecast-ledger fingerprint; existing SQLite databases migrate atomically with
`unknown` recorded for legacy entries.

Deployment verification may call:

```text
GET /api/v1/volatility/forecast?ticker=AAPL&horizon=7&record_ledger=false
```

This exercises acquisition, normalization, features, calendar generation, and
forecasting without creating a forward-ledger observation. Genuine user-facing
forecast collection keeps the default `record_ledger=true`.
