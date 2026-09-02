# Secure Live Forecast Operations

## Frozen experiment contract

Live Universe v1 begins no earlier than the completed NYSE session on
2026-09-02. Its fixed tickers are:

```text
MSFT AAPL NVDA GOOGL AMZN META JPM XOM JNJ WMT CAT NEE PLD KO NMM
```

Each eligible session contains 60 observations: 15 tickers at horizons 1, 5,
10, and 20. `model=auto` resolves through `empirical_volatility_benchmark_v3`:
GARCH(1,1) at one session and Rolling Mean (60d) at 5, 10, and 20 sessions.
Changing this universe, start date, horizon set, or model mapping creates a new
version; it must never silently rewrite Live Universe v1.

## Security boundary

`GET /api/v1/volatility/forecast` is a public, read-only preview. It computes
the same deterministic live fingerprint but cannot open or mutate the ledger.
The browser uses only this route.

`POST /api/v1/volatility/collect` is the sole live-write route. It accepts only
the frozen universe/horizons, always uses `model=auto`, and requires a bearer
credential from `FORECAST_COLLECTOR_TOKEN`. Settlement and full-ledger exports
require the same credential. Public ledger reads remain available.

The token belongs only in encrypted Render and GitHub Actions secret storage.
It must never appear in Vercel, browser bundles, command output, committed
configuration, operational manifests, or logs.

## Collector lifecycle

Run a preview locally or through the manually dispatched workflow:

```powershell
python scripts/collect_live_forecasts.py `
  --base-url https://stock-predictor-lstm.onrender.com `
  --mode dry-run
```

The preflight requires a healthy durable PostgreSQL ledger, determines the
latest completed NYSE session, and validates all 60 previews before any write.
After a Render cold start, an initial market-only degraded readiness state may
be warmed by those read-only previews; `/ready == 200` is then mandatory after
all previews and immediately before live collection. Every response must
have the expected ticker, horizon, data date, frozen policy model, code commit,
and a valid SHA-256 fingerprint. One failure aborts with zero live writes.

Live mode repeats the complete preflight, then writes sequentially at the
default 2.3-second interval. Identical retries are idempotent. A fingerprint
conflict is reported as a hard item failure. Once writing begins, later errors
produce a `partial` manifest; successful immutable records are not deleted or
recomputed.

```powershell
$env:FORECAST_COLLECTOR_TOKEN = '<scheduler secret>'
python scripts/collect_live_forecasts.py `
  --base-url https://stock-predictor-lstm.onrender.com `
  --mode live
```

Do not run live mode during development or deployment smoke testing.

## Scheduling and activation

`.github/workflows/live-forecast-operations.yml` invokes the collector at both
21:30 and 22:30 UTC on weekdays. The script converts to Europe/London and uses
the NYSE exchange calendar, so exactly the candidate matching the intended
22:30 local window can proceed. Weekends, holidays, pre-start dates, and stale
market snapshots abort safely.

Scheduled jobs are disabled by default. They require explicit repository
variables:

```text
LIVE_COLLECTION_ENABLED=true
LIVE_SETTLEMENT_ENABLED=true
LIVE_EXPORT_ENABLED=true
```

Do not set `LIVE_COLLECTION_ENABLED` until a production dry run reports 60/60
on one identical completed session and a human authorizes genuine collection.

## Settlement and audit exports

Settlement is an independent authenticated operation and changes only outcome
and scoring fields for records whose future sessions have resolved:

```powershell
python scripts/collect_live_forecasts.py --base-url <api> --mode settle
```

Weekly export is also authenticated and includes live records only:

```powershell
python scripts/collect_live_forecasts.py --base-url <api> --mode export
```

The export produces `live-ledger.json`, `live-ledger.csv`, and `SHA256SUMS`.
Historical replay is never mixed into these files. GitHub workflow artifacts
are operational audit copies, not a replacement for PostgreSQL durability.

## Operational status vocabulary

- `dry_run_passed`: all 60 previews passed; no records were written.
- `aborted`: preflight or eligibility failed; no live batch began.
- `complete`: all 60 authenticated writes succeeded or resolved idempotently.
- `partial`: live writes began, but fewer than 60 items succeeded.

Every manifest records its timestamp, expected session, universe version,
ordered ticker/horizon results, data dates, models, fingerprints, deployed code
commit, and exact success/failure counts. Never present a partial batch as
complete or backfill an earlier date as genuine live evidence.
