# ORATS sample fixtures (Gate 0 — free, public, deterministic)

## Near-EOD strikes sample

- URL: https://s3.amazonaws.com/assets.orats.com/ORATS_SMV_Strikes_20240103.zip
- Local: `data/options/samples/ORATS_SMV_Strikes_20240103.zip` (63.7 MB, **gitignored**)
- SHA256: `e04a37310b0f453fedc03a000c55adfc517a119279150173916ebdc5fb514570`
- Contents: `ORATS_SMV_Strikes_20240103.csv` — 716,822 rows × 39 columns,
  5,787 tickers, trade_date 2024-01-03 (e.g. SPY: 3,894 strike-rows,
  34 expiries, stkPx 469.05).
- Grain: one row per ticker/expiry/strike, call+put legs side by side.
- Snapshot timing: ~15:46 ET (near-EOD), strictly before the session close.

Re-fetch deterministically: download the URL above and verify the SHA256
before running the adapter. Never commit the zip.
