# v8 Deployment Runbook — historical_temporal_test_plus_asset_transfer

This runbook is the v8 counterpart to `docs/DEPLOYMENT_GATE.md` (v7 future-prospective).
v8 is the **primary** route for predictions *now*; v7 remains the stronger future-temporal track.

## Evidence labels (do not mix)

- v8 numeric (certified today): `metric_source=locked_historical_temporal_test_plus_asset_transfer`, `news_status=not_certified`, `model_version=global-volatility-v8-numeric`
- v8 news-enhanced (when archive exists): `metric_source=locked_historical_temporal_test_plus_asset_transfer`, `news_status=certified`, `model_version=global-volatility-news-fusion-v8`
- Never label v8 as `locked_purged_walk_forward` — that is v7 future evidence only.

## Prerequisites (all must be true)

- `docs/VOLATILITY_V8_PREREGISTRATION.md` frozen and committed before test access
- Universe `universe-v8-manifest.json` with `sha256` and `per_exchange_counts >=25` (or `allow_sparse` explicitly)
- Immutable v8 market snapshot with `v8_market.universe_manifest_sha256`
- Numeric fallback news snapshot (`news_status=not_certified`) OR real historical news archive with `available_at` checks
- Chronological split manifest `split-v8-manifest.json` with `temporal_test_rows` + `asset_transfer_test_rows` separate
- Candidate trained only on `train` (70%), selected on `val` (15%), never opened `test` (15%)
- One-shot certification `scripts/certify_v8_candidate.py --open-sealed-test` produced `v8-locked-certification.json` with `status=passed` and `locked_v8_certification_candidate` directory
- ONNX parity `scripts/export_v8_onnx.py` passed (`onnx-parity.json` status passed)
- Release signed via `scripts/package_volatility_release.py` (uses `backend/release_keys/volatility-v1.public.pem`) and `archive_sha256` recorded
- `research/tests/test_v8_purge_embargo.py` green

## Build v8 market snapshot (Slice 3)

```powershell
python scripts/build_v8_market_snapshot.py `
  --source-panel-dir C:\path\to\panel-69-ticker `
  --universe-manifest universe-v8-manifest.json `
  --out-root C:\path\to\v8-market-root
```

Verify `manifest.json` contains `v8_market.universe_manifest_sha256` and `pooled_checksum` unchanged.

## Dry-run (proves pipeline runnable today)

```powershell
python scripts/run_v8_certification_dry_run.py `
  --panel-dir C:\path\to\v8-market-snapshot `
  --out research/results/v8-dry-run.json
# Expected: train 70748 val 13585 pooled 16974 (temporal 13530 asset_transfer 3444) with explicit holdouts
```

## Train (Slice 9, RTX required for TCN)

```powershell
python scripts/run_v8_volatility_research.py `
  --panel-dir C:\path\to\v8-market-snapshot `
  --universe-manifest universe-v8-manifest.json `
  --out C:\path\to\v8-candidate `
  --news-enabled false   # true only when historical archive exists
# Output: prospective_v8_development_candidate with seeds 41,42,43
```

Record GPU model, driver, CUDA, torch, python, split SHA, duration, peak VRAM.

## Certify (Slice 12, one-shot)

```powershell
python scripts/certify_v8_candidate.py `
  --candidate-dir C:\path\to\v8-candidate `
  --panel-dir C:\path\to\v8-market-snapshot `
  --universe-manifest universe-v8-manifest.json `
  --out C:\path\to\v8-cert `
  --open-sealed-test `
  --holdouts NMM,MSFT
# Must create v8-holdout-opened.json BEFORE evaluation (fail-closed)
# On passed: candidate/ locked_v8_certification_candidate
```

Verify `v8-locked-certification.json` has `metric_source=locked_historical_temporal_test_plus_asset_transfer` and `release_eligible` true.

## ONNX parity (Slice 13)

```powershell
python scripts/export_v8_onnx.py `
  --candidate-dir C:\path\to\v8-cert\candidate `
  --out C:\path\to\v8-onnx
# Requires parity passed before signing
```

## Sign & package (Slice 14)

```powershell
python scripts/assemble_volatility_release.py `
  --candidate-dir C:\path\to\v8-cert\candidate `
  --output-dir C:\path\to\v8-release `
  --private-key-path C:\secure\volatility-v1.private.pem `
  --public-key-path backend\release_keys\volatility-v1.public.pem

python scripts/package_volatility_release.py `
  --release-dir C:\path\to\v8-release `
  --public-key-path backend\release_keys\volatility-v1.public.pem `
  --output C:\path\to\stocklstm-volatility-v8.zip
# Record archive_sha256, do not use mutable latest URL
```

## Render (Slice 15, inference only)

Set in Render dashboard (never in Git):

```
VOLATILITY_SERVING_REQUIRED=true
VOLATILITY_RELEASE_ARCHIVE_URL=<immutable https URL to stocklstm-volatility-v8.zip>
VOLATILITY_RELEASE_ARCHIVE_SHA256=<exact sha256>
VOLATILITY_PUBLIC_KEY_PATH=backend/release_keys/volatility-v1.public.pem
```

Render verifies signature, checksums, schema, `global-volatility-distribution-v8-*` protocol, and `feature_schema_version` before caching. No training on request.

Verify:

```
GET /health
GET /ready          # 200 only when v8 release verified
GET /models         # global_volatility.status=ready, model_version, metric_source, news_status
GET /api/v2/forecast?ticker=MSFT&horizon=7
GET /api/v2/forecast?ticker=NMM&horizon=7
GET /api/v2/forecast?ticker=AAPL&horizon=7
GET /api/v2/forecast?ticker=VOD.L&horizon=7  # LSE example where available
```

Expected evidence:

```json
{
  "execution_mode": "server_artifact_loaded",
  "model_version": "global-volatility-v8-numeric",
  "metric_source": "locked_historical_temporal_test_plus_asset_transfer",
  "certification_scope": "historical_temporal_test_plus_asset_transfer",
  "news_enabled": false,
  "news_status": "not_certified"
}
```

## Vercel (Slice 16)

Only after Render smoke passes:

```
VITE_VOLATILITY_SERVING_ENABLED=true
```

UI must show `Certified v8 global volatility model — Historical temporal test + asset-transfer, News: not certified` with correct `metric_source`, never `locked_purged_walk_forward`.

## Smoke (Slice 17)

`scripts/verify_release.ps1` plus:

- invalid ticker → 400
- invalid horizon → 400
- tampered archive → 503
- wrong sha → 503
- CORS from Vercel → ok
- repeat request → cache hit
- rollback `VOLATILITY_SERVING_REQUIRED=false` → abstention

## Retention (Slice 18)

Active + previous + audit retention protected, generation-aware GC, dry-run mode, audit logs. See `docs/VOLATILITY_V8_PREREGISTRATION.md: retention` and GitHub issue #2.

## Contingency

If historical news unavailable: certify `global-volatility-v8-numeric` now, keep news branch as uncertified experiment, add live news only as context. Do not claim `news_enabled=true` without archive.
