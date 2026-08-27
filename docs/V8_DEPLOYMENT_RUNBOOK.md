# v8 training, certification, and deployment runbook

v8 uses a historical chronological test plus unseen-asset transfer test so it can be evaluated
without waiting for the v7 prospective window. It is not automatically stronger than v7, and it
is not certified merely because the pipeline exists.

## Current state (2026-08-27)

- The implementation supports numeric and point-in-time news-fusion candidates.
- No certifiable v8 universe, market snapshot, candidate, locked result, or signed release has been
  produced yet.
- The existing 69-ticker panel and dry-run universe are diagnostic only. They cannot be relabelled,
  copied, or promoted into v8 certification.
- The sealed 15% test has not been opened.
- Production correctly abstains until a signed release passes verification.
- A numeric v8 release can be served after certification. A news release additionally requires a
  production provider that reproduces the signed news schema; until then readiness reports
  `news_input_unavailable` and forecasting returns a sanitized 503.

Evidence labels must remain distinct:

- numeric v8 after a passed one-shot certification:
  `model_version=global-volatility-v8-numeric`,
  `metric_source=locked_historical_temporal_test_plus_asset_transfer`,
  `news_status=not_certified`;
- news v8 only after paired ablation and sealed certification pass:
  `model_version=global-volatility-news-fusion-v8`,
  `metric_source=locked_historical_temporal_test_plus_asset_transfer`,
  `news_status=certified`;
- v7 prospective evidence uses a different identity and must never be used as v8 evidence.

## 1. Build an attested point-in-time universe

Prepare a member CSV using the schema enforced by
`research/volatility_forecasting/universe_ingest_v8.py`. Every source needs immutable evidence and
an operator attestation covering licensing, point-in-time membership, historical listing status,
and delisted-security availability. The certifiable universe must include XNAS, XNYS, and XLON,
point-in-time S&P 500 membership rows, sufficient sector/history/liquidity coverage, and the
predeclared NMM/MSFT holdouts.

```powershell
python scripts/build_v8_universe.py `
  --members-csv C:\v8\sources\members.csv `
  --source-attestations C:\v8\sources\source-attestations.json `
  --evidence-file security-master=C:\v8\sources\security-master.csv `
  --evidence-file sp500-history=C:\v8\sources\sp500-membership.csv `
  --selection-policy C:\v8\sources\selection-policy.json `
  --out-dir C:\v8\universe
```

Do not use `--diagnostic-allow-sparse` for a candidate intended for certification. Verify
`coverage_certifiable=true`, all source checksums, and the manifest SHA before acquisition.

## 2. Acquire the immutable raw-plus-adjusted market snapshot

The certifiable path downloads the complete universe with raw OHLC, adjusted OHLC, volume,
dividends, and splits. Review the provider terms before acknowledging the license.

```powershell
python scripts/build_v8_market_snapshot.py `
  --download-from-universe `
  --universe-manifest C:\v8\universe\universe-v8-manifest.json `
  --out-root C:\v8\market `
  --years 10 `
  --provider <provider-id> `
  --provider-snapshot-id <immutable-provider-snapshot-id> `
  --provider-license-id <reviewed-license-id> `
  --license-acknowledged `
  --v8-protocol-version global-volatility-distribution-v8-numeric
```

`--source-panel-dir` is intentionally diagnostic: legacy adjusted-only panels do not preserve the
raw corporate-action evidence required for certification. Verify exact universe coverage, per-
security ID/MIC/currency/timezone identity, minimum history, and `v8_market.coverage_certifiable`.

## 3. Choose the numeric or news research route

The numeric route is the shortest certifiable route and should be run first. It uses a content-
addressed no-news identity and does not claim news evidence.

For the news route, first build an immutable point-in-time event lake (for example with the GDELT
snapshot tools), complete the exact ticker alias and exposure maps for the frozen universe, then
bind the archive:

```powershell
python scripts/prepare_v8_news_snapshot.py `
  --news-snapshot-dir C:\v8\news\events `
  --universe-manifest C:\v8\universe\universe-v8-manifest.json `
  --market-manifest C:\v8\market\<panel-id>\manifest.json `
  --ticker-aliases C:\v8\news\ticker-aliases.json `
  --provider-license-id <reviewed-news-license-id> `
  --out-dir C:\v8\news\binding
```

The archive must cover the market period plus the full initial lookback with no silent provider
gaps. `snapshot_ready_uncertified` means the archive is eligible for paired experiments, not that a
news model is certified. `--allow-provider-gaps` is diagnostic only.

## 4. Generate and review the frozen split

The research runner constructs the chronological 70/15/15 split. Assignment hashes bind every row
to its stable security ID and exchange MIC. A certifiable split proves XNAS/XNYS/XLON coverage in
development assets, holdout assets, train rows, validation rows, temporal-test rows, and transfer-
test rows. NMM and MSFT remain unseen asset-transfer holdouts.

Before GPU work, run the split/provenance tests and archive the universe, market, news (if used),
and split checksums. Never inspect test labels, metrics, or predictions at this stage.

## 5. Run validation-only GPU research

Use the dedicated CUDA environment and write candidates outside Git.

Numeric candidate:

```powershell
$env:PYTHONPATH='research;backend;scripts'
C:\Users\Babar\Documents\Coding\OpenSource\autoresearch-win-rtx\.venv-stocklstm\Scripts\python.exe `
  scripts\run_v8_volatility_research.py `
  --panel-dir C:\v8\market\<panel-id> `
  --universe-manifest C:\v8\universe\universe-v8-manifest.json `
  --out C:\v8\candidates\numeric-001 `
  --news-enabled false `
  --device cuda
```

News-fusion candidate:

```powershell
$env:PYTHONPATH='research;backend;scripts'
C:\Users\Babar\Documents\Coding\OpenSource\autoresearch-win-rtx\.venv-stocklstm\Scripts\python.exe `
  scripts\run_v8_volatility_research.py `
  --panel-dir C:\v8\market\<panel-id> `
  --universe-manifest C:\v8\universe\universe-v8-manifest.json `
  --out C:\v8\candidates\news-001 `
  --news-enabled true `
  --news-snapshot-dir C:\v8\news\events `
  --news-manifest C:\v8\news\binding\news-v8-manifest.json `
  --ticker-aliases C:\v8\news\ticker-aliases.json `
  --news-exposures C:\v8\news\ticker-exposures.json `
  --device cuda
```

The news path first runs matched market-only and market-plus-news five-fold evaluations for every
seed. It requires incremental QLIKE, block-bootstrap, DM/Holm, fold-count, worst-fold, and matched-
HAR evidence before `validation_selected` can be true. Training completion alone never promotes a
model. Record GPU, driver, CUDA, Python, Torch, duration, peak VRAM, and every immutable input SHA.

## 6. Review development evidence before test access

The candidate must have all of the following:

- `artifact_role=prospective_v8_development_candidate`;
- `release_eligible=false` (only certification may change this);
- seeds 41, 42, and 43 with real `.pt` weights and checksums;
- strict validation selection on required horizons 1, 3, 5, and 7;
- exact split, universe, market, and news identities;
- for news, a train-only scaler, ordered news feature names, matrix checksum, and all paired ablation
  evidence;
- no placeholder, diagnostic universe, missing venue, or stale schema.

If the role is `rejected_v8_development_evidence`, do not open the test to rescue it. Change the
preregistered model protocol, create a new development cycle, and keep the prior evidence immutable.

## 7. Open the sealed test exactly once

Numeric:

```powershell
python scripts/certify_v8_candidate.py `
  --candidate-dir C:\v8\candidates\numeric-001 `
  --panel-dir C:\v8\market\<panel-id> `
  --universe-manifest C:\v8\universe\universe-v8-manifest.json `
  --out C:\v8\certification\numeric-001 `
  --holdouts NMM,MSFT `
  --open-sealed-test
```

For news, add the same four news/alias/exposure inputs used during training. The certifier verifies
provenance before access, writes `v8-holdout-opened.json` before loading derived examples,
recomputes the causal news matrix after that marker, and compares its identity to the candidate.

Certification is one-shot. A failed result is evidence, not permission to tune against the test.
Only `status=passed`, `release_eligible=true`, complete temporal and asset-transfer decisions, and a
materialized `locked_v8_certification_candidate` may proceed.

## 8. Export ONNX and prove parity

```powershell
python scripts/export_v8_onnx.py `
  --candidate-dir C:\v8\certification\numeric-001\candidate `
  --out C:\v8\onnx\numeric-001
```

Every seed must pass PyTorch-to-ONNX parity. A news graph has a third `news_features` input; its
ordered names and count are included in signed runtime metadata. Do not omit a failed member or
replace it with another seed after certification.

## 9. Assemble, sign, and package an immutable release

```powershell
python scripts/assemble_volatility_release.py `
  --candidate-dir C:\v8\certification\numeric-001\candidate `
  --output-dir C:\v8\release\numeric-001 `
  --private-key-path C:\secure\volatility-v1.private.pem `
  --public-key-path backend\release_keys\volatility-v1.public.pem

python scripts/package_volatility_release.py `
  --release-dir C:\v8\release\numeric-001 `
  --public-key-path backend\release_keys\volatility-v1.public.pem `
  --output C:\v8\archives\stocklstm-volatility-v8-numeric-001.zip
```

Store the archive outside Git at an immutable URL and record its SHA-256. The signed bundle binds
the certification report, universe, split, feature schemas, ensemble membership, ONNX files, and
parity evidence.

## 10. Deploy and smoke-test Render/Vercel

Configure Render with the immutable archive URL/SHA and public key. The service must start without
training or PyTorch. Verify `/health`, `/ready`, `/models`, and `/api/v2/forecast` for MSFT, NMM, a
Nasdaq security, a NYSE security, and an LSE security. Also test a tampered archive, wrong SHA,
unsupported horizon, invalid ticker, CORS, cache partitioning by model ID, and rollback.

A numeric release can become ready immediately after these gates. A news release must additionally
have a live point-in-time provider that reproduces the signed feature schema. Without it, the
correct production result is `news_input_unavailable`; never feed zero news or silently switch the
model family.

Enable `VITE_VOLATILITY_SERVING_ENABLED=true` only after the backend smoke passes. The UI must show
the signed model version and metric source and must not describe volatility bands as directional
price predictions.

## 11. Retention and rollback

Inventory and dry-run retention before deletion:

```powershell
python scripts/gc_v8_releases.py --root C:\v8\release --list-inventory
python scripts/gc_v8_releases.py `
  --root C:\v8\release `
  --active-release-id <active> `
  --previous-release-id <previous> `
  --audit-log C:\v8\release-gc.jsonl
```

Only add `--execute` after reviewing the plan. Active, previous, in-use leases, the audit window,
and minimum newest releases remain protected; staged debris and deletion are checksummed, locked,
race-rechecked, path-safe, and audited. Rollback changes the immutable archive pointer and SHA, then
forces release-state reload and repeats the smoke suite.

## Stop conditions

Stop without certification or deployment when any provenance, licensing, coverage, leakage,
identity, validation, ablation, parity, signature, resource, or production-input condition fails.
The accepted outcomes are a genuinely certified release or an explicit abstention—never a hidden
baseline, fabricated news vector, or relabelled diagnostic artifact.
