# Volatility v8 preregistration — historical temporal + cross-asset transfer

Status: **preregistered; no v8 candidate trained, no certification opened, no release.**

Implementation note (2026-08-27): the repository now contains fail-closed universe/market/news
provenance builders, security-ID/MIC-bound splits, paired five-fold news ablation, GPU candidate
training, one-shot numeric/news certification, ONNX parity, signed release packaging, and release
retention. This is implementation readiness only. No real attested input cohort or v8 model evidence
exists yet, and the sealed 15% test remains unopened. The legacy 69-ticker panel is diagnostic-only.
A certified news release also requires a production point-in-time provider capable of reproducing
the signed news feature schema; otherwise serving must abstain.

This document freezes the v8 research cycle **before** any sealed test rows are inspected. It must be read together with [VOLATILITY_V7_PREREGISTRATION.md](VOLATILITY_V7_PREREGISTRATION.md) and [GLOBAL_MODELS.md](GLOBAL_MODELS.md). v7 remains a separate future-prospective experiment and is not modified.

## Why v8 exists

v7 is a stronger prospective test (252 future target-complete origins + asset-transfer) but cannot mature until ~`2027-10-11` (`scripts/check_v7_maturity.py:35` reports `not_mature`). Production requires predictions now. v8 provides a **historical, reproducible** certification that is strictly weaker than v7 and must be labelled differently.

Evidence labels:
- v7 future success: `locked_purged_walk_forward`
- v8 historical success: `locked_historical_temporal_test_plus_asset_transfer` (primary v8 route)

Never label a v8 result `locked_purged_walk_forward` or `locked_v7_prospective_walk_forward`.

## Frozen v8 identities

- Protocol: `global-volatility-distribution-v8-news-transfer`
- Model family: `global-volatility-news-fusion-v8`
- Target: `future-rv-total-v1` (same realized-variance definition as v7, extended to news-aware variant `future-rv-total-v1-news-v8` when news is present)
- Feature contract: Deployable Schema v5 causal + versioned news feature contract `news-v1` (see `research/volatility_forecasting/news.py:1`)
- Input window: 60 market sessions
- Horizons: 1, 3, 5, 7, 14, 30 (required: 1,3,5,7)
- Seeds: 41, 42, 43
- Split: chronological 70/15/15 train/validation/test by forecast-origin timestamp (not random)
- Purge: every label overlapping a split boundary removed
- Embargo: 30 sessions after each boundary (≥ max horizon)
- Asset-transfer holdouts: point-in-time, predeclared before any training
- News cut-off: `available_at = max(published_at, first_seen_at)` strictly < origin; ambiguous timestamps quarantined
- Certification scope: `historical_temporal_test_plus_asset_transfer`
- Metric source: `locked_historical_temporal_test_plus_asset_transfer`

## Universe definition (point-in-time, no survivorship bias)

v8 is not single-ticker. It covers three listing MICs (XNAS, XNYS, XLON) plus point-in-time S&P 500
index membership as a cohort tag; S&P 500 is not a fourth exchange.

### S&P 500
- Historical constituent list with `membership_start`, `membership_end`, source snapshot ID.
- Do not apply current S&P 500 membership backward.
- Store `ticker, company_name, ISIN, FIGI/CIK, primary_exchange, sector, industry, membership_start/end, source, source_snapshot_id`.

### Nasdaq
- Primary listing on Nasdaq (XNAS), common equity only, exclude ETFs/ETNs/warrants/preferred/rights/funds unless explicitly preregistered.
- Minimum history `756` sessions + minimum point-in-time liquidity.
- Include delisted/acquired where data available; use historical listing status.

### NYSE
- Primary listing on NYSE (XNYS), same filters as Nasdaq.

### LSE
- LSE (XLON) ordinary shares only (exclude investment trusts, ETFs, GDRs, secondary listings unless explicitly included).
- Store local currency `GBP/GBX`, local calendar, MIC, and normalize timestamps to `Europe/London` + `UTC`.

### Deduplication
- By `ISIN` + `FIGI` + `CIK` + `provider_security_id`, not ticker alone.

### Initial v8 cohort (configurable, deterministic seed)
- All available point-in-time S&P 500 members
- Liquidity-stratified Nasdaq / NYSE / LSE cohorts
- ≥ several hundred total, ≥25–50 per exchange group for transfer testing
- `NMM` and `MSFT` only after `scripts/check_v7_maturity.py` + asset-transfer audit confirms they were holdouts in v7 and were never used for feature selection/scaler/HPO.

The frozen manifest is `universe-v8-manifest.json` with deterministic seed and source checksums. Code: `research/volatility_forecasting/universe_v8.py`.

## Market snapshot

Immutable, content-addressed, stores raw + adjusted OHLCV:

```text
timestamp, session_date, open, high, low, close, adjusted_close,
volume, dividends, split_factor, ticker, security_id, exchange,
currency, timezone, provider, retrieved_at
```

Rules: exchange-specific calendars, no forward-fill across non-trading sessions, preserve halts/missing sessions explicitly, deterministic split/dividend handling, provider/license/retrieved_at checksummed. No overwrite – new snapshot gets new `panel_id`/`pooled_checksum`. Builder refuses historical-prefix drift (same check as `research/volatility_forecasting/prospective.py:25` `validate_prospective_panel_manifest`).

Module: `research/volatility_forecasting/data.py:85` unchanged; panel provenance via `backend/panel/snapshots.py:66` `canonical_csv`/`checksum_text`.

## News snapshot

Historical, licensed provider with stable `published_at` + `first_seen_at`. Raw article stored only in licensed training environment if redistribution restricted; release contains derived features/metadata only.

Raw schema:
```json
{"article_id":"...","canonical_url":"...","title":"...","published_at":"...","first_seen_at":"...","provider":"...","source_name":"...","language":"en","mentioned_entities":[],"provider_tickers":[],"license_id":"...","retrieved_at":"...","content_hash":"...","snapshot_id":"..."}
```

Timestamp policy: `available_at = max(published_at, first_seen_at)`; ambiguous → quarantined, not used in certification, logged. Daily-only timestamps → assigned to next trading session.

Deduplication by `canonical URL` + `normalized title` + `content_hash` + `source identity`.

Entity mapping via point-in-time security master (provider tag, ISIN/FIGI/CIK, aliases, subsidiaries).

Taxonomy (versioned): `earnings, guidance, merger_acquisition, capital_raise, dividend, buyback, product, regulation, litigation, credit, bankruptcy, management_change, labor, supply_chain, cybersecurity, geopolitical, macro, central_bank, commodity, energy, currency, interest_rates, analyst_action, insider_activity, other`.

Coverage, missing, deduplication stats recorded. If historical archive unavailable: do **not** synthesize news, do **not** claim `news_enabled=true`; certify `global-volatility-v8-numeric` separately and keep news as uncertified branch.

Module: `research/volatility_forecasting/news*.py`, `research/volatility_forecasting/gdelt*.py`.

## News features (causal, leakage-safe)

Four levels: security, sector/industry, exchange/country, global macro.

Per asset-origin:
```text
article_count, unique_source_count, positive/negative/neutral probabilities,
sentiment_dispersion, negative/positive tail scores, novelty, event_intensity,
topic distribution, source disagreement, pooled embeddings, missing_news_indicator
```

Windows ending ≤ origin: `1h, 4h, 1 session, 3 sessions, 5 sessions, 20 sessions`.

Encoders: FinBERT-type finance encoder, versioned (`encoder_version, tokenization_version, calibration_version`).

Novelty = embedding distance from recent articles + new entity/topic surprise.

Intensity = article volume anomaly, source breadth, cross-source disagreement, market-wide/sector-wide volume.

Cross-asset context (origin information set only): VIX, WTI/Brent, gold, USD basket, Treasury yields, credit spread, broad-market/sector/exchange returns.

All scalers/vocabularies fitted on **train only**.

## Target

- Log return `r(i,t)=log(adj_close(t)/adj_close(t-1))`
- Realized volatility `RV(i,t,h)=sqrt(sum(r(t+k)^2 for k=1..h))`
- Training target `log(RV + epsilon)` with epsilon small positive
- Horizons 1,3,5,7,14,30; outputs: point `log RV`, `RV`, quantiles, optional jump probability.

Definition matches `research/volatility_forecasting/data.py:121` `cumulative_variance_target` / `realized_variance_proxies` with `window_size=60`.

## Chronological 70/15/15 split with purge/embargo

Algorithm (immutable `split-v8-manifest.json` before training):

1. Build all valid forecast origins (window 60 + max horizon 30 complete).
2. Sort by canonical origin timestamp (UTC, per-exchange `post_close_next_session`).
3. Boundaries at 70% and 85% of sorted origins.
4. Remove rows whose target windows cross a boundary (purge).
5. Apply embargo 30 sessions after each boundary.
6. Fit all scalers only on train rows.
7. Fit vocab/encoder calibration/feature selection only on train rows.
8. Use validation for early stopping/HPO/candidate selection.
9. Seal test partition – not inspected before candidate freeze.
10. No test data enters training, no future news enters historical features.

Quality gates: global 70/15/15 ±2%, per-exchange ≥15% coverage per split, per-sector coverage, per-horizon completeness, no duplicate origin/asset, no label overlap, no test origin < validation origin, no scaler leakage, no cross-sectional statistic using future values.

## Asset-transfer holdout

Declared **before** training in `split-v8-manifest.json`:

- Exclude holdouts from train + validation, not used for HPO/selection/blending.
- Evaluate only after freeze.
- Include S&P 500 + Nasdaq + NYSE + LSE where data allows, ≥ several hundred total.
- Include `NMM/MSFT` only after contamination audit (maturity monitor + `research/volatility_forecasting/folds.py:60` `select_asset_holdouts` check).

Evidence reports both:
```text
historical temporal test (in-distribution assets, future origin window)
asset-transfer (unseen assets, same future window)
```

## Baselines (before neural training)

```text
last-realized-volatility persistence, rolling mean, EWMA, HAR-RV,
GARCH, GJR-GARCH, shrunk cross-sectional mean, Ridge, ElasticNet, DLinear
numeric-only, news-only, shuffled-news, timestamp-shifted-news
```

Shuffled/shifted controls are mandatory leakage tests.

## Candidate families

Numeric-only: Ridge, ElasticNet, DLinear, TCN, GRU, LSTM, GARCH-LSTM, residual TCN.

News-only: pooled sentiment linear, event-encoder, sequence.

Fusion (primary):
```text
Numeric branch: [60, 26] -> projection -> LSTM/TCN 64->32 + asset/exchange embeddings
News branch: causal window pooling -> event/sentiment projection 64 + attention
Fusion: gated residual -> dense 64 -> horizon heads
output = softplus(raw) + epsilon
loss = QLIKE + weighted Huber/pinball for quantiles
```

Seeds 41,42,43, early stopping on validation, gradient clipping, deterministic where supported, artifacts outside Git + sha256.

## News ablation & incremental-value gate

Identical splits, candidates A–J:
```text
A numeric-only, B news-only, C numeric+sentiment, D numeric+events,
E numeric+sentiment+events, F numeric+full embeddings, G numeric+full+cross-asset,
H shuffled-news, I timestamp-shifted, J count-only
```

Report QLIKE/MAE/MSE/RMSE, calibration/coverage, relative vs persistence, per-exchange/sector/regime, coverage/latency stats.

News is useful **only if**: beats numeric-only on sealed test **and** preregistered gate **and** not reproduced by shuffled/shifted **and** survives bootstrap/DM/Holm **and** not isolated to one ticker/period **and** remains calibrated.

Otherwise the honestly certified release is `global-volatility-v8-numeric` with `news_status=not_certified`.

## Selection rule

Use train+validation only, QLIKE primary vs `adaptive_calibrated_har_c2c_v1` (`research/volatility_forecasting/contracts.py:40`).

A profile/horizon is eligible only when volatility promotion true for every required horizon (1,3,5,7) and every seed. If neither profile eligible → abstain. Otherwise challenger displaces incumbent only when median challenger/incumbent relative QLIKE ≤0.995 and worst horizon ≤1.01 (same thresholds as `research/volatility_forecasting/prospective.py:98` `ProspectiveCycleSettings`).

Never use test metrics for selection.

## v7 separation

- Do not modify `VOLATILITY_V7_PREREGISTRATION.md` or `global-volatility-distribution-v7-prospective`.
- Do not reuse `global-volatility-ensemble:450567f...` weights.
- Do not reuse v7 development metrics as certification metrics.
- New model identity `global-volatility-news-fusion-v8:<digest>` (or `global-volatility-v8-numeric:<digest>` if news not certified).
- New cache namespace `volatility-panel-examples-v8-news-transfer`.
- Keep `check_v7_maturity.py` tracking future v7 separately.

## Certification (one-shot, sealed)

Create `scripts/certify_v8_candidate.py` (do not weaken `scripts/certify_prospective_volatility_candidate.py`):

Verifies: candidate `artifact_role=prospective_v8_development_candidate`, protocol `global-volatility-distribution-v8-news-transfer`, market+news+universe+split checksums, feature/target order, test sealing, holdout list, every member+horizon, no retraining, no future news, no current-membership leakage, NMM/MSFT coverage where applicable.

Before opening test: create `v8-holdout-opened.json` marker (one-shot). Evaluate exactly once, write `v8-locked-certification.json` atomically. On success materialize `artifact_role=locked_v8_certification_candidate` with `release_eligible=true` and `metric_source=locked_historical_temporal_test_plus_asset_transfer`; on any gate failure `release_eligible=false`, no partial candidate.

Report distinguishes `development/validation/sealed-test/asset-transfer/ablation` metrics, per-horizon/per-exchange, bootstrap/DM/Holm, calibration, and `news_improved_over_numeric` flag.

## ONNX, signing, deployment

Per-member ONNX export + research↔ONNX parity within tolerances (feature order, horizon map, scaler, non-negative volatility, missing-news path). Immutable `release-v8/` with manifest `protocol, candidate, market/news/universe/split checksums, feature order, ONNX parity, metric_source, archive_sha256` – signed `Ed25519`, verified via `backend/release_keys/volatility-v1.public.pem` (`render.yaml:24`) and `backend/release/bundle.py:122` `verify_release`.

Render (`VOLATILITY_SERVING_REQUIRED=true` + `VOLATILITY_RELEASE_ARCHIVE_URL/SHA256`) verifies signature/checksums/schema/size/CORS before caching. Vercel `VITE_VOLATILITY_SERVING_ENABLED=true` only after `/ready`+`/models`+`MSFT/NMM/AAPL/Nasdaq/NYSE/LSE`+tamper/CORS/rollback pass. UI shows `server_artifact_loaded` + `locked_historical_temporal_test_plus_asset_transfer`, never `locked_purged_walk_forward` for v8.

Bundle retention/GC before open beta (active + previous + audit protection, dry-run, generation-aware locking, audit logs).

## Contingency

If historical news unavailable: certify numeric v8 now, keep news branch as uncrowned experiment, add live news only as context – never claim `news_enabled=true` without sealed historical archive. Labels then `global-volatility-v8-numeric`, `news_status=not_certified`.

## Architecture selection (validation-only, sealed-test-safe)

The numeric v8 candidate is selected by `scripts/run_v8_architecture_search.py`
(a validation-only sweep that never opens the sealed test partition). The
script is intentionally narrower than the preregistered research cycle:

- **Search space**: encoder family (`tcn` / `patch_transformer`),
  channels, dropout, learning rate, weight decay, baseline
  regularization. Only train + validation rows enter training; the
  search report is bound to the chronological 70/15/15 split manifest.
- **Ranking**: eligible candidates first, then by worst required-
  horizon relative QLIKE (lower-is-better). Mean and worst QLIKE
  ratio upper 95% are reported alongside the point estimate.
- **Winner path**: the search winner is materialized as a
  `prospective_v8_development_candidate` only if its validation gates
  pass **and** the universe manifest's `coverage_certifiable=true`
  flag is set. Otherwise the search report stays a
  `rejected_v8_development_evidence` artifact, never signed for
  release.
- **Sealed-test guarantee**: the search script accepts
  `train_indices` and `validation_indices` only — never
  `temporal_test_indices`, `asset_transfer_test_indices`, or
  `pooled_test_indices`. A fail-closed fingerprint test in
  `research/tests/test_v8_architecture_search.py` enforces this
  invariant.
- **Architecture-evidence reference**: the current best validation-
  only architecture is recorded in the latest
  `arch-search-cuda-*/best-config.json` and detailed in the matching
  `search-report.json` plus `docs/SUMMARY.md`. The dev panel used for
  the published search (a sparse 69-ticker U.S. snapshot without the
  LSE cohort) cannot itself produce a certifiable v8 — the moving-
  block bootstrap CIs on the small validation selection region do not
  shrink below the 1.0 ratio gate — so all dev artifacts remain
  `rejected_v8_development_evidence` until the four-market panel is
  acquired.

## Permitted claims before v8 certification

May say v8 is preregistered historical+transfer program with clearly labelled development metrics. Must not claim v8 is certified, must not claim future prospective evidence, must not present baseline as learned, must not present price accuracy for volatility regression. Production API stays abstaining until signed v8 passes.
