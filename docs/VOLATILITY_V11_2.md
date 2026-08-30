# StockLSTM V11.2 numeric PIT64 protocol

V11.2 is the numeric-only follow-up to the frozen V11.1 development
checkpoint. It expands the research panel to exactly 64 manually audited,
point-in-time securities and selects an independent route for each required
horizon: 1, 3, 5, and 7 market sessions.

## What is frozen

The protocol is identified by
`stocklstm-volatility-v11.2-numeric-pit64`. Its split is chronological 70/15/15
over unique market sessions. Seven-session labels are purged at boundaries and
each boundary has a 30-session embargo. The final 15% is a new encrypted
holdout. The development runner loads only train and validation files and has
no holdout key.

V11.1 remains an immutable historical checkpoint. No V11.1 model, digest,
candidate, or sealed partition is reused by V11.2.

## Universe and evidence

The accepted universe is exactly 64 securities. Every entry records a stable
identifier, exchange MIC, sector and volatility strata, ticker aliases and
transition dates, active membership intervals, OHLCV coverage, and source
provenance digests. Current constituents are not backfilled into historical
dates. Any identity or membership correction creates a new universe version.

Reports distinguish stock-origin observations from unique market sessions. A
panel row is identified by security, origin session, and target horizon. The
split digest hashes every row assignment, not merely row counts and endpoint
dates.

## Numeric candidates and routing

M2/news is disabled by protocol. V11.2 does not fabricate news features or
activate a multimodal candidate. Historical news is reserved for an
independent V12 protocol with its own universe and sealed holdout.

The candidate set is constant variance, realized-volatility persistence, HAR,
Ridge location residual, HistGradientBoosting location residual, and a numeric
neural residual model. The neural model evaluates epoch zero before its first
optimizer update; epoch zero exactly represents the HAR variance and zero
return-location prior. If updates do not improve validation CRPS, epoch zero is
restored.

Each horizon is selected independently. A learned route must beat HAR on the
untouched validation partition, pass the paired session-block uncertainty gate,
pass Holm correction across the four horizon decisions, have non-worse QLIKE,
and keep central 80% Student-t interval coverage in the preregistered
`[0.65, 0.95]` calibration band. It must also satisfy the three-seed stability
rule. Otherwise that horizon is routed to HAR.

## Statistical method

Losses are first averaged across securities sharing an origin session. A
circular moving-block bootstrap then resamples contiguous blocks of 20 sessions
for 10,000 replicates with seed 42. A learned route requires a two-sided 95%
interval whose upper bound for candidate-minus-HAR CRPS is below zero; the
one-sided p-value is computed from a null-centered block bootstrap rather than
from the observed effect distribution. M0
adequacy comparisons against constant variance and persistence use the same
session-block method and are corrected across eight comparisons at final
certification time.

All seed artifacts retain unrounded metrics, prediction and state digests,
fold ranges, runtime information, and stop reasons. Human-readable values may
be rounded, but machine-readable evidence is not.

## Sealing and certification boundary

The split command writes train and validation NPZ files separately from an
AES-256-GCM encrypted test payload. Every partition carries the stable
`security_id` beside its origin date, and the manifests bind the ordered
security/session identity digest. Certification therefore verifies that the
holdout rows still belong to the audited PIT64 universe rather than trusting
row counts alone. The key is generated outside the repository and never passed
to the development command. A future certification command writes its one-shot
open marker before decrypting, evaluates only the frozen bundle, writes an
immutable receipt, and removes temporary plaintext.

The V11.2 development run must finish with:

```text
sealed_test_status = LOCKED_UNOPENED
```

No V11.2 test metric, prediction, candidate ranking, or test descriptive
statistic may be produced during development.

## RTX execution

First materialize the audited, point-in-time panel from an immutable OHLCV
snapshot. This adapter is offline and refuses to infer a universe from a
current constituent list:

```powershell
python scripts/build_v11_2_panel.py `
  --snapshot-dir <immutable-snapshot-directory> `
  --universe-manifest <audited-pit64-universe.json> `
  --output <audited-panel>.npz
```

The command writes `<audited-panel>.npz.manifest.json` with the snapshot and
universe digests. The snapshot must carry an explicit provider-license
acknowledgement, the universe must contain exactly 64 accepted securities, and
every security must resolve through its point-in-time ticker and membership
intervals. The repository's secondary NDX100 cache is not a V11.2
certification input.

Compute the frozen feature-contract digest and prepare the dataset. The key
path must remain outside the repository:

```powershell
$schema = python -c "from research.volatility_forecasting.v11_2_protocol import feature_schema_digest; print(feature_schema_digest())"
python scripts/prepare_v11_2_dataset.py `
  --panel <audited-panel>.npz `
  --universe-manifest <audited-pit64-universe.json> `
  --output-dir artifacts/v11_2_numeric `
  --key-path "$env:USERPROFILE\.stocklstm\secrets\v11_2_holdout.key" `
  --schema-sha256 $schema
```

Run development only on the unsealed train/validation files:

```powershell
python scripts/run_v11_2_numeric_development.py `
  --dataset-dir artifacts/v11_2_numeric `
  --output-dir artifacts/v11_2_numeric/development_results `
  --device cuda
```

The runner automatically uses CUDA when available, but can be forced to CPU
for a controlled comparison. It does not accept a sealed-test key.

Before opening the holdout, run the non-decrypting audit:

```powershell
python scripts/pre_unseal_audit_v11_2.py `
  --dataset-dir artifacts/v11_2_numeric `
  --results-dir artifacts/v11_2_numeric/development_results `
  --output artifacts/v11_2_numeric/pre_unseal_audit.json
```

Certification is a one-shot operation. It writes
`sealed/SEALED_TEST_OPENED.json` before decrypting, never retrains or selects
on test rows, and creates an immutable metrics report and receipt. A failed
certification intentionally consumes the reserve and requires a new V11.2
dataset/protocol version rather than a rerun:

```powershell
python scripts/certify_v11_2_candidate.py `
  --dataset-dir artifacts/v11_2_numeric `
  --results-dir artifacts/v11_2_numeric/development_results `
  --key-path "$env:USERPROFILE\.stocklstm\secrets\v11_2_holdout.key" `
  --output-dir artifacts/v11_2_numeric/certification `
  --open-sealed-holdout
```

The exit status is zero only when every frozen learned route passes its
holdout CRPS/uncertainty/QLIKE/coverage gate (explicit baseline routes are
reported as baselines). The report's metric source is
`sealed_holdout_once`; it must not be relabelled as a walk-forward result.
