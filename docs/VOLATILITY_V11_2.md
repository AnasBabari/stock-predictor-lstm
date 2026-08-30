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
AES-256-GCM encrypted test payload. The key is generated outside the repository
and never passed to the development command. A future certification command
writes its one-shot open marker before decrypting, evaluates only the frozen
bundle, writes an immutable receipt, and removes temporary plaintext.

The V11.2 development run must finish with:

```text
sealed_test_status = LOCKED_UNOPENED
```

No V11.2 test metric, prediction, candidate ranking, or test descriptive
statistic may be produced during development.

## RTX execution

Prepare the audited panel as an NPZ and run:

```powershell
python scripts/prepare_v11_2_dataset.py `
  --panel <audited-panel>.npz `
  --universe-manifest <universe>.json `
  --output-dir artifacts/v11_2_numeric `
  --key-path "$env:USERPROFILE\.stocklstm\secrets\v11_2_holdout.key" `
  --schema-sha256 <schema-sha256>

python scripts/run_v11_2_numeric_development.py `
  --dataset-dir artifacts/v11_2_numeric `
  --output-dir artifacts/v11_2_numeric/development_results `
  --device cuda
```

The runner automatically uses CUDA when available, but can be forced to CPU
for a controlled comparison. It does not accept a sealed-test key.
