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

**Current status (2026-08-31): no V11.2 production release exists.** The
attempted `artifacts/licensed_*` inputs are quarantined because their provider
and attestation fields are unsigned self-assertions; they do not satisfy the
receipt gate below. The diagnostic reserve remains unopened, while the
separate attempted reserve was opened once and failed certification. Do not
rerun that reserve or describe either input set as licensed production data.

V11.2 is now permanently `INVALIDATED_OPENED`. The preparation, one-shot
certification, release-assembly, and production-runtime entry points reject the
generation unconditionally. The remaining implementation and commands in this
document are retained only to explain and reproduce the historical research
method. They are not an active certification procedure. Any new attempt must
use a new protocol namespace, externally evidenced inputs, and a fresh reserve
controlled by an external custodian.

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

The residual LSTM geometry is defined once in
`research/volatility_forecasting/v11_2_model.py` and reconstructed through the
same builder during development, one-shot certification, and ONNX export. The
signed release records the architecture version and complete portable
architecture manifest so those stages cannot silently disagree about the
meaning of a frozen state dictionary.

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
from the observed effect distribution. M0 adequacy comparisons against constant
variance and persistence use the same session-block method, are corrected across
all eight comparisons, and require the HAR prior's central 80% interval coverage
to remain in `[0.65, 0.95]`. The development report records these intervals before
any route is frozen, and certification recomputes them on the sealed holdout.

All seed artifacts retain unrounded metrics, prediction and state digests,
including the epoch-zero CRPS/QLIKE and exact HAR-prior state digest, plus fold
ranges, runtime information, and stop reasons. Human-readable values may be
rounded, but machine-readable evidence is not.

## Sealing and certification boundary

The split command writes train and validation NPZ files separately from an
AES-256-GCM encrypted test payload. Every partition carries the stable
`security_id` beside its origin date, and the manifests bind the ordered
security/session identity digest. Certification therefore verifies that the
holdout rows still belong to the audited PIT64 universe rather than trusting
row counts alone. Machine-readable reports call these records
`stock_origin_observations` and separately report `unique_sessions`; a row is
not a market session. The key is generated outside the repository and never passed
to the development command. The historical certification command wrote its one-shot
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
universe digests. For a certification-eligible universe, this command also
requires two detached Ed25519 receipts: a vendor licence receipt bound to the
snapshot pooled checksum and an independent PIT64 security-master receipt
bound to the exact 64 identities. A prose `license.acknowledged` field,
provider name, or `certification_eligible=true` flag is not evidence and will
be rejected. The repository's secondary NDX100 cache is not a V11.2
certification input.

Each receipt must include the applicable model-training/derived-weight rights,
the attester key fingerprint, an independent-review declaration, and exact
SHA-256 digests for every supplied evidence file. The public keys are supplied
out of band; the verifier never creates or accepts a private key. The panel
builder accepts these arguments for an eligible input:

The complete signed-receipt schema and the operator evidence requirements are
defined in [V11.2 input-attestation schema](V11_2_ATTESTATION_SCHEMA.md).

```powershell
python scripts/build_v11_2_panel.py `
  --snapshot-dir <immutable-snapshot-directory> `
  --universe-manifest <audited-pit64-universe.json> `
  --output <audited-panel>.npz `
  --market-attestation <vendor-market-receipt.json> `
  --market-public-key <vendor-market-public.pem> `
  --pit64-attestation <independent-pit64-receipt.json> `
  --pit64-public-key <independent-pit64-public.pem> `
  --market-evidence snapshot_manifest=<snapshot-manifest.json> `
  --pit64-evidence membership_master=<membership-master-file>
```

Run the read-only input preflight before preparing the sealed dataset. It
fails closed unless the panel sidecar, audited universe, stable security/date
identities, signed input receipts, exact evidence files, and an external
32-byte holdout key are all present. It never opens or decrypts the holdout and
never prints key bytes:

```powershell
python scripts/check_v11_2_inputs.py `
  --panel <audited-panel>.npz `
  --universe-manifest <audited-pit64-universe.json> `
  --key-path "$env:USERPROFILE\.stocklstm\secrets\v11_2_holdout.key" `
  --snapshot-manifest <snapshot-manifest.json> `
  --market-attestation <vendor-market-receipt.json> `
  --market-public-key <vendor-market-public.pem> `
  --pit64-attestation <independent-pit64-receipt.json> `
  --pit64-public-key <independent-pit64-public.pem> `
  --market-evidence snapshot_manifest=<snapshot-manifest.json> `
  --pit64-evidence membership_master=<membership-master-file>
```

Compute the frozen feature-contract digest and prepare the dataset. The key
must already exist outside the repository and contain exactly 32 random bytes;
the sealing code never creates or replaces it. Each new reserve uses a new key
path and key value. Sealed format v3 records a SHA-256 key fingerprint in the
authenticated metadata, so supplying the wrong key fails before the one-shot
open marker is written. The key bytes themselves never enter an artifact.
path must remain outside the repository:

```powershell
$schema = python -c "from research.volatility_forecasting.v11_2_protocol import feature_schema_digest; print(feature_schema_digest())"
python scripts/prepare_v11_2_dataset.py `
  --panel <audited-panel>.npz `
  --universe-manifest <audited-pit64-universe.json> `
  --output-dir artifacts/v11_2_numeric `
  --key-path "$env:USERPROFILE\.stocklstm\secrets\v11_2_holdout.key" `
  --schema-sha256 $schema `
  --snapshot-manifest <snapshot-manifest.json> `
  --market-attestation <vendor-market-receipt.json> `
  --market-public-key <vendor-market-public.pem> `
  --pit64-attestation <independent-pit64-receipt.json> `
  --pit64-public-key <independent-pit64-public.pem> `
  --market-evidence snapshot_manifest=<snapshot-manifest.json> `
  --pit64-evidence membership_master=<membership-master-file>
```

Run development only on the unsealed train/validation files:

```powershell
python scripts/run_v11_2_numeric_development.py `
  --dataset-dir artifacts/v11_2_numeric `
  --output-dir artifacts/v11_2_numeric/development_results `
  --device cuda `
  --batch-size 256
```

The runner automatically uses CUDA when available, but can be forced to CPU
for a controlled comparison. Training uses deterministic chronological batches
with weighted gradient accumulation, so the objective remains the full-sample
mean while activation memory stays bounded on 6 GiB GPUs. It does not accept a
sealed-test key.

### Development-only hardware diagnostic

When licensed, independently attested PIT64 inputs are not yet available, the
existing secondary NDX100 cache may be used only to exercise the CUDA pipeline:

```powershell
python scripts/build_v11_2_diagnostic_pit64.py
```

The command resolves stable CIK/FIGI identity metadata, constructs eight
balanced research strata, writes a content-addressed snapshot, and stamps the
canonical universe with `certification_eligible=false`. The input preflight,
pre-unseal audit, and one-shot certifier reject that universe even if its files
are copied outside `data/ndx100/cache`. Diagnostic development results may
inform engineering and future candidate design, but they cannot be described
as certified, used to open the holdout, signed, or deployed.

Before opening the holdout, run the non-decrypting audit:

```powershell
python scripts/pre_unseal_audit_v11_2.py `
  --dataset-dir artifacts/v11_2_numeric `
  --results-dir artifacts/v11_2_numeric/development_results `
  --output artifacts/v11_2_numeric/pre_unseal_audit.json
```

The V11.2 certification was a one-shot operation. It wrote
`sealed/SEALED_TEST_OPENED.json` before decrypting, never retrains or selects
on test rows, and creates an immutable metrics report and receipt. A failed
certification consumed the reserve. The command below is retained as historical
documentation and now raises `V112CertificationRetiredError`:

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

## Production release conversion — permanently disabled

V11.2 cannot produce a release, regardless of any local report or metadata
change. The historical conversion command now raises
`V112CertificationRetiredError` before reading candidate artifacts:

```powershell
python scripts/assemble_v11_2_release.py `
  --results-dir artifacts/v11_2_numeric/development_results `
  --certification-dir artifacts/v11_2_numeric/certification `
  --output-dir artifacts/releases/volatility-v11-2 `
  --private-key-path "$env:USERPROFILE\.stocklstm\secrets\volatility-release.key" `
  --public-key-path backend/release_keys/volatility-v1.public.pem
```

The research graph remains testable for historical reproducibility. Runtime
metadata containing `v11.2`, `v11_2`, or `v11-2` in its release identity is
rejected before an ONNX session can load. No private key, release binary, or
opened holdout belongs in Git.
