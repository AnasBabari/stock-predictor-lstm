# Verified Model Release V12 protocol

VMR-V12 is a transparent, reproducible research and portfolio release
protocol. It is not third-party certification, independent certification, or
an assertion that an external authority has reviewed this repository.

## Frozen identity

```text
Protocol: VMR-V12
Benchmark: USE64-HIST-v1
Universe design: HISTORICAL_FIXED_V1
Terminal policy: TERMINAL_EVENT_POLICY_V1
V11.2 status: INVALIDATED_OPENED
Production status before successful completion: abstain_no_verified_release
```

This document freezes protocol metadata only. It does not create data rights,
a security master, a model candidate, an evaluation result, or a release.
Data acquisition, universe construction, public-randomness retrieval, GPU
training, official evaluation, and deployment are later boundaries.

## Release condition

A V12 artifact is eligible only when all six independent gates pass:

```text
release_eligible =
    G1_DATA_USAGE_COMPLIANT
    AND G2_BENCHMARK_INTEGRITY_VERIFIED
    AND G3_CANDIDATE_PRECOMMITTED
    AND G4_OFFICIAL_EVALUATION_VALID
    AND G5_SCIENTIFIC_ACCEPTANCE
    AND G6_RELEASE_PROVENANCE_VERIFIED
```

No editable `verified` or `release_eligible` field may override computed gate
results. A strong model with invalid rights, a valid build with weak metrics,
or a good evaluation that cannot be bound to the released artifact all fail.

## G1 — data usage compliance

The gate verifies a normalized, human-reviewed rights record and its evidence
hashes. It does not interpret arbitrary legal language autonomously and is not
automated legal certification. The record must identify the provider, dataset,
immutable raw-data digest, rights document and terms version, reviewer, review
scope, permitted uses, and transformations.

The required uses for a deployed model are historical analysis, model
training, and prediction deployment. Missing or unknown rights, pending
permission, unsupported uses, copied diagnostic data, and a standalone
`permission_status=VERIFIED` claim without supporting evidence fail closed.
Raw redistribution may be false when raw records are not redistributed.

## G2 — benchmark integrity

`USE64-HIST-v1` uses `HISTORICAL_FIXED_V1`. The historical eligibility snapshot
must be at or before evaluation start, and selection may use only information
known at that snapshot. A present-day surviving-stock list must not be projected
backwards and presented as historical membership.

The manifest binds selection inputs, selection code, permanent identifiers,
criteria, exchanges, security types, liquidity/history thresholds,
deduplication, corporate actions, missing data, survivorship, and terminal
event policy. Required integrity controls include monotonic timestamps,
origin-causal features, train-only transforms, purge and embargo at least as
large as the maximum horizon, asset-transfer separation, deterministic
reconstruction, and explicit missing-data treatment.

Automated integrity checks prove internal consistency and protocol compliance.
They do not independently prove the truth of an untrusted historical source.

## TERMINAL_EVENT_POLICY_V1

* Permanent security identity survives ticker and name changes.
* Splits and distributions follow the declared adjustment source.
* Securities remain benchmark members after historical selection.
* A security contributes observations only while valid data exist.
* Origins requiring unavailable post-terminal labels are right-censored.
* Missing post-delisting prices are never treated as unchanged or zero.
* Terminal returns are used only when supplied by an approved authoritative source.
* Missing vendor observations generate explicit quality flags.
* No security is removed merely because it later failed, merged, or delisted.
* Exclusions operate at affected observation/origin level, not retrospectively at security level.
* Counts and reasons for all censored observations are reported by security and event type.

No later implementation may carry the last price forward after delisting,
replace a missing terminal price with zero, delete a security because it later
failed, or treat a missing post-terminal label as an ordinary observation.

## G3 — candidate precommitment

The canonical candidate manifest binds the protocol and benchmark, terminal
policy, model implementation, code commit, training data, universe, feature and
target schemas, split construction, training configuration, seeds, weights,
baselines, acceptance thresholds, statistical tests, release format,
randomness policy, and the evaluation-partition catalogue.

Manifests use canonical UTF-8 JSON: sorted keys, compact separators, explicit
number handling (JSON integers and finite JSON floats are preserved exactly as
encoded by the VMR-V12 Python canonicalizer), no NaN or Infinity, and timestamp
strings with an explicit timezone. Any change
to data, weights, features, targets, thresholds, baselines, split logic,
multiple-comparison correction, partition catalogue, or randomness policy
creates a new candidate generation.

The candidate digest must be externally anchored before the future randomness
event. Git history alone is not treated as an immutable external timestamp.

The evaluation ledger is append-only. A record is validated and stored as
canonical text; appending returns a new ledger, while update and delete
operations are intentionally absent. Gate evaluation additionally requires
exactly one `official` record for the candidate and allows later
`reproduction` records only when they reference that official record.

## G4 — official evaluation

The enforceable rule is exactly one admissible official evaluation record for a
candidate generation. This does not claim that no one ever ran a local test.
Additional runs are `reproduction` records that reference, and cannot replace,
the official record.

The provider-neutral policy is:

```text
Primary: drand
Secondary: NIST Beacon V2
```

Before freeze, the protocol must bind the provider, root of trust, round or
future-time rule, endpoint, signature verification, canonical encoding,
domain-separation string, hash-to-selection algorithm, valid partition
catalogue, fallback conditions, timeout, and no-fallback-after-observation
rule. Randomness selects only from preconstructed valid temporal/asset blocks;
individual-row selection is forbidden.

The machine-readable policy fixes `drand` as primary and `nist_beacon_v2` as
the only secondary provider. A secondary record is admissible only with an
objective precommitted failure reason and a verified secondary chain/root;
changing providers after seeing a partition or result is invalid.

The only fallback reasons are
`expected_pulse_unavailable_after_committed_deadline`, `signature_invalid`,
`chain_verification_failure`, `canonicalization_failure`, and
`precommitted_provider_unavailable`. A result that is merely inconvenient
never qualifies.

Fallback is allowed only for a precommitted objective failure such as an
unavailable or invalid pulse. An undesirable partition or result is never a
fallback reason.

## G5 — scientific acceptance

Scientific acceptance is separate from provenance. It requires a precommitted
test, complete leakage-free evaluation, required baselines, a practically
meaningful improvement, uncertainty and calibration requirements, no material
asset-group regression, finite metrics, and complete per-horizon, exchange,
volatility-regime, liquidity-group, and asset-transfer reporting.

Required baseline identifiers include `persistence`, `HAR`, `EWMA`, and the
strongest eligible development baseline. Unset thresholds are not permissive
defaults; they fail the gate.

## G6 — release provenance

The local RTX training record binds the clean Git commit, environment locks,
Python/CUDA/GPU runtime, command, configuration, data/universe/schema hashes,
seed, model output, logs, metrics, and times. A dirty training worktree is not
release eligible.

The GitHub release build uses minimal permissions, SHA-pinned Actions, a
protected production environment, immutable digests, verified repository and
workflow identity, and no signing from untrusted pull requests. GitHub
attestations prove where and how an archive was built; they do not prove
scientific correctness, data licensing, absence of leakage, or local GPU
training provenance. If Cosign is used, verification must use a genuine
`cosign verify-blob` bundle, never a simulated receipt.

The release-provenance schema records and requires each of those controls,
including scoped `id-token` use, before G6 can pass. Archive, manifest, model,
candidate, training-input, and training-manifest digests are cross-bound so a
valid build cannot silently publish different training inputs.

The protocol package only validates the shape and binding of these future
records. It does not manufacture an external rights review, a Sigstore
attestation, an RTX run, or a release archive.

## V11.2 permanent invalidation

V11.2 is `INVALIDATED_OPENED`. Its candidate, weights, reserve, key, reports,
manifests, signatures, status, and evaluation records remain inspectable as
historical evidence but are unusable for V12. Generation checks are structural;
renaming a file or editing a status JSON cannot bypass them.

## Production behavior

Until all six gates pass, production remains fail closed:

```json
{"status":"abstain_no_verified_release"}
```

The successful state is `verified_release` and must include the protocol,
candidate, universe, evaluation, and release artifact identities. This protocol
phase does not enable that state.

Existing V11/V10 API consumers may still receive the legacy
`abstain_no_certified_model` status while their compatibility route remains in
service. That legacy string is not a VMR-V12 status and must not be used to
authorize a release; a later runtime migration must expose the canonical
`abstain_no_verified_release` value without weakening the 503 fail-closed
behavior.
