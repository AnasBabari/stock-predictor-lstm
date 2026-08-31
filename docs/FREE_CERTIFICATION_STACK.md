# External Certification Architecture — Planned, Not Certified

## Current status

The repository is **0/4 certification-valid**. It contains acquisition,
construction, hashing, and verification tooling, but no current artifact is
production certification evidence.

Production serving must remain fail-closed:

```http
HTTP/1.1 503 Service Unavailable
```

```json
{
  "status": "abstain_no_certified_model"
}
```

V11.2 is permanently `INVALIDATED_OPENED`. Its reserve was opened once and the
candidate failed. The dataset, secret, result, signature, or any derivative of
that generation cannot be reused to authorize a release.

## What the repository tooling can do

- `scripts/candidate_ohlcv_acquisition_tool.py` reports which acquisition
  routes are configured and prints a permission-request template. It does not
  download data, infer rights, or produce a licensed manifest. Its status is
  always `permission_pending` and `certification_eligible=false`.
- `scripts/candidate_pit64_construction_tool.py` validates and packages an
  operator-supplied candidate universe and source table for independent
  review. It does not manufacture permanent identifiers, membership dates, or
  external-review evidence.
- `scripts/candidate_version_candidate_tool.py` deterministically hashes a
  candidate artifact tree. It does not tag Git, sign a manifest, create a
  transparency-log entry, or make a certification claim.
- `research/volatility_forecasting/certification_gate_v12.py` validates an
  externally supplied evidence manifest and invokes the real `cosign
  verify-blob` command. It cannot create keys, signatures, holdout secrets, or
  substitute receipts.

These tools make evidence easier to request and audit. They do not replace the
external evidence.

## Evidence required to move from 0/4 to 4/4

### 1. Externally licensed OHLCV

Obtain a vendor export plus a written licence or permission covering the
actual uses: historical research, model training, derived-model distribution,
and production inference. Record the real provider, export identifier,
retrieval timestamp, licence or permission reference, immutable file hashes,
and an independently checkable copy or digest of the permission.

The source must be external, the permission status must be `VERIFIED`, and the
export hashes must be independently verified. A copied diagnostic cache with a
new manifest is not a new licensed source.

### 2. Independently reviewed PIT64 master

Construct a point-in-time security master for every selected security and
every membership boundary in the evaluation period. Each record needs stable
identifiers, exchange identity, effective membership intervals, and exact
source references and digests. An independent reviewer must verify the exact
artifact and methodology, including every transition boundary.

The construction tool's output is only a review candidate. Self-attestation is
not independent review.

### 3. Genuine external Cosign verification

Freeze one evidence manifest containing the code commit, OHLCV digest, PIT64
digest, dataset version, split/evaluation specification, and holdout
commitment. An independent reviewer signs that exact byte sequence outside the
repository using real Cosign/Sigstore. Certification requires successful
verification with `cosign verify-blob`, either against a pinned public key or a
pinned keyless identity and OIDC issuer.

The repository contains no simulated Sigstore generator. JSON that merely
resembles a Sigstore bundle is not acceptable.

### 4. Fresh externally controlled holdout

After the candidate and protocol are frozen, an external custodian generates
a new secret that the model developer has never possessed, commits only its
hash, and independently prepares the new reserve. The pre-unseal audit must
confirm that the secret was not revealed and the reserve was never opened.
The one-shot evaluation writes its opened marker before decryption and freezes
the result regardless of pass or fail.

No locally generated secret and no V11.2 material may participate.

## Machine gate

The verifier requires all of the following:

```text
ohlcv.permission_status == VERIFIED
ohlcv.source_is_external == true
ohlcv.hash_verified == true
pit64.completeness_verified == true
pit64.external_reviewer_verified == true
signature.real_cosign_verification == true
signature.simulated == false
holdout.generated_externally == true
holdout.previously_revealed == false
reserve.previously_opened == false
code_hash_matches == true
dataset_hash_matches == true
real cosign verify-blob succeeds for the exact evidence manifest
```

The values below are unconditional failures wherever they appear as evidence:

```text
permission_pending
simulated
demo
self_attested
copied_diagnostic
locally_generated
previously_opened
invalidated
```

Required negative field names such as `reserve.previously_opened` are allowed
only because their typed values are separately required to be exactly `false`.
A free-text value of `previously_opened` still fails.

## External reviewer handoff

The repository owner supplies the reviewer with the unsigned evidence
manifest, PIT64 candidate, source table, methodology, verifier, and permitted
digests. Raw vendor data is shared only if the licence permits it. The reviewer
obtains these materials independently, performs the review, signs the exact
manifest, publishes the public verification material, and keeps the holdout
secret outside the developer's custody until the prescribed unseal.

Illustrative reviewer-side commands:

```bash
cosign sign-blob --key reviewer.key --bundle review.sigstore.json evidence.json
cosign verify-blob evidence.json --bundle review.sigstore.json --key reviewer.pub
```

Keyless verification may be used only with a pinned certificate identity and
pinned OIDC issuer. The generated private key, holdout secret, and any signing
credential must never enter this repository.

## Candidate-tool examples

```powershell
python scripts/candidate_ohlcv_acquisition_tool.py
python scripts/candidate_pit64_construction_tool.py `
  --universe path\to\pit64.json `
  --sources path\to\sources.json `
  --output path\outside\the\inputs\pit64-review-request.json
python scripts/candidate_version_candidate_tool.py `
  --root path\to\candidate-tree `
  --output path\outside\candidate-tree\version-manifest.json
```

Every output remains non-certifying until the complete external chain passes.
There is no command in this repository that can convert a candidate into
certification evidence by declaration.
