# V11.2 input-attestation schema

> **Archived:** V11.2 is permanently `INVALIDATED_OPENED`. This schema is
> retained to audit the historical attempt; it cannot make any new input or
> receipt certification-eligible. Use a new post-V11.2 protocol and the
> external evidence architecture in `FREE_CERTIFICATION_STACK.md`.

The historical V11.2 gate accepted only input data accompanied by signed,
replayable provenance receipts. A provider label, a manually edited JSON
field, or an unsigned statement that data is licensed was not an attestation.

Two receipts are required and they must be signed by different Ed25519 keys:

1. `v11_2_market_data_license` — signed by the market-data vendor or its
   licensing authority. It binds the immutable snapshot pooled checksum and
   grants historical OHLCV, model-training, derived-model-distribution, and
   production-inference rights.
2. `v11_2_pit64_membership` — signed by an independent reviewer or security
   master provider. It binds the exact 64-security master digest and declares
   point-in-time membership, stable security identity, and independent review.

Both receipts use schema version `1` and are signed over the canonical JSON
bytes of every field except `signature`:

```json
{
  "schema_version": 1,
  "attestation_type": "v11_2_market_data_license",
  "subject": {
    "kind": "immutable_ohlcv_snapshot",
    "id": "panel-licensed-example",
    "content_digest": "sha256:<snapshot-pooled-checksum>"
  },
  "issuer": {
    "name": "<vendor or licensing authority>",
    "key_id": "<sha256 of Ed25519 SubjectPublicKeyInfo DER>"
  },
  "issued_at": "2026-08-30T00:00:00Z",
  "rights": {
    "historical_ohlcv": true,
    "model_training": true,
    "derived_model_distribution": true,
    "production_inference": true
  },
  "independent_review": {
    "independent": true,
    "reviewer": "<named reviewer or organization>",
    "method": "<auditable review method>"
  },
  "evidence_files": {
    "snapshot_manifest": "sha256:<exact-file-digest>",
    "license_document": "sha256:<exact-file-digest>"
  },
  "signature_algorithm": "ed25519",
  "signature": "<base64 Ed25519 signature>"
}
```

For the PIT64 receipt, use `attestation_type` `v11_2_pit64_membership`,
`subject.kind` `pit64_security_master`, and set `subject.id` to the exact
`universe_version`. Its `subject.content_digest` is the digest returned by
`security_master_digest()` over the canonical 64-entry `securities` array.
Its rights object must contain `point_in_time_membership`,
`security_identity`, and `independent_review`, all set to `true`.

The receipt's `evidence_files` names and digests must exactly match the
`NAME=PATH` arguments supplied to the V11.2 builders. The snapshot manifest,
license document, membership archive, and any other evidence are hashed as
their exact bytes. Public keys are supplied out of band and copied into the
sealed dataset only as verification material; private signing keys and the
32-byte holdout key never enter the repository.

The panel sidecar must additionally record `panel_sha256`, the exact digest of
the generated NPZ. Preparation verifies that digest together with the attested
snapshot-manifest digest, universe-manifest digest, and receipt summary before
the holdout is sealed; a sidecar from an older or unrelated panel is rejected.

## Verification commands

Before asking either external issuer to sign, generate an unsigned request
pack containing the exact subjects, rights, and evidence-file hashes. This
command never creates a key, accepts a private key, signs a receipt, or opens a
holdout:

```powershell
python scripts/create_v11_2_attestation_requests.py `
  --snapshot-manifest <snapshot-manifest.json> `
  --universe-manifest <audited-pit64-universe.json> `
  --market-evidence snapshot_manifest=<snapshot-manifest.json> `
  --market-evidence license_document=<license-document> `
  --pit64-evidence membership_master=<membership-archive> `
  --output-dir <new-empty-attestation-request-directory>
```

The generated JSON documents have status
`unsigned_external_signatures_required`. They are request material only and
will not pass the verifier. The market-data licensor and PIT64 reviewer must
independently inspect the evidence, construct the receipt schema above, sign
it with their own Ed25519 key, and provide the corresponding public key via a
separate authenticated channel.

The following commands perform verification without opening the holdout:

```powershell
python scripts/check_v11_2_inputs.py `
  --panel <panel>.npz `
  --universe-manifest <universe.json> `
  --key-path "$env:USERPROFILE\.stocklstm\secrets\v11_2_holdout.key" `
  --snapshot-manifest <snapshot-manifest.json> `
  --market-attestation <market-receipt.json> `
  --market-public-key <market-public.pem> `
  --pit64-attestation <pit64-receipt.json> `
  --pit64-public-key <pit64-public.pem> `
  --market-evidence snapshot_manifest=<snapshot-manifest.json> `
  --market-evidence license_document=<license-document> `
  --pit64-evidence membership_master=<membership-archive>
```

The command returns non-zero unless every signature, key fingerprint, subject
digest, right, evidence-file checksum, panel identity, and external holdout
key check passes. A receipt can prove provenance only when the operator can
produce the actual vendor contract/licence and the independent reviewer can
be contacted; generating a receipt locally does not create those rights.
