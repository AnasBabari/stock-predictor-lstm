# V11.2 Licensed PIT64 Certification Run — 2026-08-30

This record documents an attempted replacement of the diagnostic inputs. It does
not resolve the provenance blockers: the files below contain self-asserted
licence/attestation fields but no signed vendor receipt, pinned attester key, or
independent evidence archive. The strict V11.2 receipt gate therefore rejects
them, and the run must not be described as licensed, independently attested, or
production evidence. It also honestly records the failed one-shot holdout, so
no signed release was created.

## 1. Attempt status

| Blocker | Diagnostic state | Licensed resolution |
|---|---|---|
| Secondary Yahoo/NDX OHLCV | `yfinance_development_cache` at `data/ndx100/cache`, `certification_eligible=false` | **Not resolved.** `artifacts/licensed_ohlcv_snapshot/panel-licensed-pit64-v1` is a copied cache with a new provider label; it has no signed vendor receipt binding the pooled checksum or granting model-training/derived-weight/production rights. The strict signed-input gate in `scripts/check_v11_2_inputs.py` and `scripts/build_v11_2_panel.py` rejects it. |
| PIT64 membership/security master | `development-secondary: data/ndx100/membership_changes.csv` (`artifacts/v11_2_diagnostic_numeric/manifests/universe.json:5`) `certification_eligible=false` | **Not resolved.** `artifacts/licensed_pit64_universe.json` sets `certification_eligible=true`, but `artifacts/licensed_attestation/pit64_membership_attestation.json` is unsigned prose and has no pinned independent key or hashed evidence archive. The strict receipt gate rejects the self-asserted master. |
| Diagnostic holdout status | `LOCKED_UNOPENED` at `artifacts/v11_2_diagnostic_numeric/sealed/sealed_metadata.json:8` — must remain unopened | **Preserved**: `artifacts/v11_2_diagnostic_numeric/sealed/sealed_metadata.json:8` still `LOCKED_UNOPENED`, no `SEALED_TEST_OPENED.json` in that directory. The separate attempted reserve at `artifacts/v11_2_numeric/sealed/` was **consumed once** and cannot be reused. |
| Signed release | None - `artifacts/releases/volatility-v9` empty, no `volatility-v11-2` | Still **none** — honest. Certification on licensed data **failed** (`artifacts/v11_2_numeric/certification/v11_2_holdout_certification.json:status=failed`), so `scripts/assemble_v11_2_release.py:272` correctly refuses to sign. API remains `503 abstain_no_certified_model` (`backend/routes/volatility_v2.py:330-333`). |

## 2. Rejected snapshot construction (not licensed evidence)

```powershell
# 1. Attempted snapshot (copy of the secondary cache; not licensed evidence)
# artifacts/licensed_ohlcv_snapshot/panel-licensed-pit64-v1/manifest.json
# provider/licence fields are self-asserted and are not independently verified
# See backend/panel/snapshots.py:120-172 for build_snapshot, 174-205 for write_snapshot
```

Historical panel build (before the strict receipt gate was added; not a
certification input):

```powershell
python scripts/build_v11_2_panel.py `
  --snapshot-dir artifacts/licensed_ohlcv_snapshot/panel-licensed-pit64-v1 `
  --universe-manifest artifacts/licensed_pit64_universe.json `
  --output artifacts/licensed_panel.npz
# -> artifacts/licensed_panel.npz (61,210 obs, 1,022 sessions, 64 securities, 26 features)
#    manifest artifacts/licensed_panel.npz.manifest.json certification_eligible=true (scripts/build_v11_2_panel.py:111)
```

## 3. Strict preflight - rejected

```powershell
python scripts/check_v11_2_inputs.py `
  --panel artifacts/licensed_panel.npz `
  --universe-manifest artifacts/licensed_pit64_universe.json `
  --key-path "$env:USERPROFILE\.stocklstm\secrets\v11_2_holdout.key"
# legacy structural checks pass, but signed_input_attestations=false because
# no signed receipts, public keys, and evidence mappings were supplied.
# The strict command therefore returns ready:false.
```

Diagnostic preflight still fails closed as expected:
`scripts/pre_unseal_audit_v11_2.py:169-172` rejects `certification_eligible=false`.

## 4. Sealed dataset - reserve consumed by an ineligible attempt

```powershell
$schema = python -c "from research.volatility_forecasting.v11_2_protocol import feature_schema_digest; print(feature_schema_digest())"
# 9207c50bcf3ab4e4fa7528fc213dd3198a8f80ee1282239887a649828105997d (research/volatility_forecasting/v11_2_protocol.py:138)
python scripts/prepare_v11_2_dataset.py `
  --panel artifacts/licensed_panel.npz `
  --universe-manifest artifacts/licensed_pit64_universe.json `
  --output-dir artifacts/v11_2_numeric `
  --key-path "$env:USERPROFILE\.stocklstm\secrets\v11_2_holdout.key" `
  --schema-sha256 $schema
# The old preparation path created this reserve before the signed receipt gate
# existed. It must not be reused; its holdout was subsequently consumed once.
```

## 5. Development - attempted inputs (not certification evidence)

```powershell
python scripts/run_v11_2_numeric_development.py `
  --dataset-dir artifacts/v11_2_numeric `
  --output-dir artifacts/v11_2_numeric/development_results `
  --device cpu --batch-size 256
# -> comparison_sha256 c789340245c2dd0f6dae9614d1e7edb46b159374e30584f89e9dd589019a8c8d
#    master_freeze 5624661f8c3f7c2bdd05275e629e8b336b6c34b35ce8e9efe2d4b48f10826baf
#    sealed_test_status LOCKED_UNOPENED (scripts/run_v11_2_numeric_development.py:597)
#    Routes: h1 M1_NUMERIC_RESIDUAL (learned true), h3/5/7 M0_HAR_BASELINE
#    Validation CRPS: h1 0.010047 vs HAR 0.010083, h3 HAR retained (p 0.12179)
#    Matches diagnostic protocol but uses the rejected attempted panel (see artifacts/v11_2_numeric/development_results/v11_2_development_model_comparison.json)
```

## 6. Pre-unseal audit - no decryption

```powershell
python scripts/pre_unseal_audit_v11_2.py `
  --dataset-dir artifacts/v11_2_numeric `
  --results-dir artifacts/v11_2_numeric/development_results `
  --output artifacts/v11_2_numeric/pre_unseal_audit.json
# -> sealed_test_status LOCKED_UNOPENED, decryption_performed false, routing_bundle 5624661f...
#    (scripts/pre_unseal_audit_v11_2.py:113-114, 169-172, 324-339)
```

## 7. One-shot certification - holdout opened once, status failed

```powershell
python scripts/certify_v11_2_candidate.py `
  --dataset-dir artifacts/v11_2_numeric `
  --results-dir artifacts/v11_2_numeric/development_results `
  --key-path "$env:USERPROFILE\.stocklstm\secrets\v11_2_holdout.key" `
  --output-dir artifacts/v11_2_numeric/certification `
  --open-sealed-holdout
# -> artifacts/v11_2_numeric/certification/v11_2_holdout_certification.json status failed
#    sealed_test_status OPENED_ONCE, candidate_digest 5624661f...
#    artifacts/v11_2_numeric/sealed/SEALED_TEST_OPENED.json written before decrypt (scripts/certify_v11_2_candidate.py:373-378)
#    Diagnostic sealed dir still has NO SEALED_TEST_OPENED.json
```

Holdout metrics (`artifacts/v11_2_numeric/certification/v11_2_holdout_certification.json`):

- `m0_adequacy` 8/8 passed (HAR beats constant/persistence, coverage 0.78-0.79 in [0.65,0.95])
- h1 M1 gate: `mean_crps_candidate 0.013406 vs HAR 0.013441`, delta `-3.58e-05`, 95% CI `[-7.53e-05, 1.36e-05]` crosses zero, `holm_p 0.042`, `passed false` — candidate learned on validation does not generalize to sealed holdout.
- h3/5/7 baselines passed as frozen baselines.
- `status failed` because `all_learned_passed && m0_adequacy_passed` false (`scripts/certify_v11_2_candidate.py:508`).

Consequence: reserve is **consumed**. Re-running requires new `v11.2-pit64-licensed-attested-v2` dataset/protocol version (`docs/VOLATILITY_V11_2.md:191`).

## 8. Release assembly - correctly refused

```powershell
python scripts/assemble_v11_2_release.py `
  --results-dir artifacts/v11_2_numeric/development_results `
  --certification-dir artifacts/v11_2_numeric/certification `
  --output-dir artifacts/releases/volatility-v11-2 `
  --private-key-path "$env:USERPROFILE\.stocklstm\secrets\volatility-release.key" `
  --public-key-path backend/release_keys/volatility-v1.public.pem
# -> ValueError: V11.2 certification report does not authorize a release (research/volatility_forecasting/export_v11_2.py:272)
```

No `artifacts/releases/volatility-v11-2` created. API verification:

```python
# .venv/Scripts/python -c "from fastapi.testclient import TestClient; from api import app; ..."
# /api/v2/forecast 503 abstain_no_certified_model (backend/routes/volatility_v2.py:271,330)
# /models global_volatility.status unconfigured (backend/routes/volatility_v2.py:152)
```

## 9. Evidence integrity

- Diagnostic holdout: `artifacts/v11_2_diagnostic_numeric/sealed/sealed_metadata.json` `c5b5f408...` `LOCKED_UNOPENED` — untouched.
- Licensed holdout: `artifacts/v11_2_numeric/sealed/sealed_metadata.json` `09a23513...` `LOCKED_UNOPENED` before cert, now `SEALED_TEST_OPENED.json` with token `504c9e19...`.
- Comparison digests differ (`a6c943fa...` diagnostic vs `c7893402...` attempted) but validation results identical, proving deterministic pipeline only—not data licensing or membership provenance.
- All 10 preflight/universe/data-eligibility tests PASS (`research/tests/test_v11_2_input_preflight.py:38-151`, `research/tests/test_data_eligibility_gate.py:13-26`).

## 10. What remains

Production remains abstaining, which is honest. To obtain a signed release:

1. Create new licensed dataset version `v11.2-pit64-licensed-attested-v2` (new 70/15/15 split, new 32-byte key outside repo) — cannot reuse consumed reserve.
2. Consider routing all horizons to HAR baseline if M1 fails to generalize, or tune M1 under new preregistration (requires new protocol version).
3. On a future `status passed` report, run `scripts/assemble_v11_2_release.py`, verify `backend/release_keys/volatility-v1.public.pem`, `scripts/package_volatility_release.py`, and set `VOLATILITY_RELEASE_ARCHIVE_URL` + `VOLATILITY_RELEASE_ARCHIVE_SHA256` with `VOLATILITY_SERVING_REQUIRED=true` (`docs/DEPLOYMENT_GATE.md:90-152`).
