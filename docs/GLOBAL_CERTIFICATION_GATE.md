# Global Model Certification Gate

This gate defines the immutable certification contract for offline global multi-asset models. Reported performance metrics and release artifacts must never outlive or misrepresent the methodology that produced them.

---

## 1. Governance & Certification Contract

A candidate architecture is eligible for release **only when all of the following gates pass simultaneously**:

1. **Stationary Feature Parity**: Features conform strictly to Deployable Schema v5 with zero lookahead, verified identical across Python and client runtimes.
2. **Purged Calendar Split**: Folds enforce expanding calendar training with an $(h-1)$ purge gap and an embargo $E \ge 5$ sessions.
3. **Independent Multi-Seed Admissibility**: Neural candidates must be trained across at least 3 independent seeds (`seeds=[42, 43, 44]`) with seed dispersion $\sigma_{\text{seed}} \le 0.05$.
4. **Statistical Edge over Baselines**:
   - $\text{Relative RMSE} < 1.0$ (beats naive persistence on out-of-fold validation).
   - Bootstrap ratio upper bound $R_{0.95} < 1.0$.
   - Diebold-Mariano HAC test with family-wise Holm adjustment $p_{\text{adj}} < 0.05$.
5. **Locked Holdout Certification**:
   - The holdout is opened **exactly once** via `--open-locked-certification-holdout`.
   - The temporal holdout (last 252 sessions) and asset-transfer holdout must achieve $\text{Relative RMSE} < 1.00$.
6. **Provenance & Release Integrity**:
   - Refit models are signed with detached Ed25519 signatures.
   - Verification confirms SHA256 checksums, manifest structure, and cryptographic signature.

---

## 2. Frozen Global Certification Record

```yaml
schema_version: 1
global_certification_status: "development_verified"
panel_snapshot_id: "panel-fixture-v1"
config_digest: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
development_fold_digest: "purged-calendar-v5"
candidate_evidence_digest: "evidence-v1-verified"
certification_digest: "locked-holdout-verified"
release_signature_algorithm: "ed25519"
runtime_environment:
  gpu: "NVIDIA GeForce RTX 2060"
  vram_mb: 6144
  platform: "WSL2 Ubuntu 24.04 LTS"
  cuda_version: "12.x"
  tensorflow_version: "2.16+"
```

---

## 3. Full Check Battery for Global Release

Before committing a certified release or updating this gate, the entire suite must be clean:

```bash
# 1. Pipeline CLI & Stage Unit Suite
pytest backend/tests/test_global_pipeline_cli.py -v
pytest backend/tests/test_pipeline_certification.py -v
pytest backend/tests/test_release_bundle.py -v

# 2. End-to-End Pipeline Execution
python scripts/run_global_pipeline.py --mode fixture --stage all --open-locked-certification-holdout
```
