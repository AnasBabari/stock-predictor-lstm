"""End-to-end integration test of the complete V10 synthetic forecasting cycle.

Verifies end-to-end plumbing across:
  prepare -> train -> select -> freeze -> certify -> export -> detached signing -> backend verification
Labeled strictly as development_diagnostic_synthetic_only.
"""

from __future__ import annotations

from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from research.volatility_forecasting.candidate_freeze_v10 import (
    FrozenCandidatePackageV10,
    FrozenHorizonCandidate,
)
from research.volatility_forecasting.certification_v10 import (
    CertificationReportV10,
    HorizonCertificationDecision,
    SealedTestOpeningRecordV10,
)
from research.volatility_forecasting.export_v10 import (
    assemble_release_bundle,
    verify_release_bundle_integrity,
)
from research.volatility_forecasting.gpu_harness_v10 import (
    TrainingExecutionConfig,
    train_candidate_fold,
)
from research.volatility_forecasting.horizon_selection_v10 import select_champions_by_horizon
from research.volatility_forecasting.provenance import (
    ImmutableRunManifest,
    capture_runtime_hardware_info,
)
from research.volatility_forecasting.signing_v10 import sign_release_manifest_detached


def test_v10_synthetic_end_to_end_cycle(tmp_path: Path) -> None:
    # 1. PREPARE: Create run directory and manifest
    run_dir = tmp_path / "run-synthetic-v10"
    run_dir.mkdir(parents=True, exist_ok=True)

    dummy_file = run_dir / "protocol.json"
    dummy_file.write_text("{}", encoding="utf-8")

    manifest = ImmutableRunManifest(
        run_id="run-synthetic-v10",
        artifact_role="development_diagnostic_synthetic_only",
        git_sha="0" * 40,
        protocol_id="volatility-v10",
        protocol_sha256="0" * 64,
        universe_snapshot_id="universe-synth-v1",
        universe_sha256="1" * 64,
        panel_snapshot_id="panel-synth-v1",
        panel_sha256="2" * 64,
        split_manifest_sha256="3" * 64,
        feature_schema_sha256="4" * 64,
        news_snapshot_sha256=None,
        dependency_lock_sha256="5" * 64,
        candidate_registry_sha256="6" * 64,
        hardware=capture_runtime_hardware_info("cpu"),
        created_at="2026-08-29T22:00:00Z",
    )
    manifest.save(run_dir / "run_manifest.json")
    assert (run_dir / "run_manifest.json").exists()

    # 2. TRAIN: Execute candidate fold training on synthetic data
    import numpy as np

    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(50, 26))
    y_train = np.maximum(0.0004 + 0.0001 * X_train[:, 0], 1e-6)
    X_val = rng.normal(size=(20, 26))
    y_val = np.maximum(0.0004 + 0.0001 * X_val[:, 0], 1e-6)

    cfg = TrainingExecutionConfig(candidate_family="ridge", horizon=1, fold_idx=0)
    train_res = train_candidate_fold(cfg, X_train, y_train, X_val, y_val)
    assert train_res["status"] == "success"

    ledger = [
        {"horizon": 1, "family": "ridge", "relative_qlike": 0.95, "ratio_upper_95": 0.98},
        {"horizon": 3, "family": "har", "relative_qlike": 1.00, "ratio_upper_95": 1.00},
    ]

    # 3. SELECT: Independent horizon selection
    champions = select_champions_by_horizon(ledger, horizons=[1, 3])
    assert champions[1]["champion_family"] == "ridge"
    assert champions[1]["role"] == "learned_candidate"
    assert champions[3]["champion_family"] == "har"
    assert champions[3]["role"] == "development_baseline_candidate"

    # 4. FREEZE: Atomically freeze package
    h1 = FrozenHorizonCandidate(
        horizon=1,
        family="ridge",
        role="learned_candidate",
        config={},
        selected_seed=41,
        scaler_parameters={"mean": 0.0, "std": 1.0},
        baseline_parameters=None,
        weights_relative_path="ridge_h1.bin",
        weights_sha256="7" * 64,
    )
    package = FrozenCandidatePackageV10(
        package_id="cand-synthetic-01",
        protocol_id="volatility-v10",
        protocol_sha256="0" * 64,
        git_sha="0" * 40,
        feature_schema_sha256="4" * 64,
        panel_snapshot_sha256="2" * 64,
        development_ledger_sha256="8" * 64,
        created_at_utc="2026-08-29T22:00:00Z",
        horizons=(h1,),
    )
    pkg_dir = package.save_package_atomic(
        run_dir, weights_map={"ridge_h1.bin": b"dummy_ridge_weights"}
    )
    assert (pkg_dir / "candidate_manifest.json").exists()

    # 5. CERTIFY: Evaluate one-shot sealed record
    opening = SealedTestOpeningRecordV10(
        test_opened_at_utc="2026-08-29T22:00:00Z",
        candidate_package_sha256="6" * 64,
        protocol_sha256="0" * 64,
        split_manifest_sha256="3" * 64,
        operator="ci_runner",
        attempt=1,
    )
    opening.save_atomic(run_dir)

    d1 = HorizonCertificationDecision(
        horizon=1,
        family="ridge",
        outcome="certified_learned_model",
        relative_qlike=0.95,
        ratio_upper_95=0.98,
        dm_p_value=0.02,
        holm_adjusted_p_value=0.02,
        transfer_relative_qlike=0.96,
        passed_all_gates=True,
        reason="Cleared all synthetic gates",
    )
    cert_report = CertificationReportV10(
        report_id="cert-synthetic-001",
        protocol_id="volatility-v10",
        protocol_sha256="0" * 64,
        candidate_package_sha256="6" * 64,
        data_snapshot_sha256="2" * 64,
        evaluated_at_utc="2026-08-29T22:00:00Z",
        decisions=(d1,),
        certified_horizons=(1,),
    )
    cert_report.save(run_dir)

    # 6. EXPORT & DETACHED SIGNING
    priv_key = Ed25519PrivateKey.generate()
    pub_pem = priv_key.public_key().public_bytes(
        encoding=Encoding.PEM,
        format=PublicFormat.SubjectPublicKeyInfo,
    )

    bundle_dir = assemble_release_bundle(
        output_dir=run_dir,
        bundle_id="release-synthetic-v10",
        certification_report=cert_report,
        protocol_version="volatility-v10",
        model_family_by_horizon={1: "ridge"},
        feature_schema_sha256="4" * 64,
        universe_sha256="1" * 64,
        files_to_include={"models/h1_ridge.bin": b"dummy_ridge_weights"},
    )
    manifest_bytes = (bundle_dir / "manifest.json").read_bytes()
    sig = sign_release_manifest_detached(manifest_bytes, priv_key)
    (bundle_dir / "signature.ed25519").write_bytes(sig)

    # 7. VERIFY: Verify release bundle integrity with public key
    assert verify_release_bundle_integrity(bundle_dir, pub_pem) is True
