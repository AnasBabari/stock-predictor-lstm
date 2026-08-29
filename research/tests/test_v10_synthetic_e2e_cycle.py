"""End-to-end synthetic CLI subprocess integration test for StockLSTM Volatility V10.

Exercises the real operational research runner via CLI subprocess commands:
prepare -> train -> select -> freeze -> certify -> export -> sign -> verify
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from research.volatility_forecasting.export_v10 import verify_release_bundle_integrity
from research.volatility_forecasting.signing_v10 import sign_release_manifest_detached


def test_v10_full_cli_lifecycle_end_to_end(tmp_path: Path) -> None:
    # 1. Setup synthetic environment in tmp_path
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)

    # Copy configs and scripts to tmp repo
    configs_dir = repo_root / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    real_protocol = Path(
        r"c:\Users\Babar\stock-predictor-lstm\configs\volatility_v10_protocol.json"
    )
    (configs_dir / "volatility_v10_protocol.json").write_text(
        real_protocol.read_text(encoding="utf-8"), encoding="utf-8"
    )

    # Create dummy backend/pyproject.toml
    backend_dir = repo_root / "backend"
    backend_dir.mkdir(parents=True, exist_ok=True)
    (backend_dir / "pyproject.toml").write_text("[project]\nname='stock-lstm'\n", encoding="utf-8")

    # Generate synthetic market panel CSV
    dates = pd.date_range("2022-01-01", periods=120, freq="B").strftime("%Y-%m-%d").tolist()
    rows = []
    rng = np.random.default_rng(42)
    for d in dates:
        for sec in ["SEC_AAPL_001", "SEC_AMZN_001"]:
            row = {"Date": d, "SecurityID": sec, "target_h1": 0.0004, "target_h5": 0.0020}
            for feat_idx in range(26):
                row[f"feat_{feat_idx}"] = float(rng.normal())
            rows.append(row)

    df_panel = pd.DataFrame(rows)
    panel_csv = tmp_path / "synthetic_panel.csv"
    df_panel.to_csv(panel_csv, index=False)

    # Generate universe manifest
    universe_json = tmp_path / "synthetic_universe.json"
    universe_json.write_text(
        json.dumps({"universe_name": "synthetic_ndx100"}, indent=2), encoding="utf-8"
    )

    # Generate split manifest
    split_json = tmp_path / "synthetic_splits.json"
    split_json.write_text(
        json.dumps(
            {
                "train_sessions": dates[:70],
                "val_sessions": dates[70:95],
                "test_sessions": dates[95:],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Generate feature schema
    schema_json = tmp_path / "synthetic_schema.json"
    schema_json.write_text(
        json.dumps({"feature_names": [f"feat_{i}" for i in range(26)]}, indent=2),
        encoding="utf-8",
    )

    script_path = Path(r"c:\Users\Babar\stock-predictor-lstm\scripts\run_v10_pipeline.py")
    run_id = "test_e2e_cycle_run"

    # 2. RUN SUBPROCESS: PREPARE
    env = dict(os.environ)
    env["PYTHONPATH"] = "research;backend;."
    cmd_prep = [
        sys.executable,
        str(script_path),
        "--repo-root",
        str(repo_root),
        "prepare",
        "--run-id",
        run_id,
        "--protocol",
        str(configs_dir / "volatility_v10_protocol.json"),
        "--universe-manifest",
        str(universe_json),
        "--market-snapshot",
        str(panel_csv),
        "--split-manifest",
        str(split_json),
        "--feature-schema",
        str(schema_json),
    ]
    res_prep = subprocess.run(cmd_prep, capture_output=True, text=True, env=env)
    assert res_prep.returncode == 0, f"prepare failed: {res_prep.stderr}"

    run_dir = repo_root / "research" / "results" / "v10-development" / run_id
    assert (run_dir / "run_manifest.json").exists()

    # 3. RUN SUBPROCESS: TRAIN
    cmd_train = [
        sys.executable,
        str(script_path),
        "--repo-root",
        str(repo_root),
        "train",
        "--run-id",
        run_id,
        "--horizons",
        "1,5",
        "--families",
        "har,ridge,tcn",
        "--max-epochs",
        "2",
    ]
    res_train = subprocess.run(cmd_train, capture_output=True, text=True, env=env)
    assert res_train.returncode == 0, f"train failed: {res_train.stderr}"
    assert (run_dir / "development_ledger.jsonl").exists()

    # 4. RUN SUBPROCESS: SELECT
    cmd_sel = [
        sys.executable,
        str(script_path),
        "--repo-root",
        str(repo_root),
        "select",
        "--run-id",
        run_id,
        "--horizons",
        "1,5",
    ]
    res_sel = subprocess.run(cmd_sel, capture_output=True, text=True, env=env)
    assert res_sel.returncode == 0, f"select failed: {res_sel.stderr}"
    assert (run_dir / "selected_champions.json").exists()

    # 5. RUN SUBPROCESS: FREEZE
    cmd_frz = [
        sys.executable,
        str(script_path),
        "--repo-root",
        str(repo_root),
        "freeze",
        "--run-id",
        run_id,
    ]
    res_frz = subprocess.run(cmd_frz, capture_output=True, text=True, env=env)
    assert res_frz.returncode == 0, f"freeze failed: {res_frz.stderr}"
    assert (run_dir / "frozen_package" / "candidate_manifest.json").exists()

    # 6. RUN SUBPROCESS: CERTIFY
    cmd_cert = [
        sys.executable,
        str(script_path),
        "--repo-root",
        str(repo_root),
        "certify",
        "--run-id",
        run_id,
        "--universe-eligible",
        "--market-panel-eligible",
    ]
    res_cert = subprocess.run(cmd_cert, capture_output=True, text=True, env=env)
    assert res_cert.returncode == 0, f"certify failed: {res_cert.stderr}"

    cert_dir = repo_root / "research" / "results" / "v10-certification" / run_id
    assert (cert_dir / "certification_record_v10.json").exists()
    assert (cert_dir / "test_opening_receipt.json").exists()

    # 7. RUN SUBPROCESS: EXPORT
    cmd_exp = [
        sys.executable,
        str(script_path),
        "--repo-root",
        str(repo_root),
        "export",
        "--run-id",
        run_id,
    ]
    res_exp = subprocess.run(cmd_exp, capture_output=True, text=True, env=env)
    assert res_exp.returncode == 0, f"export failed: {res_exp.stderr}"

    # 8. RUN SUBPROCESS: REPORT
    cmd_rep = [
        sys.executable,
        str(script_path),
        "--repo-root",
        str(repo_root),
        "report",
        "--run-id",
        run_id,
    ]
    res_rep = subprocess.run(cmd_rep, capture_output=True, text=True, env=env)
    assert res_rep.returncode == 0, f"report failed: {res_rep.stderr}"
    assert (run_dir / "V10_RUN_REPORT.md").exists()

    # 9. SIGN AND VERIFY RELEASE BUNDLE
    bundle_dir = repo_root / "backend" / "releases" / f"release-v10-{run_id}"
    assert bundle_dir.exists()
    assert (bundle_dir / "manifest.json").exists()

    # External detached signing
    priv_key = Ed25519PrivateKey.generate()
    pub_pem = priv_key.public_key().public_bytes(
        encoding=Encoding.PEM, format=PublicFormat.SubjectPublicKeyInfo
    )

    manifest_bytes = (bundle_dir / "manifest.json").read_bytes()
    sig_bytes = sign_release_manifest_detached(manifest_bytes, priv_key)
    (bundle_dir / "signature.ed25519").write_bytes(sig_bytes)

    # Secure verification
    assert verify_release_bundle_integrity(bundle_dir, pub_pem) is True
