"""Comprehensive tests for the global offline training and certification pipeline CLI."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from scripts.run_global_pipeline import (  # noqa: E402
    PipelineConfig,
    generate_synthetic_universe,
    run_pipeline,
)


@pytest.fixture()
def key_pair(tmp_path: Path) -> tuple[Path, Path]:
    private_key = Ed25519PrivateKey.generate()
    private_pem = tmp_path / "signing.pem"
    private_pem.write_bytes(
        private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    public_pem = tmp_path / "verify.pem"
    public_pem.write_bytes(
        private_key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    return private_pem, public_pem


def test_development_mode_fails_closed_without_panel_directory(tmp_path: Path) -> None:
    cfg = PipelineConfig(
        run_id="test_fail_closed",
        mode="development",
        panel_dir=None,  # Missing panel directory
    )
    run_dir = tmp_path / "fail_closed_run"
    with pytest.raises(ValueError, match="requires an explicit --panel-dir"):
        run_pipeline(config=cfg, run_dir=run_dir)


def test_development_mode_fails_closed_when_panel_directory_does_not_exist(tmp_path: Path) -> None:
    cfg = PipelineConfig(
        run_id="test_missing_dir",
        mode="development",
        panel_dir=str(tmp_path / "non_existent_panel_dir"),
    )
    run_dir = tmp_path / "missing_dir_run"
    with pytest.raises(FileNotFoundError, match="Panel directory does not exist"):
        run_pipeline(config=cfg, run_dir=run_dir)


def test_fixture_mode_allows_synthetic_universe(tmp_path: Path) -> None:
    cfg = PipelineConfig(
        run_id="test_fixture_run",
        mode="fixture",
        horizons=[1],
        folds=2,
        embargo=2,
        min_train_sessions=200,
        temporal_holdout_sessions=50,
        candidate_families=["persistence", "ridge_global"],
        seeds=[42],
        resamples=50,
    )
    run_dir = tmp_path / "fixture_run"
    results = run_pipeline(
        config=cfg,
        run_dir=run_dir,
        stage="snapshot",
    )
    assert results["mode"] == "fixture"
    assert "snapshot" in results["stages"]
    assert (run_dir / "stages" / "01_snapshot.json").exists()


def test_pipeline_executes_explicit_stages_and_resumes(
    tmp_path: Path, key_pair: tuple[Path, Path]
) -> None:
    private_pem, public_pem = key_pair
    tickers = ["MIC1", "MIC2", "MIC3", "MIC4"]
    universe = generate_synthetic_universe(tickers, n_sessions=450, seed=42)

    cfg = PipelineConfig(
        run_id="test_resumable_run",
        mode="fixture",
        horizons=[1, 5],
        folds=2,
        embargo=2,
        min_train_sessions=150,
        temporal_holdout_sessions=50,
        candidate_families=["persistence", "ridge_global"],
        seeds=[42, 43],
        resamples=50,
        private_key_path=str(private_pem),
        public_key_path=str(public_pem),
    )
    run_dir = tmp_path / "stages_run"

    # 1. Execute stage snapshot
    res_snap = run_pipeline(config=cfg, run_dir=run_dir, stage="snapshot", universe_data=universe)
    assert "snapshot" in res_snap["stages"]
    assert (run_dir / "stages" / "01_snapshot.json").exists()

    # 2. Execute stage features
    res_feat = run_pipeline(config=cfg, run_dir=run_dir, stage="features", universe_data=universe)
    assert "features" in res_feat["stages"]
    assert (run_dir / "stages" / "02_features.json").exists()

    # 3. Execute stage folds
    res_folds = run_pipeline(config=cfg, run_dir=run_dir, stage="folds", universe_data=universe)
    assert "folds" in res_folds["stages"]
    assert (run_dir / "stages" / "03_folds.json").exists()

    # 4. Execute stage evaluate
    res_eval = run_pipeline(config=cfg, run_dir=run_dir, stage="evaluate", universe_data=universe)
    assert "evaluate" in res_eval["stages"]
    assert (run_dir / "stages" / "05_evaluate.json").exists()

    # 5. Execute stage select
    res_sel = run_pipeline(config=cfg, run_dir=run_dir, stage="select", universe_data=universe)
    assert "selection" in res_sel["stages"]
    assert (run_dir / "stages" / "06_selection.json").exists()

    # 6. Execute stage certify without explicit holdout flag (must remain locked)
    res_cert_locked = run_pipeline(
        config=cfg,
        run_dir=run_dir,
        stage="certify",
        open_locked_certification_holdout=False,
        universe_data=universe,
    )
    assert res_cert_locked["stages"]["certification"]["status"] == "locked_untouched"

    # 7. Execute stage certify with explicit flag (must open and evaluate holdout)
    res_cert_open = run_pipeline(
        config=cfg,
        run_dir=run_dir,
        stage="certify",
        open_locked_certification_holdout=True,
        universe_data=universe,
    )
    assert res_cert_open["stages"]["certification"]["status"] == "holdout_opened"
    assert "decisions" in res_cert_open["stages"]["certification"]

    # 8. Execute stage release (refit and release bundle assembly)
    res_all = run_pipeline(
        config=cfg,
        run_dir=run_dir,
        stage="release",
        open_locked_certification_holdout=True,
        universe_data=universe,
    )
    assert "release" in res_all["stages"]
    assert (run_dir / "refit" / "model_h1.json").exists()
    assert (run_dir / "release" / "manifest.json").exists()
    assert (run_dir / "release" / "manifest.sig").exists()


def test_load_panel_from_directory(tmp_path: Path) -> None:
    from panel.snapshots import load_panel_from_directory, write_snapshot

    tickers = ["T1", "T2"]
    universe = generate_synthetic_universe(tickers, n_sessions=350, seed=123)
    snap_dir = write_snapshot(tmp_path / "snapshots", universe, license_acknowledged=True)

    loaded = load_panel_from_directory(snap_dir)
    assert set(loaded.keys()) == {"T1", "T2"}
    assert len(loaded["T1"]) == 350


def test_cli_main_invokes_cleanly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.run_global_pipeline import main

    run_dir = tmp_path / "cli_run"
    test_args = [
        "run_global_pipeline.py",
        "--mode",
        "fixture",
        "--stage",
        "snapshot",
        "--run-dir",
        str(run_dir),
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    rc = main()
    assert rc == 0
    assert (run_dir / "pipeline_manifest.json").exists()
