"""Tests for V10 candidate package freeze and pre-certification audit."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from research.volatility_forecasting.candidate_freeze_v10 import (
    FreezeIntegrityError,
    FrozenCandidatePackageV10,
    FrozenHorizonCandidate,
)


@pytest.fixture
def sample_package(tmp_path: Path) -> FrozenCandidatePackageV10:
    weight_file = tmp_path / "tcn_h1.bin"
    weight_file.write_bytes(b"dummy_weights_h1")
    w_sha = hashlib.sha256(b"dummy_weights_h1").hexdigest()

    h1 = FrozenHorizonCandidate(
        horizon=1,
        family="tcn",
        role="learned_candidate",
        config={"channels": 64, "kernel_size": 3},
        selected_seed=41,
        scaler_parameters={"mean": 0.0, "std": 1.0},
        baseline_parameters=None,
        weights_relative_path="tcn_h1.bin",
        weights_sha256=w_sha,
    )
    h3 = FrozenHorizonCandidate(
        horizon=3,
        family="har",
        role="certified_baseline",
        config={},
        selected_seed=0,
        scaler_parameters={},
        baseline_parameters={"beta": [1.0, 0.5, 0.3, 0.2]},
        weights_relative_path=None,
        weights_sha256=None,
    )
    return FrozenCandidatePackageV10(
        package_id="cand-v10-test-01",
        protocol_id="volatility-v10",
        protocol_sha256="0" * 64,
        git_sha="1" * 40,
        feature_schema_sha256="2" * 64,
        panel_snapshot_sha256="3" * 64,
        development_ledger_sha256="4" * 64,
        created_at_utc="2026-08-29T21:00:00Z",
        horizons=(h1, h3),
    )


def test_package_roundtrip_and_weights_verification(
    sample_package: FrozenCandidatePackageV10, tmp_path: Path
) -> None:
    pkg_dir = sample_package.save_package(tmp_path)
    # Copy dummy weight file into saved directory
    (pkg_dir / "tcn_h1.bin").write_bytes(b"dummy_weights_h1")

    reloaded = FrozenCandidatePackageV10.from_package_dir(pkg_dir)
    assert reloaded.package_id == sample_package.package_id
    assert len(reloaded.horizons) == 2
    reloaded.verify_weights_integrity(pkg_dir)


def test_tampered_weights_fail_audit(
    sample_package: FrozenCandidatePackageV10, tmp_path: Path
) -> None:
    pkg_dir = sample_package.save_package(tmp_path)
    # Write tampered weight file
    (pkg_dir / "tcn_h1.bin").write_bytes(b"tampered_weights")

    reloaded = FrozenCandidatePackageV10.from_package_dir(pkg_dir)
    with pytest.raises(FreezeIntegrityError, match="Weight checksum mismatch"):
        reloaded.verify_weights_integrity(pkg_dir)
