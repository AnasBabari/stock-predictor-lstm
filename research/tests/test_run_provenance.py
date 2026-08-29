"""Tests for cryptographic run manifests and provenance validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.volatility_forecasting.provenance import (
    ImmutableRunManifest,
    ProvenanceMismatchError,
    compute_canonical_json_sha256,
    compute_ledger_evidence_sha256,
)


@pytest.fixture
def sample_manifest() -> ImmutableRunManifest:
    return ImmutableRunManifest(
        run_id="run-20260829-test",
        artifact_role="development_diagnostic",
        git_sha="4b2018fb7ec3fa9c31cd41691459e5d9696a841f",
        protocol_id="volatility-v10",
        protocol_sha256="d205b6394cc39e0e63e6d5c5bf1f6d4a8ca20ceea8a4917f3963ed44f78523b1",
        universe_snapshot_id="universe-ndx100-v1",
        universe_sha256="a" * 64,
        panel_snapshot_id="panel-ndx100-v1",
        panel_sha256="b" * 64,
        split_manifest_sha256="c" * 64,
        feature_schema_sha256="d" * 64,
        news_snapshot_sha256=None,
        dependency_lock_sha256="e" * 64,
        candidate_registry_sha256="f" * 64,
        hardware={"platform": "Windows-10", "device": "cuda"},
        created_at="2026-08-29T21:00:00Z",
    )


def test_manifest_canonical_serialization_and_hash(sample_manifest: ImmutableRunManifest) -> None:
    digest = sample_manifest.manifest_sha256()
    assert isinstance(digest, str) and len(digest) == 64
    reloaded = ImmutableRunManifest.from_dict(sample_manifest.to_dict())
    assert reloaded.manifest_sha256() == digest


def test_manifest_file_roundtrip(sample_manifest: ImmutableRunManifest, tmp_path: Path) -> None:
    path = tmp_path / "run_manifest.json"
    sample_manifest.save(path)
    loaded = ImmutableRunManifest.from_file(path)
    assert loaded == sample_manifest
    loaded.verify_matching(sample_manifest)


def test_provenance_mismatch_raises_on_divergent_protocol_hash(sample_manifest: ImmutableRunManifest) -> None:
    tampered_dict = sample_manifest.to_dict()
    tampered_dict["protocol_sha256"] = "0" * 64
    with pytest.raises(ProvenanceMismatchError, match="protocol_sha256"):
        sample_manifest.verify_matching(tampered_dict)


def test_provenance_mismatch_raises_on_divergent_panel_hash(sample_manifest: ImmutableRunManifest) -> None:
    tampered_dict = sample_manifest.to_dict()
    tampered_dict["panel_sha256"] = "1" * 64
    with pytest.raises(ProvenanceMismatchError, match="panel_sha256"):
        sample_manifest.verify_matching(tampered_dict)


def test_compute_ledger_evidence_sha256_is_deterministic() -> None:
    records_1 = [
        {"fold": 1, "family": "tcn", "seed": 41, "horizon": 1, "loss": 0.5},
        {"fold": 1, "family": "har", "seed": 0, "horizon": 1, "loss": 0.6},
    ]
    records_2 = [
        {"fold": 1, "family": "har", "seed": 0, "horizon": 1, "loss": 0.6},
        {"fold": 1, "family": "tcn", "seed": 41, "horizon": 1, "loss": 0.5},
    ]
    # Reordered input must produce identical digest
    assert compute_ledger_evidence_sha256(records_1) == compute_ledger_evidence_sha256(records_2)
