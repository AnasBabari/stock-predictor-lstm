"""Tests for deterministic external V11.2 attestation request packaging."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.volatility_forecasting.v11_2_attestation import AttestationError
from research.volatility_forecasting.v11_2_protocol import (
    V11_2_PROTOCOL_ID,
    canonical_json_digest,
)
from scripts.create_v11_2_attestation_requests import create_requests


def _fixture_inputs(root: Path) -> tuple[Path, Path, dict[str, Path], dict[str, Path]]:
    snapshot = root / "snapshot.json"
    snapshot.write_text(
        json.dumps(
            {
                "panel_id": "panel-v11-2-fixture",
                "pooled_checksum": "sha256:" + "a" * 64,
                "license": {"acknowledged": False},
            }
        ),
        encoding="utf-8",
    )
    securities = [{"security_id": f"SEC-{index:03d}"} for index in range(64)]
    universe = root / "universe.json"
    universe.write_text(
        json.dumps(
            {
                "protocol_id": V11_2_PROTOCOL_ID,
                "universe_version": "v11.2-pit64-fixture",
                "certification_eligible": True,
                "securities": securities,
            }
        ),
        encoding="utf-8",
    )
    license_document = root / "license.pdf"
    license_document.write_bytes(b"fixture external contract bytes")
    membership_master = root / "membership.json"
    membership_master.write_text(json.dumps(securities), encoding="utf-8")
    return (
        snapshot,
        universe,
        {"snapshot_manifest": snapshot, "license_document": license_document},
        {"membership_master": membership_master},
    )


def test_request_pack_binds_exact_subjects_and_rights(tmp_path: Path) -> None:
    snapshot, universe, market_evidence, pit_evidence = _fixture_inputs(tmp_path)
    output = tmp_path / "requests"
    manifest = create_requests(
        snapshot_manifest_path=snapshot,
        universe_manifest_path=universe,
        market_evidence_files=market_evidence,
        pit64_evidence_files=pit_evidence,
        output_dir=output,
    )

    market = json.loads((output / "market_data_attestation_request.json").read_text())
    pit64 = json.loads((output / "pit64_attestation_request.json").read_text())
    assert manifest["status"] == "unsigned_external_signatures_required"
    assert manifest["private_keys_created"] is False
    assert manifest["holdout_accessed"] is False
    assert market["subject"]["id"] == "panel-v11-2-fixture"
    assert market["subject"]["content_digest"] == "sha256:" + "a" * 64
    assert "derived_model_distribution" in market["required_rights"]
    assert pit64["subject"]["id"] == "v11.2-pit64-fixture"
    assert len(pit64["subject"]["content_digest"]) == len("sha256:") + 64
    for request in (market, pit64):
        request_digest = request.pop("request_sha256")
        assert canonical_json_digest(request) == request_digest
        assert "signature" not in request


def test_request_pack_refuses_overwrite(tmp_path: Path) -> None:
    snapshot, universe, market_evidence, pit_evidence = _fixture_inputs(tmp_path)
    output = tmp_path / "requests"
    output.mkdir()
    with pytest.raises(AttestationError, match="overwrite"):
        create_requests(
            snapshot_manifest_path=snapshot,
            universe_manifest_path=universe,
            market_evidence_files=market_evidence,
            pit64_evidence_files=pit_evidence,
            output_dir=output,
        )


def test_request_pack_rejects_different_snapshot_evidence(tmp_path: Path) -> None:
    snapshot, universe, market_evidence, pit_evidence = _fixture_inputs(tmp_path)
    other = tmp_path / "other-snapshot.json"
    other.write_text(snapshot.read_text(encoding="utf-8"), encoding="utf-8")
    market_evidence["snapshot_manifest"] = other
    with pytest.raises(AttestationError, match="exact snapshot manifest"):
        create_requests(
            snapshot_manifest_path=snapshot,
            universe_manifest_path=universe,
            market_evidence_files=market_evidence,
            pit64_evidence_files=pit_evidence,
            output_dir=tmp_path / "requests",
        )
