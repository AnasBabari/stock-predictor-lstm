"""Tests for the signed V11.2 input-provenance boundary."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from volatility_forecasting.v11_2_attestation import (
    MARKET_DATA_ATTESTATION,
    MARKET_REQUIRED_RIGHTS,
    PIT64_ATTESTATION,
    PIT64_REQUIRED_RIGHTS,
    AttestationError,
    public_key_fingerprint,
    security_master_digest,
    verify_dataset_attestation_record,
    verify_receipt,
)
from volatility_forecasting.v11_2_protocol import canonical_json_digest


def _write_keypair(root: Path) -> tuple[Path, str, Ed25519PrivateKey]:
    private = Ed25519PrivateKey.generate()
    root.mkdir(parents=True, exist_ok=True)
    public_path = root / "public.pem"
    public_path.write_bytes(
        private.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    return public_path, public_key_fingerprint(public_path), private


def _write_receipt(
    path: Path,
    private: Ed25519PrivateKey,
    *,
    attestation_type: str,
    subject_kind: str,
    subject_id: str,
    subject_digest: str,
    evidence: dict[str, Path],
    rights: frozenset[str],
    key_id: str,
    issuer: str = "fixture attester",
) -> None:
    payload = {
        "schema_version": 1,
        "attestation_type": attestation_type,
        "signature_algorithm": "ed25519",
        "subject": {
            "kind": subject_kind,
            "id": subject_id,
            "content_digest": f"sha256:{subject_digest}",
        },
        "issuer": {"name": issuer, "key_id": key_id},
        "issued_at": "2026-08-30T00:00:00Z",
        "rights": {name: True for name in sorted(rights)},
        "independent_review": {
            "independent": True,
            "reviewer": "fixture independent reviewer",
            "method": "fixture byte and identity audit",
        },
        "evidence_files": {
            name: f"sha256:{hashlib.sha256(file.read_bytes()).hexdigest()}"
            for name, file in sorted(evidence.items())
        },
    }
    unsigned = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    payload["signature"] = base64.b64encode(private.sign(unsigned.encode("utf-8"))).decode("ascii")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_signed_receipt_verifies_exact_evidence(tmp_path: Path) -> None:
    public_path, key_id, private = _write_keypair(tmp_path)
    evidence = tmp_path / "snapshot.json"
    evidence.write_text('{"snapshot":"fixture"}\n', encoding="utf-8")
    receipt = tmp_path / "market.json"
    _write_receipt(
        receipt,
        private,
        attestation_type=MARKET_DATA_ATTESTATION,
        subject_kind="immutable_ohlcv_snapshot",
        subject_id="panel-fixture",
        subject_digest="a" * 64,
        evidence={"snapshot_manifest": evidence},
        rights=MARKET_REQUIRED_RIGHTS,
        key_id=key_id,
    )
    verified = verify_receipt(
        receipt,
        public_path,
        attestation_type=MARKET_DATA_ATTESTATION,
        subject_kind="immutable_ohlcv_snapshot",
        subject_id="panel-fixture",
        subject_digest="a" * 64,
        required_rights=MARKET_REQUIRED_RIGHTS,
        evidence_files={"snapshot_manifest": evidence},
    )
    assert verified["public_key_sha256"] == key_id
    assert len(verified["receipt_sha256"]) == 64


def test_unsigned_vendor_claim_is_rejected(tmp_path: Path) -> None:
    receipt = tmp_path / "claim.json"
    receipt.write_text(
        json.dumps(
            {
                "attestation": "independently attested",
                "license_id": "LIC-FAKE",
                "universe_size": 64,
            }
        ),
        encoding="utf-8",
    )
    public_path, _key_id, _private = _write_keypair(tmp_path)
    with pytest.raises(AttestationError, match="schema|signature"):
        verify_receipt(
            receipt,
            public_path,
            attestation_type=PIT64_ATTESTATION,
            subject_kind="pit64_security_master",
            subject_id="fixture",
            subject_digest="b" * 64,
            required_rights=PIT64_REQUIRED_RIGHTS,
            evidence_files={},
        )


def test_dataset_attestation_record_binds_both_inputs(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    attestation_root = dataset / "manifests" / "attestations"
    evidence_root = attestation_root / "evidence"
    evidence_root.mkdir(parents=True)
    public_market, market_key_id, market_private = _write_keypair(tmp_path / "market")
    public_pit64, pit64_key_id, pit64_private = _write_keypair(tmp_path / "pit64")
    market_evidence = evidence_root / "market-snapshot.json"
    market_evidence.write_text(
        json.dumps(
            {
                "panel_id": "panel-fixture",
                "pooled_checksum": "sha256:" + "a" * 64,
                "license": {"acknowledged": True},
            }
        ),
        encoding="utf-8",
    )
    securities = [{"security_id": f"SEC-{index:03d}"} for index in range(64)]
    universe = {
        "universe_version": "v11.2-fixture",
        "certification_eligible": True,
        "securities": securities,
    }
    universe_path = dataset / "manifests" / "universe.json"
    universe_path.parent.mkdir(parents=True, exist_ok=True)
    universe_path.write_text(json.dumps(universe), encoding="utf-8")
    pit_evidence = evidence_root / "pit64-master.json"
    pit_evidence.write_text(json.dumps(securities), encoding="utf-8")
    market_receipt = attestation_root / "market_receipt.json"
    pit64_receipt = attestation_root / "pit64_receipt.json"
    _write_receipt(
        market_receipt,
        market_private,
        attestation_type=MARKET_DATA_ATTESTATION,
        subject_kind="immutable_ohlcv_snapshot",
        subject_id="panel-fixture",
        subject_digest="a" * 64,
        evidence={"snapshot_manifest": market_evidence},
        rights=MARKET_REQUIRED_RIGHTS,
        key_id=market_key_id,
        issuer="fixture market vendor",
    )
    _write_receipt(
        pit64_receipt,
        pit64_private,
        attestation_type=PIT64_ATTESTATION,
        subject_kind="pit64_security_master",
        subject_id="v11.2-fixture",
        subject_digest=security_master_digest(universe),
        evidence={"membership_master": pit_evidence},
        rights=PIT64_REQUIRED_RIGHTS,
        key_id=pit64_key_id,
        issuer="fixture independent reviewer",
    )
    # The copied key files must be inside the dataset for replayable auditing.
    market_key_copy = attestation_root / "market_public_key.pem"
    pit64_key_copy = attestation_root / "pit64_public_key.pem"
    market_key_copy.write_bytes(public_market.read_bytes())
    pit64_key_copy.write_bytes(public_pit64.read_bytes())
    record = {
        "schema_version": 1,
        "market": {
            "receipt": "manifests/attestations/market_receipt.json",
            "public_key": "manifests/attestations/market_public_key.pem",
            "evidence": {
                "snapshot_manifest": "manifests/attestations/evidence/market-snapshot.json"
            },
        },
        "pit64": {
            "receipt": "manifests/attestations/pit64_receipt.json",
            "public_key": "manifests/attestations/pit64_public_key.pem",
            "evidence": {"membership_master": "manifests/attestations/evidence/pit64-master.json"},
        },
    }
    record["verification"] = {
        "market_data": verify_receipt(
            market_receipt,
            market_key_copy,
            attestation_type=MARKET_DATA_ATTESTATION,
            subject_kind="immutable_ohlcv_snapshot",
            subject_id="panel-fixture",
            subject_digest="a" * 64,
            required_rights=MARKET_REQUIRED_RIGHTS,
            evidence_files={"snapshot_manifest": market_evidence},
        ),
        "pit64_membership": verify_receipt(
            pit64_receipt,
            pit64_key_copy,
            attestation_type=PIT64_ATTESTATION,
            subject_kind="pit64_security_master",
            subject_id="v11.2-fixture",
            subject_digest=security_master_digest(universe),
            required_rights=PIT64_REQUIRED_RIGHTS,
            evidence_files={"membership_master": pit_evidence},
        ),
    }
    record["record_sha256"] = canonical_json_digest(record)
    (dataset / "manifests" / "attestations.json").write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )
    summary = verify_dataset_attestation_record(dataset, universe_path)
    assert summary["record_sha256"] == record["record_sha256"]
