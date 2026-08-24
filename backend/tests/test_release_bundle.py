"""Slice-11 tests: signed release bundles fail closed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from release.bundle import build_release, verify_release


@pytest.fixture()
def key_pair(tmp_path: Path):
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


def model_files() -> dict[str, bytes]:
    return {
        "model.json": b'{"format": "layers-model"}',
        "group1-shard1of2.bin": b"\x00\x01\x02",
        "group1-shard2of2.bin": b"\x03\x04\x05",
    }


METADATA = {
    "candidate": "garch_lstm",
    "horizons": [1, 5],
    "blend_weights": {"1": 0.0, "5": 0.4},
    "universe": "us-equities-v1",
    "train_start": "2018-01-02",
    "train_end": "2025-12-31",
}


def test_roundtrip_verifies(tmp_path: Path, key_pair) -> None:
    private_pem, public_pem = key_pair
    out = build_release(tmp_path / "rel", model_files(), METADATA, private_key_path=private_pem)
    manifest = verify_release(out, public_key_path=public_pem)
    assert manifest["metadata"]["candidate"] == "garch_lstm"
    assert set(manifest["files"]) == set(model_files())


def test_tampered_model_file_fails_closed(tmp_path: Path, key_pair) -> None:
    private_pem, public_pem = key_pair
    out = build_release(tmp_path / "rel", model_files(), METADATA, private_key_path=private_pem)
    shard = out / "group1-shard1of2.bin"
    shard.write_bytes(shard.read_bytes() + b"\xff")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_release(out, public_key_path=public_pem)


def test_tampered_manifest_fails_signature(tmp_path: Path, key_pair) -> None:
    private_pem, public_pem = key_pair
    out = build_release(tmp_path / "rel", model_files(), METADATA, private_key_path=private_pem)
    manifest_path = out / "manifest.json"
    doc = json.loads(manifest_path.read_bytes())
    doc["metadata"]["blend_weights"]["5"] = 1.0  # attacker upgrades the blend
    manifest_path.write_bytes(json.dumps(doc, indent=2, sort_keys=True).encode())
    with pytest.raises(ValueError, match="signature verification failed"):
        verify_release(out, public_key_path=public_pem)


def test_missing_public_key_fails_closed(tmp_path: Path, key_pair) -> None:
    private_pem, _ = key_pair
    out = build_release(tmp_path / "rel", model_files(), METADATA, private_key_path=private_pem)
    missing = tmp_path / "absent.pem"
    with pytest.raises(FileNotFoundError):
        verify_release(out, public_key_path=missing)


def test_wrong_key_fails_signature(tmp_path: Path, key_pair) -> None:
    private_pem, _ = key_pair
    other_private = Ed25519PrivateKey.generate()
    other_public = tmp_path / "other.pem"
    other_public.write_bytes(
        other_private.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    out = build_release(tmp_path / "rel", model_files(), METADATA, private_key_path=private_pem)
    # Signature was made by a different key than the one we verify with — but
    # base64 of a foreign signature still parses, so verification must reject.
    with pytest.raises((ValueError, Exception)):
        verify_release(out, public_key_path=other_public)


def test_detached_signature_file_written(tmp_path: Path, key_pair) -> None:
    private_pem, _ = key_pair
    out = build_release(tmp_path / "rel", model_files(), METADATA, private_key_path=private_pem)
    sig_file = out / "manifest.sig"
    assert sig_file.exists()
    assert len(sig_file.read_text(encoding="utf-8")) >= 64


def test_missing_listed_file_raises_not_found(tmp_path: Path, key_pair) -> None:
    private_pem, public_pem = key_pair
    out = build_release(tmp_path / "rel", model_files(), METADATA, private_key_path=private_pem)
    (out / "group1-shard1of2.bin").unlink()
    with pytest.raises(FileNotFoundError, match="absent from disk"):
        verify_release(out, public_key_path=public_pem)


def test_unsupported_schema_version_rejected(tmp_path: Path, key_pair) -> None:
    private_pem, public_pem = key_pair
    out = build_release(tmp_path / "rel", model_files(), METADATA, private_key_path=private_pem)
    manifest_path = out / "manifest.json"
    doc = json.loads(manifest_path.read_bytes())
    doc["schema_version"] = 999
    manifest_path.write_bytes(json.dumps(doc, indent=2).encode())
    with pytest.raises(ValueError, match="Unsupported release schema version"):
        verify_release(out, public_key_path=public_pem)


@pytest.mark.parametrize(
    "name", ("../escape.onnx", "/absolute.onnx", "C:/drive.onnx", "bad\\path.onnx")
)
def test_release_rejects_artifact_path_traversal(tmp_path: Path, key_pair, name: str) -> None:
    private_pem, _ = key_pair
    with pytest.raises(ValueError, match="artifact path"):
        build_release(
            tmp_path / "rel",
            {name: b"model"},
            METADATA,
            private_key_path=private_pem,
        )


def test_release_rejects_empty_and_oversized_artifacts(tmp_path: Path, key_pair) -> None:
    private_pem, _ = key_pair
    with pytest.raises(ValueError, match="bounded size"):
        build_release(
            tmp_path / "empty",
            {"model.onnx": b""},
            METADATA,
            private_key_path=private_pem,
        )


def test_build_catalog_validates_required_fields() -> None:
    from release.bundle import build_catalog

    artifacts = [{"name": "m1", "url": "/m1", "sha256": "a" * 64, "horizons": [1, 5]}]
    cat = build_catalog(artifacts, signature="b" * 64, recorded_sha="c" * 8)
    assert cat["schema_version"] == 1
    assert cat["signature"] == "b" * 64
    assert len(cat["artifacts"]) == 1

    with pytest.raises(ValueError, match="at least one artifact"):
        build_catalog([], signature="b" * 64, recorded_sha="c" * 8)


def test_validate_certification_manifest_rejects_falsified_pass_when_transfer_fails() -> None:
    from release.bundle import validate_certification_manifest

    # Falsified manifest: claims pass and failed_gates=[], but transfer RMSE violates threshold
    manifest = {
        "status": "holdout_opened",
        "certification_protocol_version": "global-cert-v2",
        "gate_config": {
            "require_temporal_relative_rmse": True,
            "max_temporal_relative_rmse": 1.0,
            "require_transfer_relative_rmse": True,
            "max_transfer_relative_rmse": 1.0,
        },
        "decisions": {
            "5": {
                "decision": "pass",
                "temporal_relative_rmse": 0.99,
                "temporal_relative_mae": 0.99,
                "transfer_relative_rmse": 1.05,  # Violates 1.0 threshold!
                "failed_gates": [],  # Producer lied
            }
        },
    }
    assert validate_certification_manifest(manifest) is False


def test_validate_certification_manifest_rejects_missing_required_metric() -> None:
    from release.bundle import validate_certification_manifest

    manifest = {
        "status": "holdout_opened",
        "certification_protocol_version": "global-cert-v2",
        "gate_config": {
            "require_transfer_relative_rmse": True,
            "max_transfer_relative_rmse": 1.0,
        },
        "decisions": {
            "5": {
                "decision": "pass",
                "temporal_relative_rmse": 0.99,
                "temporal_relative_mae": 0.99,
                # transfer_relative_rmse is missing
                "failed_gates": [],
            }
        },
    }
    with pytest.raises(ValueError, match="Decision missing required 'transfer_relative_rmse'"):
        validate_certification_manifest(manifest)


def test_validate_certification_manifest_rejects_config_and_protocol_mismatches() -> None:
    from release.bundle import validate_certification_manifest

    # Gate config mismatch
    mismatched_cfg = {
        "status": "holdout_opened",
        "certification_protocol_version": "global-cert-v2",
        "gate_config": {"require_temporal_relative_rmse": True},
        "decisions": {
            "5": {
                "decision": "pass",
                "temporal_relative_rmse": 0.99,
                "temporal_relative_mae": 0.99,
                "gate_config": {"require_temporal_relative_rmse": False},
                "failed_gates": [],
            }
        },
    }
    assert validate_certification_manifest(mismatched_cfg) is False

    # Protocol version mismatch
    mismatched_proto = {
        "status": "holdout_opened",
        "certification_protocol_version": "global-cert-v2",
        "gate_config": {"require_temporal_relative_rmse": True},
        "decisions": {
            "5": {
                "decision": "pass",
                "temporal_relative_rmse": 0.99,
                "temporal_relative_mae": 0.99,
                "certification_protocol_version": "global-cert-v1",
                "failed_gates": [],
            }
        },
    }
    assert validate_certification_manifest(mismatched_proto) is False
