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
