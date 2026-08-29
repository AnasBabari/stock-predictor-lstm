"""Tests for V10 release bundle assembly, checksum verification, and detached signature."""

from __future__ import annotations

from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from research.volatility_forecasting.export_v10 import (
    assemble_release_bundle,
    verify_release_bundle_integrity,
)
from research.volatility_forecasting.signing_v10 import (
    ReleaseSignatureError,
    verify_detached_signature,
)


def test_bundle_assembly_and_checksum_verification(tmp_path: Path) -> None:
    bundle_dir = assemble_release_bundle(
        output_dir=tmp_path,
        bundle_id="release-v10-001",
        protocol_version="volatility-v10",
        certified_horizons=[1, 3],
        model_family_by_horizon={1: "tcn", 3: "har"},
        feature_schema_sha256="0" * 64,
        universe_sha256="1" * 64,
        files_to_include={
            "models/h1_tcn.onnx": b"fake_onnx_model_bytes",
            "scalers/h1_scaler.json": b"{}",
        },
    )
    assert (bundle_dir / "manifest.json").exists()
    assert (bundle_dir / "models" / "h1_tcn.onnx").exists()
    assert verify_release_bundle_integrity(bundle_dir) is True


def test_tampered_bundle_file_fails_verification(tmp_path: Path) -> None:
    bundle_dir = assemble_release_bundle(
        output_dir=tmp_path,
        bundle_id="release-v10-002",
        protocol_version="volatility-v10",
        certified_horizons=[1],
        model_family_by_horizon={1: "tcn"},
        feature_schema_sha256="0" * 64,
        universe_sha256="1" * 64,
        files_to_include={
            "models/h1_tcn.onnx": b"valid_onnx_bytes",
        },
    )
    # Tamper with the model file
    (bundle_dir / "models" / "h1_tcn.onnx").write_bytes(b"tampered_onnx_bytes")

    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_release_bundle_integrity(bundle_dir)


def test_detached_ed25519_signature_verification(tmp_path: Path) -> None:
    # Generate temporary test key in memory only
    private_key = Ed25519PrivateKey.generate()
    public_key_pem = private_key.public_key().public_bytes(
        encoding=Encoding.PEM,
        format=PublicFormat.SubjectPublicKeyInfo,
    )

    data = b"canonical_manifest_bytes_to_sign"
    signature = private_key.sign(data)

    assert verify_detached_signature(data, signature, public_key_pem) is True

    # Tampered data must fail
    with pytest.raises(ReleaseSignatureError):
        verify_detached_signature(b"tampered_data", signature, public_key_pem)
