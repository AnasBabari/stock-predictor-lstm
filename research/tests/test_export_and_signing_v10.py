"""Tests for V10 release bundle assembly, path traversal rejection, and detached signature."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from research.volatility_forecasting.export_v10 import (
    ReleasePathTraversalError,
    assemble_release_bundle,
    verify_release_bundle_integrity,
)
from research.volatility_forecasting.signing_v10 import (
    sign_release_manifest_detached,
)


@pytest.fixture
def dummy_keys() -> tuple[Ed25519PrivateKey, bytes]:
    priv = Ed25519PrivateKey.generate()
    pub_pem = priv.public_key().public_bytes(
        encoding=Encoding.PEM,
        format=PublicFormat.SubjectPublicKeyInfo,
    )
    return priv, pub_pem


def test_bundle_assembly_rejects_path_traversal(tmp_path: Path) -> None:
    cert_report = {"certified_horizons": [1]}
    with pytest.raises(ReleasePathTraversalError, match="Unsafe path"):
        assemble_release_bundle(
            output_dir=tmp_path,
            bundle_id="release-v10-bad-path",
            certification_report=cert_report,
            protocol_version="volatility-v10",
            model_family_by_horizon={1: "tcn"},
            feature_schema_sha256="0" * 64,
            universe_sha256="1" * 64,
            files_to_include={
                "../models/h1_tcn.onnx": b"fake_bytes",
            },
        )


def test_bundle_assembly_requires_certified_horizons(tmp_path: Path) -> None:
    cert_report = {"certified_horizons": []}
    with pytest.raises(ValueError, match="0 certified horizons"):
        assemble_release_bundle(
            output_dir=tmp_path,
            bundle_id="release-v10-no-cert",
            certification_report=cert_report,
            protocol_version="volatility-v10",
            model_family_by_horizon={},
            feature_schema_sha256="0" * 64,
            universe_sha256="1" * 64,
            files_to_include={"models/h1.onnx": b"bytes"},
        )


def test_bundle_assembly_and_detached_signing_verification(
    tmp_path: Path, dummy_keys: tuple[Ed25519PrivateKey, bytes]
) -> None:
    priv_key, pub_pem = dummy_keys

    bundle_dir = assemble_release_bundle(
        output_dir=tmp_path,
        bundle_id="release-v10-valid",
        certification_report={"certified_horizons": [1, 3]},
        protocol_version="volatility-v10",
        model_family_by_horizon={1: "tcn", 3: "har"},
        feature_schema_sha256="0" * 64,
        universe_sha256="1" * 64,
        files_to_include={
            "models/h1_tcn.onnx": b"model_onnx_bytes",
            "scalers/h1_scaler.json": b"{}",
        },
    )

    # Detached sign manifest
    manifest_bytes = (bundle_dir / "manifest.json").read_bytes()
    sig_bytes = sign_release_manifest_detached(manifest_bytes, priv_key)
    (bundle_dir / "signature.ed25519").write_bytes(sig_bytes)

    assert verify_release_bundle_integrity(bundle_dir, pub_pem) is True


def test_model_substitution_with_tampered_checksums_fails_verification(
    tmp_path: Path, dummy_keys: tuple[Ed25519PrivateKey, bytes]
) -> None:
    """Security regression test: substituting a model and updating checksums.json
    WITHOUT resigning manifest.json MUST fail verification."""
    priv_key, pub_pem = dummy_keys

    bundle_dir = assemble_release_bundle(
        output_dir=tmp_path,
        bundle_id="release-v10-tamper-target",
        certification_report={"certified_horizons": [1]},
        protocol_version="volatility-v10",
        model_family_by_horizon={1: "tcn"},
        feature_schema_sha256="0" * 64,
        universe_sha256="1" * 64,
        files_to_include={"models/h1_tcn.onnx": b"original_legit_model_bytes"},
    )

    # Detached sign the original manifest
    manifest_bytes = (bundle_dir / "manifest.json").read_bytes()
    sig_bytes = sign_release_manifest_detached(manifest_bytes, priv_key)
    (bundle_dir / "signature.ed25519").write_bytes(sig_bytes)

    # Attacker replaces model and updates checksums.json, but cannot sign manifest
    (bundle_dir / "models" / "h1_tcn.onnx").write_bytes(b"malicious_replacement_model_bytes")
    import hashlib

    fake_sha = hashlib.sha256(b"malicious_replacement_model_bytes").hexdigest()
    (bundle_dir / "checksums.json").write_text(
        json.dumps({"models/h1_tcn.onnx": fake_sha}, indent=2), encoding="utf-8"
    )

    # Verification MUST fail closed
    with pytest.raises(ValueError, match="Checksum mismatch|diverges"):
        verify_release_bundle_integrity(bundle_dir, pub_pem)
