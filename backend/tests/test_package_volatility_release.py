from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from scripts.package_volatility_release import package_release

from release.bundle import build_release


def _signed_release(tmp_path: Path) -> tuple[Path, Path]:
    private = Ed25519PrivateKey.generate()
    private_path = tmp_path / "private.pem"
    public_path = tmp_path / "public.pem"
    private_path.write_bytes(
        private.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    public_path.write_bytes(
        private.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    release = build_release(
        tmp_path / "release",
        {"members/seed-41.onnx": b"onnx-member"},
        {"model_id": "candidate-v1"},
        private_key_path=private_path,
    )
    return release, public_path


def test_packages_only_verified_release_files_deterministically(tmp_path: Path) -> None:
    release, public_path = _signed_release(tmp_path)
    first = package_release(release, public_path, tmp_path / "first.zip")
    second = package_release(release, public_path, tmp_path / "second.zip")
    assert first["archive_sha256"] == second["archive_sha256"]
    assert first["model_id"] == "candidate-v1"
    with zipfile.ZipFile(tmp_path / "first.zip") as archive:
        assert archive.namelist() == [
            "manifest.json",
            "manifest.sig",
            "members/seed-41.onnx",
        ]


def test_refuses_unexpected_files_and_overwrite(tmp_path: Path) -> None:
    release, public_path = _signed_release(tmp_path)
    (release / "private.pem").write_text("must never ship", encoding="utf-8")
    with pytest.raises(ValueError, match="not canonical"):
        package_release(release, public_path, tmp_path / "release.zip")
    (release / "private.pem").unlink()
    output = tmp_path / "release.zip"
    package_release(release, public_path, output)
    with pytest.raises(FileExistsError, match="immutable"):
        package_release(release, public_path, output)
