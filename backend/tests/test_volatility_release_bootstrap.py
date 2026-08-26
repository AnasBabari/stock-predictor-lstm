from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from config import Settings
from release.bundle import build_release
from services.volatility_release_bootstrap import (
    ReleaseBootstrapError,
    release_source_configured,
    resolve_release_dir,
)


class _Response(io.BytesIO):
    def geturl(self) -> str:
        return "https://releases.example/model-v1.zip"


def _release_archive(tmp_path: Path) -> tuple[bytes, Path]:
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
        tmp_path / "signed-release",
        {"members/seed-41.onnx": b"verified-model-member"},
        {"runtime_schema_version": "volatility-runtime-v1"},
        private_key_path=private_path,
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in release.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(release).as_posix())
    return buffer.getvalue(), public_path


def _settings(tmp_path: Path, archive: bytes, public_path: Path):
    return SimpleNamespace(
        volatility_release_dir=None,
        volatility_public_key_path=str(public_path),
        volatility_release_archive_url="https://releases.example/model-v1.zip",
        volatility_release_archive_sha256=hashlib.sha256(archive).hexdigest(),
        volatility_release_cache_dir=str(tmp_path / "cache"),
        volatility_release_max_archive_mb=2,
        volatility_release_download_timeout_seconds=5,
    )


def test_remote_release_is_downloaded_verified_and_reused(tmp_path: Path) -> None:
    archive, public_path = _release_archive(tmp_path)
    settings = _settings(tmp_path, archive, public_path)
    calls = 0

    def opener(_request, *, timeout):
        nonlocal calls
        calls += 1
        assert timeout == 5
        return _Response(archive)

    first = resolve_release_dir(settings, opener=opener)
    second = resolve_release_dir(settings, opener=opener)
    assert first == second
    assert (first / "manifest.json").is_file()
    assert (first / "members" / "seed-41.onnx").is_file()
    assert calls == 1
    assert release_source_configured(settings)


def test_remote_release_checksum_mismatch_fails_closed(tmp_path: Path) -> None:
    archive, public_path = _release_archive(tmp_path)
    settings = _settings(tmp_path, archive, public_path)
    settings.volatility_release_archive_sha256 = "0" * 64
    with pytest.raises(ReleaseBootstrapError, match="checksum mismatch"):
        resolve_release_dir(settings, opener=lambda *_args, **_kwargs: _Response(archive))
    assert not (Path(settings.volatility_release_cache_dir) / ("0" * 64)).exists()


def test_archive_path_traversal_fails_before_release_verification(tmp_path: Path) -> None:
    _archive, public_path = _release_archive(tmp_path)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("../manifest.json", "{}")
    payload = buffer.getvalue()
    settings = _settings(tmp_path, payload, public_path)
    with pytest.raises(ReleaseBootstrapError, match="escapes"):
        resolve_release_dir(settings, opener=lambda *_args, **_kwargs: _Response(payload))


def test_remote_release_configuration_requires_https_and_digest() -> None:
    with pytest.raises(ValueError, match="configured together"):
        Settings(volatility_release_archive_url="https://example.test/release.zip", _env_file=None)
    with pytest.raises(ValueError, match="must use HTTPS"):
        Settings(
            volatility_release_archive_url="http://example.test/release.zip",
            volatility_release_archive_sha256="a" * 64,
            _env_file=None,
        )
    with pytest.raises(ValueError, match="VOLATILITY_PUBLIC_KEY_PATH"):
        Settings(
            volatility_release_archive_url="https://example.test/release.zip",
            volatility_release_archive_sha256="a" * 64,
            _env_file=None,
        )
    with pytest.raises(ValueError, match="VOLATILITY_PUBLIC_KEY_PATH"):
        Settings(volatility_release_dir="/release", _env_file=None)
    configured = Settings(
        volatility_release_archive_url="https://example.test/release.zip",
        volatility_release_archive_sha256="A" * 64,
        volatility_public_key_path="/keys/public.pem",
        _env_file=None,
    )
    assert configured.volatility_release_archive_sha256 == "a" * 64
