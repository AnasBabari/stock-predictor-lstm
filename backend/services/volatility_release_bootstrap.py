"""Securely materialise an immutable signed volatility release archive.

Render's free service filesystem is ephemeral and does not provide a
persistent disk. A certified bundle can therefore be published as an
immutable HTTPS archive and downloaded on cold start. The archive digest is
configuration, while the embedded manifest and every model member are still
verified with the pinned Ed25519 public key before the runtime sees them.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tempfile
import threading
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

from release.bundle import MAX_RELEASE_FILES, MAX_RELEASE_TOTAL_BYTES, verify_release

_BOOTSTRAP_LOCK = threading.Lock()


class ReleaseBootstrapError(RuntimeError):
    """Remote release could not be downloaded and verified safely."""


def release_source_configured(settings: Any) -> bool:
    """Return whether either supported signed-release source is complete."""
    local = bool(settings.volatility_release_dir and settings.volatility_public_key_path)
    remote = bool(
        settings.volatility_release_archive_url
        and settings.volatility_release_archive_sha256
        and settings.volatility_public_key_path
    )
    return local or remote


def _safe_member_path(root: Path, member: zipfile.ZipInfo) -> Path:
    name = member.filename
    if not name or "\\" in name or ":" in name:
        raise ReleaseBootstrapError("release archive contains an invalid path")
    relative = PurePosixPath(name)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ReleaseBootstrapError("release archive path escapes the extraction directory")
    mode = member.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise ReleaseBootstrapError("release archive may not contain symbolic links")
    target = root.joinpath(*relative.parts).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as error:
        raise ReleaseBootstrapError(
            "release archive path escapes the extraction directory"
        ) from error
    return target


def _download_archive(
    url: str,
    destination: Path,
    *,
    expected_sha256: str,
    maximum_bytes: int,
    timeout_seconds: int,
    opener: Callable[..., Any],
) -> None:
    if urlparse(url).scheme.lower() != "https":
        raise ReleaseBootstrapError("release archive URL must use HTTPS")
    digest = hashlib.sha256()
    downloaded = 0
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/zip", "User-Agent": "StockLSTM-release-bootstrap/1"},
    )
    try:
        with opener(request, timeout=timeout_seconds) as response, destination.open("xb") as handle:
            final_url = response.geturl() if hasattr(response, "geturl") else url
            if urlparse(final_url).scheme.lower() != "https":
                raise ReleaseBootstrapError("release archive redirected away from HTTPS")
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                downloaded += len(chunk)
                if downloaded > maximum_bytes:
                    raise ReleaseBootstrapError("release archive exceeds the configured size limit")
                digest.update(chunk)
                handle.write(chunk)
    except ReleaseBootstrapError:
        raise
    except (OSError, ValueError) as error:
        raise ReleaseBootstrapError("release archive download failed") from error
    if downloaded == 0:
        raise ReleaseBootstrapError("release archive is empty")
    if digest.hexdigest() != expected_sha256:
        raise ReleaseBootstrapError("release archive checksum mismatch")


def _extract_archive(archive: Path, destination: Path, *, maximum_bytes: int) -> None:
    try:
        with zipfile.ZipFile(archive) as bundle:
            members = bundle.infolist()
            file_count = len([member for member in members if not member.is_dir()])
            if not members or file_count > MAX_RELEASE_FILES + 2:
                raise ReleaseBootstrapError("release archive file count is invalid")
            total = sum(member.file_size for member in members if not member.is_dir())
            expanded_limit = min(maximum_bytes, MAX_RELEASE_TOTAL_BYTES + 2 * 1024 * 1024)
            if total <= 0 or total > expanded_limit:
                raise ReleaseBootstrapError("release archive expands beyond the size limit")
            for member in members:
                target = _safe_member_path(destination, member)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
    except ReleaseBootstrapError:
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError) as error:
        raise ReleaseBootstrapError("release archive extraction failed") from error


def resolve_release_dir(
    settings: Any,
    *,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> Path:
    """Resolve and verify the configured local or immutable remote release."""
    public_key = settings.volatility_public_key_path
    if settings.volatility_release_dir:
        if not public_key:
            raise ReleaseBootstrapError("volatility public key is not configured")
        local = Path(settings.volatility_release_dir).resolve()
        verify_release(local, public_key_path=Path(public_key))
        return local

    url = settings.volatility_release_archive_url
    expected = settings.volatility_release_archive_sha256
    if not url or not expected or not public_key:
        raise ReleaseBootstrapError("no certified volatility release is configured")
    cache_root = Path(settings.volatility_release_cache_dir).resolve()
    target = cache_root / expected
    maximum_bytes = int(settings.volatility_release_max_archive_mb) * 1024 * 1024

    with _BOOTSTRAP_LOCK:
        if target.exists():
            verify_release(target, public_key_path=Path(public_key))
            return target
        cache_root.mkdir(parents=True, exist_ok=True)
        workspace = Path(tempfile.mkdtemp(prefix=".bootstrap-", dir=cache_root))
        try:
            archive = workspace / "release.zip"
            extracted = workspace / "bundle"
            extracted.mkdir()
            _download_archive(
                url,
                archive,
                expected_sha256=expected,
                maximum_bytes=maximum_bytes,
                timeout_seconds=int(settings.volatility_release_download_timeout_seconds),
                opener=opener,
            )
            _extract_archive(archive, extracted, maximum_bytes=maximum_bytes)
            verify_release(extracted, public_key_path=Path(public_key))
            try:
                os.replace(extracted, target)
            except OSError as error:
                # The lock is process-local. Multiple service processes can
                # finish the same cold-start download concurrently, and
                # directory replacement reports platform-specific OSError
                # subclasses. Accept only a winner that independently passes
                # the full signed-release verification; otherwise preserve the
                # original promotion failure.
                if not target.exists():
                    raise ReleaseBootstrapError(
                        "verified release could not be promoted into the cache"
                    ) from error
                verify_release(target, public_key_path=Path(public_key))
            return target
        finally:
            shutil.rmtree(workspace, ignore_errors=True)
