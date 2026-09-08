"""Private, local fitted-model artifacts. Never load user-supplied pickle files.

Joblib preserves sklearn pipelines (including their fitted scalers) exactly.
The configured directory must be writable only by the service administrator.
Checksums detect corruption, not malicious writes by someone controlling that directory.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import tempfile
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib

ARTIFACT_VERSION = 1
MAX_BYTES = 128 * 1024 * 1024


def runtime_identity() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        **{p: version(p) for p in ("numpy", "scikit-learn", "joblib")},
    }


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save(path: Path, key: str, payload: dict[str, Any]) -> None:
    buffer = io.BytesIO()
    joblib.dump(payload, buffer)
    content = buffer.getvalue()
    if len(content) > MAX_BYTES:
        raise ValueError("Forecast artifact exceeds size limit")
    digest = hashlib.sha256(content).hexdigest()
    # Content-addressed immutable blob, then atomic manifest publication.
    atomic_write(path.parent / f"{digest}.joblib", content)
    manifest = {
        "version": ARTIFACT_VERSION,
        "key": key,
        "runtime": runtime_identity(),
        "sha256": digest,
    }
    atomic_write(path, json.dumps(manifest, sort_keys=True).encode())


def load(path: Path, key: str) -> dict[str, Any] | None:
    try:
        if path.stat().st_size > 16384:
            return None
        manifest = json.loads(path.read_bytes())
        if (
            manifest.get("version") != ARTIFACT_VERSION
            or manifest.get("key") != key
            or manifest.get("runtime") != runtime_identity()
        ):
            return None
        digest = manifest["sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            return None
        blob = path.parent / f"{digest}.joblib"
        if blob.stat().st_size > MAX_BYTES:
            return None
        content = blob.read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            return None
        value = joblib.load(io.BytesIO(content))
        return value if isinstance(value, dict) else None
    except Exception:
        # Cache failures must not replace the existing learned training path.
        return None
