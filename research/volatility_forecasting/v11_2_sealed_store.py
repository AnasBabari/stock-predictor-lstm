"""Physically separated, encrypted V11.2 development and test datasets.

The development loader never receives test arrays, indices, or a decryption key.
The one-shot certification loader is deliberately a separate API and writes its
consumption marker before decrypting the holdout.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import gc
import hashlib
import json
import os
import secrets
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .v11_2_protocol import V11_2_PROTOCOL_ID
from .v11_2_split import V112Split


class V112SealedAccessError(RuntimeError):
    """Raised when sealed data cannot be safely consumed."""


@dataclass(frozen=True)
class V112DevelopmentData:
    train_features: np.ndarray
    train_returns: np.ndarray
    train_rv: np.ndarray
    train_dates: tuple[str, ...]
    validation_features: np.ndarray
    validation_returns: np.ndarray
    validation_rv: np.ndarray
    validation_dates: tuple[str, ...]
    protocol_id: str
    panel_sha256: str
    split_sha256: str


@dataclass(frozen=True)
class V112SealedMetadata:
    protocol_id: str
    panel_sha256: str
    split_sha256: str
    schema_sha256: str
    ciphertext_sha256: str
    nonce_hex: str
    test_stock_origin_observations: int
    test_unique_sessions: int
    test_sessions: tuple[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_id": self.protocol_id,
            "panel_sha256": self.panel_sha256,
            "split_sha256": self.split_sha256,
            "schema_sha256": self.schema_sha256,
            "ciphertext_sha256": self.ciphertext_sha256,
            "nonce_hex": self.nonce_hex,
            "test_stock_origin_observations": self.test_stock_origin_observations,
            "test_unique_sessions": self.test_unique_sessions,
            "test_sessions": self.test_sessions,
            "sealed_test_status": "LOCKED_UNOPENED",
        }


@dataclass(frozen=True)
class V112SealedTestPayload:
    features: np.ndarray
    returns: np.ndarray
    rv: np.ndarray
    dates: tuple[str, ...]
    unseal_token: str
    split_sha256: str


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _key_bytes(key_path: Path, *, create: bool) -> bytes:
    if key_path.exists():
        key = key_path.read_bytes()
    elif create:
        key = secrets.token_bytes(32)
        key_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(key_path, key)
        with contextlib.suppress(OSError):
            key_path.chmod(0o600)
    else:
        raise V112SealedAccessError(f"external V11.2 holdout key not found: {key_path}")
    if len(key) != 32:
        raise V112SealedAccessError("V11.2 holdout key must be exactly 32 bytes")
    return key


def _assert_external_key(key_path: Path, repository_root: Path | None) -> None:
    if repository_root is None:
        return
    key_abs = key_path.resolve()
    repo_abs = repository_root.resolve()
    try:
        key_abs.relative_to(repo_abs)
    except ValueError:
        return
    raise V112SealedAccessError("holdout encryption key must be outside the repository")


def _array_payload(
    features: np.ndarray,
    returns: np.ndarray,
    rv: np.ndarray,
    dates: Iterable[str],
) -> bytes:
    with tempfile.SpooledTemporaryFile(max_size=64 * 1024 * 1024) as buffer:
        np.savez_compressed(
            buffer,
            features=np.asarray(features, dtype=np.float32),
            returns=np.asarray(returns, dtype=np.float32),
            rv=np.asarray(rv, dtype=np.float32),
            dates=np.asarray(list(dates), dtype="U32"),
        )
        buffer.seek(0)
        return buffer.read()


def _validate_arrays(
    features: np.ndarray,
    returns: np.ndarray,
    rv: np.ndarray,
    dates: Iterable[str],
) -> None:
    n = len(list(dates)) if not isinstance(dates, tuple) else len(dates)
    if n == 0 or len(features) != n or len(returns) != n or len(rv) != n:
        raise ValueError("feature, target, and date row counts must match and be non-zero")
    if not all(np.isfinite(np.asarray(value)).all() for value in (features, returns, rv)):
        raise ValueError("V11.2 sealed payload contains non-finite values")


def seal_v112_dataset(
    *,
    dates: list[str],
    features: np.ndarray,
    returns: np.ndarray,
    rv: np.ndarray,
    split: V112Split,
    output_dir: Path,
    panel_sha256: str,
    schema_sha256: str,
    key_path: Path,
    repository_root: Path | None = None,
) -> V112SealedMetadata:
    """Write development files and encrypted test bytes, never test plaintext."""
    _assert_external_key(key_path, repository_root)
    dates_list = list(dates)
    _validate_arrays(features, returns, rv, dates_list)
    if max(split.test_indices, default=-1) >= len(dates_list):
        raise ValueError("split index exceeds panel length")
    development_dir = output_dir / "development"
    sealed_dir = output_dir / "sealed"
    manifests_dir = output_dir / "manifests"
    development_dir.mkdir(parents=True, exist_ok=True)
    sealed_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    train = split.train_indices
    validation = split.validation_indices
    test = split.test_indices
    np.savez_compressed(
        development_dir / "train.npz",
        features=np.asarray(features[train], dtype=np.float32),
        returns=np.asarray(returns[train], dtype=np.float32),
        rv=np.asarray(rv[train], dtype=np.float32),
        dates=np.asarray([dates_list[i] for i in train], dtype="U32"),
    )
    np.savez_compressed(
        development_dir / "validation.npz",
        features=np.asarray(features[validation], dtype=np.float32),
        returns=np.asarray(returns[validation], dtype=np.float32),
        rv=np.asarray(rv[validation], dtype=np.float32),
        dates=np.asarray([dates_list[i] for i in validation], dtype="U32"),
    )

    nonce = secrets.token_bytes(12)
    associated = json.dumps(
        {
            "protocol_id": V11_2_PROTOCOL_ID,
            "panel_sha256": panel_sha256,
            "schema_sha256": schema_sha256,
            "split_sha256": split.split_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    plaintext = _array_payload(
        features[test], returns[test], rv[test], [dates_list[i] for i in test]
    )
    ciphertext = AESGCM(_key_bytes(key_path, create=True)).encrypt(nonce, plaintext, associated)
    ciphertext_sha = hashlib.sha256(ciphertext).hexdigest()
    _atomic_write(sealed_dir / "test_payload.aesgcm", ciphertext)
    _atomic_write((sealed_dir / "test_payload.sha256"), ciphertext_sha.encode("ascii"))
    metadata = V112SealedMetadata(
        protocol_id=V11_2_PROTOCOL_ID,
        panel_sha256=panel_sha256,
        split_sha256=split.split_sha256,
        schema_sha256=schema_sha256,
        ciphertext_sha256=ciphertext_sha,
        nonce_hex=nonce.hex(),
        test_stock_origin_observations=len(test),
        test_unique_sessions=len({dates_list[i] for i in test}),
        test_sessions=split.test_sessions,
    )
    _atomic_write(
        sealed_dir / "sealed_metadata.json",
        json.dumps(metadata.to_dict(), indent=2, sort_keys=True).encode("utf-8"),
    )
    _atomic_write(
        manifests_dir / "development_manifest.json",
        json.dumps(
            {
                "protocol_id": V11_2_PROTOCOL_ID,
                "panel_sha256": panel_sha256,
                "schema_sha256": schema_sha256,
                "split_sha256": split.split_sha256,
                "sealed_test_status": "LOCKED_UNOPENED",
            },
            indent=2,
            sort_keys=True,
        ).encode("utf-8"),
    )
    del plaintext, ciphertext
    gc.collect()
    return metadata


def load_v112_development(output_dir: Path) -> V112DevelopmentData:
    """Load only train/validation files; no sealed path or key is accepted."""
    development_dir = output_dir / "development"
    train_path = development_dir / "train.npz"
    validation_path = development_dir / "validation.npz"
    manifest_path = output_dir / "manifests" / "development_manifest.json"
    if not train_path.exists() or not validation_path.exists() or not manifest_path.exists():
        raise V112SealedAccessError("V11.2 development dataset is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with (
        np.load(train_path, allow_pickle=False) as train,
        np.load(validation_path, allow_pickle=False) as validation,
    ):
        return V112DevelopmentData(
            train_features=np.asarray(train["features"], dtype=np.float32),
            train_returns=np.asarray(train["returns"], dtype=np.float32),
            train_rv=np.asarray(train["rv"], dtype=np.float32),
            train_dates=tuple(str(value) for value in train["dates"].tolist()),
            validation_features=np.asarray(validation["features"], dtype=np.float32),
            validation_returns=np.asarray(validation["returns"], dtype=np.float32),
            validation_rv=np.asarray(validation["rv"], dtype=np.float32),
            validation_dates=tuple(str(value) for value in validation["dates"].tolist()),
            protocol_id=str(manifest["protocol_id"]),
            panel_sha256=str(manifest["panel_sha256"]),
            split_sha256=str(manifest["split_sha256"]),
        )


def unseal_v112_test_once(
    *,
    output_dir: Path,
    key_path: Path,
    candidate_digest: str,
    repository_root: Path | None = None,
) -> V112SealedTestPayload:
    """One-shot certification loader; writes the lock before decrypting."""
    _assert_external_key(key_path, repository_root)
    if len(candidate_digest) < 64:
        raise V112SealedAccessError("candidate digest must be a SHA-256 hex digest")
    sealed_dir = output_dir / "sealed"
    lock_path = sealed_dir / "SEALED_TEST_OPENED.json"
    metadata_path = sealed_dir / "sealed_metadata.json"
    payload_path = sealed_dir / "test_payload.aesgcm"
    if lock_path.exists():
        raise V112SealedAccessError("V11.2 sealed test has already been opened")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    ciphertext = payload_path.read_bytes()
    expected = str(metadata["ciphertext_sha256"])
    if hashlib.sha256(ciphertext).hexdigest() != expected:
        raise V112SealedAccessError("sealed ciphertext digest mismatch")
    split_sha = str(metadata["split_sha256"])
    token = hashlib.sha256(f"{candidate_digest}:{split_sha}".encode()).hexdigest()
    lock = {
        "protocol_id": metadata["protocol_id"],
        "candidate_digest": candidate_digest,
        "split_sha256": split_sha,
        "ciphertext_sha256": expected,
        "opened_at": dt.datetime.now(dt.UTC).isoformat(),
        "unseal_token": token,
    }
    if lock_path.exists():
        raise V112SealedAccessError("sealed test open race detected")
    _atomic_write(lock_path, json.dumps(lock, indent=2, sort_keys=True).encode("utf-8"))
    associated = json.dumps(
        {
            "protocol_id": metadata["protocol_id"],
            "panel_sha256": metadata["panel_sha256"],
            "schema_sha256": metadata["schema_sha256"],
            "split_sha256": split_sha,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    plaintext = AESGCM(_key_bytes(key_path, create=False)).decrypt(
        bytes.fromhex(str(metadata["nonce_hex"])), ciphertext, associated
    )
    with tempfile.NamedTemporaryFile(suffix=".npz") as temporary:
        temporary.write(plaintext)
        temporary.flush()
        with np.load(temporary.name, allow_pickle=False) as payload:
            features = np.asarray(payload["features"], dtype=np.float32)
            returns = np.asarray(payload["returns"], dtype=np.float32)
            rv = np.asarray(payload["rv"], dtype=np.float32)
            dates = tuple(str(value) for value in payload["dates"].tolist())
    del plaintext, ciphertext
    gc.collect()
    return V112SealedTestPayload(
        features=features,
        returns=returns,
        rv=rv,
        dates=dates,
        unseal_token=token,
        split_sha256=split_sha,
    )
