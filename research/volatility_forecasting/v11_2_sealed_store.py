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
import io
import json
import os
import secrets
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .v11_2_protocol import (
    V11_2_HORIZONS,
    V11_2_PROTOCOL_ID,
    V112Protocol,
    feature_schema_digest,
)
from .v11_2_split import V112Split


class V112SealedAccessError(RuntimeError):
    """Raised when sealed data cannot be safely consumed."""


@dataclass(frozen=True)
class V112DevelopmentData:
    train_features: np.ndarray
    train_returns: np.ndarray
    train_rv: np.ndarray
    train_dates: tuple[str, ...]
    train_security_ids: tuple[str, ...]
    validation_features: np.ndarray
    validation_returns: np.ndarray
    validation_rv: np.ndarray
    validation_dates: tuple[str, ...]
    validation_security_ids: tuple[str, ...]
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
    test_unique_securities: int
    test_identity_sha256: str
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
            "test_unique_securities": self.test_unique_securities,
            "test_identity_sha256": self.test_identity_sha256,
            "test_sessions": self.test_sessions,
            "sealed_test_status": "LOCKED_UNOPENED",
        }


@dataclass(frozen=True)
class V112SealedTestPayload:
    features: np.ndarray
    returns: np.ndarray
    rv: np.ndarray
    dates: tuple[str, ...]
    security_ids: tuple[str, ...]
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


def _atomic_create(path: Path, data: bytes) -> None:
    """Create a marker exactly once, failing if another process won the race."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise V112SealedAccessError("sealed test open race detected") from exc
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        with contextlib.suppress(OSError):
            path.unlink()
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_digest(value: object, label: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise V112SealedAccessError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    """Read a JSON object and expose malformed artifacts as a sealed-access error."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise V112SealedAccessError(f"{label} is malformed") from exc
    if not isinstance(payload, dict):
        raise V112SealedAccessError(f"{label} must contain a JSON object")
    return payload


def _key_bytes(key_path: Path, *, create: bool) -> bytes:
    if key_path.exists():
        key = key_path.read_bytes()
    elif create:
        key = secrets.token_bytes(32)
        key_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            _atomic_create(key_path, key)
        except V112SealedAccessError:
            # Another sealing process won the creation race; use its fully
            # fsynced key rather than replacing it with a different key.
            key = key_path.read_bytes()
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
    security_ids: Iterable[str],
) -> bytes:
    with tempfile.SpooledTemporaryFile(max_size=64 * 1024 * 1024) as buffer:
        np.savez_compressed(
            buffer,
            features=np.asarray(features, dtype=np.float32),
            returns=np.asarray(returns, dtype=np.float32),
            rv=np.asarray(rv, dtype=np.float32),
            dates=np.asarray(list(dates), dtype="U32"),
            security_ids=np.asarray(list(security_ids), dtype="U128"),
        )
        buffer.seek(0)
        return buffer.read()


def _validate_arrays(
    features: np.ndarray,
    returns: np.ndarray,
    rv: np.ndarray,
    dates: Iterable[str],
    security_ids: Iterable[str],
) -> None:
    n = len(dates)
    security_list = [str(value) for value in security_ids]
    feature_values = np.asarray(features)
    return_values = np.asarray(returns)
    rv_values = np.asarray(rv)
    if (
        n == 0
        or len(feature_values) != n
        or len(return_values) != n
        or len(rv_values) != n
        or len(security_list) != n
    ):
        raise ValueError("feature, target, and date row counts must match and be non-zero")
    if any(not value.strip() for value in security_list):
        raise ValueError("security IDs must be non-empty")
    if len(set(zip(security_list, dates, strict=True))) != n:
        raise ValueError("security/session observations must be unique")
    if feature_values.ndim not in (2, 3):
        raise ValueError("features must have shape [rows, features] or [rows, window, features]")
    if return_values.ndim != 2 or return_values.shape[1] != len(V11_2_HORIZONS):
        raise ValueError("returns must contain one column for each V11.2 horizon")
    if rv_values.ndim != 2 or rv_values.shape[1] != len(V11_2_HORIZONS):
        raise ValueError("realized variance must contain one column for each V11.2 horizon")
    if not all(np.isfinite(value).all() for value in (feature_values, return_values, rv_values)):
        raise ValueError("V11.2 sealed payload contains non-finite values")


def security_identity_digest(security_ids: Iterable[str], dates: Iterable[str]) -> str:
    """Hash the ordered security/session identity paired with each row."""
    ids = [str(value) for value in security_ids]
    date_values = [str(value) for value in dates]
    if len(ids) != len(date_values) or not ids:
        raise ValueError("security IDs and dates must have equal non-zero length")
    payload = "\n".join(
        f"{security}|{date}" for security, date in zip(ids, date_values, strict=True)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def seal_v112_dataset(
    *,
    dates: list[str],
    security_ids: list[str],
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
    _require_digest(panel_sha256, "panel digest")
    _require_digest(split.split_sha256, "split digest")
    _require_digest(schema_sha256, "schema digest")
    if schema_sha256 != feature_schema_digest():
        raise V112SealedAccessError("schema digest does not match the V11.2 feature contract")
    dates_list = list(dates)
    security_list = [str(value) for value in security_ids]
    _validate_arrays(features, returns, rv, dates_list, security_list)
    if max(split.test_indices, default=-1) >= len(dates_list):
        raise ValueError("split index exceeds panel length")
    all_indices = np.concatenate(
        (split.train_indices, split.validation_indices, split.test_indices)
    )
    if len(all_indices) != len(set(int(value) for value in all_indices)):
        raise ValueError("split partitions contain duplicate row indices")
    if any(int(value) < 0 or int(value) >= len(dates_list) for value in all_indices):
        raise ValueError("split partitions contain an out-of-range row index")
    development_dir = output_dir / "development"
    sealed_dir = output_dir / "sealed"
    manifests_dir = output_dir / "manifests"
    existing = [
        development_dir / "train.npz",
        development_dir / "validation.npz",
        sealed_dir / "test_payload.aesgcm",
        sealed_dir / "sealed_metadata.json",
        manifests_dir / "development_manifest.json",
    ]
    if any(path.exists() for path in existing):
        raise V112SealedAccessError("V11.2 dataset output already exists and is immutable")
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
        security_ids=np.asarray([security_list[i] for i in train], dtype="U128"),
    )
    np.savez_compressed(
        development_dir / "validation.npz",
        features=np.asarray(features[validation], dtype=np.float32),
        returns=np.asarray(returns[validation], dtype=np.float32),
        rv=np.asarray(rv[validation], dtype=np.float32),
        dates=np.asarray([dates_list[i] for i in validation], dtype="U32"),
        security_ids=np.asarray([security_list[i] for i in validation], dtype="U128"),
    )
    train_path = development_dir / "train.npz"
    validation_path = development_dir / "validation.npz"

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
        features[test],
        returns[test],
        rv[test],
        [dates_list[i] for i in test],
        [security_list[i] for i in test],
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
        test_unique_securities=len({security_list[i] for i in test}),
        test_identity_sha256=security_identity_digest(
            [security_list[i] for i in test], [dates_list[i] for i in test]
        ),
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
                "feature_schema_version": V112Protocol().feature_schema_version,
                "feature_names": list(V112Protocol().feature_names),
                "window_size": V112Protocol().window_size,
                "horizons": list(V11_2_HORIZONS),
                "train_sha256": _sha256_file(train_path),
                "validation_sha256": _sha256_file(validation_path),
                "train_identity_sha256": security_identity_digest(
                    [security_list[i] for i in train], [dates_list[i] for i in train]
                ),
                "validation_identity_sha256": security_identity_digest(
                    [security_list[i] for i in validation], [dates_list[i] for i in validation]
                ),
                "train_rows": len(train),
                "validation_rows": len(validation),
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
    manifest = _read_json_object(manifest_path, "V11.2 development manifest")
    protocol = V112Protocol()
    if manifest.get("protocol_id") != protocol.protocol_id:
        raise V112SealedAccessError("V11.2 development manifest protocol mismatch")
    if manifest.get("sealed_test_status") != "LOCKED_UNOPENED":
        raise V112SealedAccessError("V11.2 development manifest does not prove an unopened holdout")
    if manifest.get("schema_sha256") != feature_schema_digest():
        raise V112SealedAccessError("V11.2 development schema digest mismatch")
    if manifest.get("feature_schema_version") != protocol.feature_schema_version:
        raise V112SealedAccessError("V11.2 development feature schema version mismatch")
    if manifest.get("feature_names") != list(protocol.feature_names):
        raise V112SealedAccessError("V11.2 development feature ordering mismatch")
    if manifest.get("window_size") != protocol.window_size or manifest.get("horizons") != list(
        protocol.horizons
    ):
        raise V112SealedAccessError("V11.2 development target geometry mismatch")
    if _sha256_file(train_path) != _require_digest(manifest.get("train_sha256"), "train digest"):
        raise V112SealedAccessError("V11.2 train bytes do not match the development manifest")
    if _sha256_file(validation_path) != _require_digest(
        manifest.get("validation_sha256"), "validation digest"
    ):
        raise V112SealedAccessError("V11.2 validation bytes do not match the development manifest")
    try:
        with (
            np.load(train_path, allow_pickle=False) as train,
            np.load(validation_path, allow_pickle=False) as validation,
        ):
            train_features = np.asarray(train["features"], dtype=np.float32)
            train_returns = np.asarray(train["returns"], dtype=np.float32)
            train_rv = np.asarray(train["rv"], dtype=np.float32)
            train_dates = tuple(str(value) for value in train["dates"].tolist())
            train_security_ids = tuple(str(value) for value in train["security_ids"].tolist())
            validation_features = np.asarray(validation["features"], dtype=np.float32)
            validation_returns = np.asarray(validation["returns"], dtype=np.float32)
            validation_rv = np.asarray(validation["rv"], dtype=np.float32)
            validation_dates = tuple(str(value) for value in validation["dates"].tolist())
            validation_security_ids = tuple(
                str(value) for value in validation["security_ids"].tolist()
            )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise V112SealedAccessError("V11.2 development arrays are malformed") from exc
    for values, returns, rv, dates, security_ids, label in (
        (
            train_features,
            train_returns,
            train_rv,
            train_dates,
            train_security_ids,
            "train",
        ),
        (
            validation_features,
            validation_returns,
            validation_rv,
            validation_dates,
            validation_security_ids,
            "validation",
        ),
    ):
        try:
            _validate_arrays(values, returns, rv, dates, security_ids)
        except ValueError as exc:
            raise V112SealedAccessError(f"V11.2 {label} arrays are invalid") from exc
        if values.ndim != 3 or values.shape[1:] != (
            protocol.window_size,
            len(protocol.feature_names),
        ):
            raise V112SealedAccessError(f"V11.2 {label} feature geometry is invalid")
    if manifest.get("train_rows") != len(train_dates) or manifest.get("validation_rows") != len(
        validation_dates
    ):
        raise V112SealedAccessError("V11.2 development row counts do not match the manifest")
    if manifest.get("train_identity_sha256") != security_identity_digest(
        train_security_ids, train_dates
    ) or manifest.get("validation_identity_sha256") != security_identity_digest(
        validation_security_ids, validation_dates
    ):
        raise V112SealedAccessError(
            "V11.2 development security identity does not match the manifest"
        )
    if manifest.get("panel_sha256") is None or manifest.get("split_sha256") is None:
        raise V112SealedAccessError("V11.2 development panel/split identity is missing")
    _require_digest(manifest.get("panel_sha256"), "panel digest")
    _require_digest(manifest.get("split_sha256"), "split digest")
    return V112DevelopmentData(
        train_features=train_features,
        train_returns=train_returns,
        train_rv=train_rv,
        train_dates=train_dates,
        train_security_ids=train_security_ids,
        validation_features=validation_features,
        validation_returns=validation_returns,
        validation_rv=validation_rv,
        validation_dates=validation_dates,
        validation_security_ids=validation_security_ids,
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
    candidate_digest = _require_digest(candidate_digest, "candidate digest")
    sealed_dir = output_dir / "sealed"
    lock_path = sealed_dir / "SEALED_TEST_OPENED.json"
    metadata_path = sealed_dir / "sealed_metadata.json"
    payload_path = sealed_dir / "test_payload.aesgcm"
    if lock_path.exists():
        raise V112SealedAccessError("V11.2 sealed test has already been opened")
    metadata = _read_json_object(metadata_path, "V11.2 sealed metadata")
    if metadata.get("protocol_id") != V11_2_PROTOCOL_ID:
        raise V112SealedAccessError("sealed metadata protocol mismatch")
    if metadata.get("sealed_test_status") != "LOCKED_UNOPENED":
        raise V112SealedAccessError("sealed metadata does not prove an unopened holdout")
    panel_sha = _require_digest(metadata.get("panel_sha256"), "sealed panel digest")
    schema_sha = _require_digest(metadata.get("schema_sha256"), "sealed schema digest")
    split_sha = _require_digest(metadata.get("split_sha256"), "sealed split digest")
    if schema_sha != feature_schema_digest():
        raise V112SealedAccessError("sealed metadata schema digest mismatch")
    nonce_hex = str(metadata.get("nonce_hex", ""))
    if len(nonce_hex) != 24 or any(value not in "0123456789abcdef" for value in nonce_hex):
        raise V112SealedAccessError("sealed metadata nonce is invalid")
    try:
        nonce = bytes.fromhex(nonce_hex)
    except ValueError as exc:
        raise V112SealedAccessError("sealed metadata nonce is invalid") from exc
    try:
        ciphertext = payload_path.read_bytes()
    except OSError as exc:
        raise V112SealedAccessError("sealed ciphertext is unavailable") from exc
    expected = _require_digest(metadata.get("ciphertext_sha256"), "sealed ciphertext digest")
    if hashlib.sha256(ciphertext).hexdigest() != expected:
        raise V112SealedAccessError("sealed ciphertext digest mismatch")
    token = hashlib.sha256(f"{candidate_digest}:{split_sha}".encode()).hexdigest()
    lock = {
        "protocol_id": V11_2_PROTOCOL_ID,
        "candidate_digest": candidate_digest,
        "split_sha256": split_sha,
        "ciphertext_sha256": expected,
        "opened_at": dt.datetime.now(dt.UTC).isoformat(),
        "unseal_token": token,
    }
    _atomic_create(lock_path, json.dumps(lock, indent=2, sort_keys=True).encode("utf-8"))
    associated = json.dumps(
        {
            "protocol_id": V11_2_PROTOCOL_ID,
            "panel_sha256": panel_sha,
            "schema_sha256": schema_sha,
            "split_sha256": split_sha,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    try:
        plaintext = AESGCM(_key_bytes(key_path, create=False)).decrypt(
            nonce, ciphertext, associated
        )
    except (InvalidTag, OSError, ValueError, TypeError) as exc:
        raise V112SealedAccessError("sealed test decryption failed") from exc
    try:
        with np.load(io.BytesIO(plaintext), allow_pickle=False) as payload:
            features = np.asarray(payload["features"], dtype=np.float32)
            returns = np.asarray(payload["returns"], dtype=np.float32)
            rv = np.asarray(payload["rv"], dtype=np.float32)
            dates = tuple(str(value) for value in payload["dates"].tolist())
            security_ids = tuple(str(value) for value in payload["security_ids"].tolist())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise V112SealedAccessError("sealed test payload is malformed") from exc
    protocol = V112Protocol()
    try:
        _validate_arrays(features, returns, rv, dates, security_ids)
    except ValueError as exc:
        raise V112SealedAccessError("sealed test payload arrays are invalid") from exc
    if features.ndim != 3 or features.shape[1:] != (
        protocol.window_size,
        len(protocol.feature_names),
    ):
        raise V112SealedAccessError("sealed test feature geometry is invalid")
    metadata_identity = _require_digest(
        metadata.get("test_identity_sha256"), "sealed test identity digest"
    )
    if metadata_identity != security_identity_digest(security_ids, dates):
        raise V112SealedAccessError("sealed test security identity digest mismatch")
    if metadata.get("test_unique_securities") != len(set(security_ids)):
        raise V112SealedAccessError("sealed test security count does not match metadata")
    del plaintext, ciphertext
    gc.collect()
    return V112SealedTestPayload(
        features=features,
        returns=returns,
        rv=rv,
        dates=dates,
        security_ids=security_ids,
        unseal_token=token,
        split_sha256=split_sha,
    )
