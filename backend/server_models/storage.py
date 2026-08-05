"""Object storage for immutable server forecast bundles.

Bundles live at ``{prefix}/{version_id}/bundle.json`` in a private bucket and
are never overwritten once written.  No presigned URLs are generated; callers
use explicit credentials configured per role (trainer write-only, API read-only).
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Any

BUNDLE_FILENAME = "bundle.json"


def _error_code(exc: Exception) -> str:
    """Best-effort S3/boto error code (``NoSuchKey``, ``PreconditionFailed``...)."""
    code = getattr(exc, "response", None)
    if isinstance(code, dict):
        code = code.get("Error") or {}
        if isinstance(code, dict):
            return str(code.get("Code", ""))
    return exc.__class__.__name__


class ObjectStoreError(RuntimeError):
    """An object store operation failed."""


class ObjectStore(ABC):
    """Minimal put/get/exists contract shared by all backends."""

    @abstractmethod
    def put(self, key: str, data: bytes) -> None: ...

    @abstractmethod
    def get(self, key: str) -> bytes: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...


class InMemoryObjectStore(ObjectStore):
    """Dict-backed store for unit tests and local dry runs."""

    def __init__(self, *, prefix: str = "artifacts") -> None:
        self._objects: dict[str, bytes] = {}
        self._prefix = prefix.strip("/")
        self._lock = threading.RLock()

    def put(self, key: str, data: bytes) -> None:
        if not key:
            raise ObjectStoreError("Object key must not be empty.")
        with self._lock:
            self._objects[key] = bytes(data)

    def get(self, key: str) -> bytes:
        with self._lock:
            try:
                return self._objects[key]
            except KeyError:
                raise ObjectStoreError(f"Object not found: {key}") from None

    def exists(self, key: str) -> bool:
        with self._lock:
            return key in self._objects

    def bundle_key(self, version_id: str) -> str:
        if not version_id:
            raise ObjectStoreError("Artifact version_id must not be empty.")
        return f"{self._prefix}/{version_id}/{BUNDLE_FILENAME}"

    def put_bundle(self, version_id: str, bundle_json_bytes: bytes) -> str:
        key = self.bundle_key(version_id)
        with self._lock:
            if key in self._objects:
                raise ObjectStoreError(f"Bundle is immutable and already exists: {key}")
            self._objects[key] = bytes(bundle_json_bytes)
        return key

    def get_bundle(self, version_id: str) -> bytes:
        return self.get(self.bundle_key(version_id))

    def bundle_exists(self, version_id: str) -> bool:
        return self.exists(self.bundle_key(version_id))


class S3ObjectStore(ObjectStore):
    """S3-compatible store (R2/S3 in production, MinIO/localstack in dev/CI)."""

    def __init__(
        self,
        bucket: str,
        *,
        prefix: str = "artifacts",
        endpoint_url: str | None = None,
        client: Any | None = None,
    ) -> None:
        if not bucket:
            raise ObjectStoreError("S3 bucket name must not be empty.")
        self._bucket = bucket
        self._prefix = prefix.strip("/")
        if client is not None:
            self._client = client
        else:
            import boto3  # type: ignore

            self._client = boto3.client("s3", endpoint_url=endpoint_url)

    def bundle_key(self, version_id: str) -> str:
        if not version_id:
            raise ObjectStoreError("Artifact version_id must not be empty.")
        return f"{self._prefix}/{version_id}/{BUNDLE_FILENAME}"

    def put(self, key: str, data: bytes) -> None:
        self._client.put_object(Bucket=self._bucket, Key=key, Body=bytes(data))

    def get(self, key: str) -> bytes:
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=key)
        except Exception as exc:
            raise ObjectStoreError(f"Object not found: {key}") from exc
        return response["Body"].read()

    def exists(self, key: str) -> bool:
        # Only a genuine 404/NoSuchKey means "absent". Network, authentication,
        # or bucket errors must propagate so readiness logic never mistakes an
        # infrastructure failure for a missing artifact.
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
        except Exception as exc:
            code = _error_code(exc)
            if code in ("404", "NoSuchKey", "NotFound"):
                return False
            raise ObjectStoreError(f"Object store check failed for {key}") from exc
        return True

    def put_bundle(self, version_id: str, bundle_json_bytes: bytes) -> str:
        key = self.bundle_key(version_id)
        # Conditional write: the object is created only if no object exists at
        # the key yet, closing the check-then-write race between concurrent
        # trainers that a separate exists() call cannot close.
        try:
            self._client.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=bytes(bundle_json_bytes),
                IfNoneMatch="*",
            )
        except Exception as exc:
            if _error_code(exc) in ("PreconditionFailed", "412"):
                raise ObjectStoreError(f"Bundle is immutable and already exists: {key}") from exc
            raise ObjectStoreError(f"Bundle write failed: {key}") from exc
        return key

    def ensure_bucket(self) -> None:
        """Create the configured bucket when missing (MinIO/CI/localstack)."""
        try:
            self._client.create_bucket(Bucket=self._bucket)
        except Exception as exc:
            code = _error_code(exc)
            if code not in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
                raise ObjectStoreError(
                    f"Bucket {self._bucket} could not be created: {code}"
                ) from exc

    def get_bundle(self, version_id: str) -> bytes:
        return self.get(self.bundle_key(version_id))

    def bundle_exists(self, version_id: str) -> bool:
        return self.exists(self.bundle_key(version_id))
