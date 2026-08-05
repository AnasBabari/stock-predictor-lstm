"""Object storage for immutable server forecast bundles.

Bundles live at ``{prefix}/{version_id}/bundle.json`` in a private bucket and
are never overwritten once written.  No presigned URLs are generated; callers
use explicit credentials configured per role (trainer write-only, API read-only).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

BUNDLE_FILENAME = "bundle.json"


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

    def put(self, key: str, data: bytes) -> None:
        if not key:
            raise ObjectStoreError("Object key must not be empty.")
        self._objects[key] = bytes(data)

    def get(self, key: str) -> bytes:
        try:
            return self._objects[key]
        except KeyError:
            raise ObjectStoreError(f"Object not found: {key}") from None

    def exists(self, key: str) -> bool:
        return key in self._objects

    def bundle_key(self, version_id: str) -> str:
        if not version_id:
            raise ObjectStoreError("Artifact version_id must not be empty.")
        return f"{self._prefix}/{version_id}/{BUNDLE_FILENAME}"

    def put_bundle(self, version_id: str, bundle_json_bytes: bytes) -> str:
        key = self.bundle_key(version_id)
        if self.exists(key):
            raise ObjectStoreError(f"Bundle is immutable and already exists: {key}")
        self.put(key, bundle_json_bytes)
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
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
        except Exception:
            return False
        return True

    def put_bundle(self, version_id: str, bundle_json_bytes: bytes) -> str:
        key = self.bundle_key(version_id)
        if self.exists(key):
            raise ObjectStoreError(f"Bundle is immutable and already exists: {key}")
        self.put(key, bundle_json_bytes)
        return key

    def get_bundle(self, version_id: str) -> bytes:
        return self.get(self.bundle_key(version_id))

    def bundle_exists(self, version_id: str) -> bool:
        return self.exists(self.bundle_key(version_id))
