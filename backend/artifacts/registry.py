"""Local evidence registry with atomic promotion and rollback.

The registry is deliberately independent from TensorFlow.  Keras artifacts and
small deterministic baseline engines can therefore use the same lifecycle.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

Channel = Literal["candidate", "eligible", "current", "previous", "rejected"]
_IDENTITY = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}")


class ArtifactRegistryError(RuntimeError):
    """A registry transition or manifest validation failed."""


def _safe_identity(value: str, label: str) -> str:
    if not isinstance(value, str) or not _IDENTITY.fullmatch(value):
        raise ArtifactRegistryError(f"Invalid {label} identity.")
    return value


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PromotionManifest:
    """Portable evidence required before a candidate can become current."""

    ticker: str
    engine: str
    version: str
    benchmark_id: str
    snapshot_id: str
    promoted: bool
    evidence: dict[str, Any]
    files: dict[str, str]
    created_at: str
    schema_version: int = 1
    signature: str | None = None

    @classmethod
    def create(
        cls,
        *,
        ticker: str,
        engine: str,
        version: str,
        benchmark_id: str,
        snapshot_id: str,
        promoted: bool,
        evidence: dict[str, Any],
        files: dict[str, str],
    ) -> PromotionManifest:
        return cls(
            ticker=_safe_identity(ticker, "ticker"),
            engine=_safe_identity(engine, "engine"),
            version=_safe_identity(version, "version"),
            benchmark_id=_safe_identity(benchmark_id, "benchmark"),
            snapshot_id=_safe_identity(snapshot_id, "snapshot"),
            promoted=bool(promoted),
            evidence=evidence,
            files=dict(sorted(files.items())),
            created_at=datetime.now(UTC).isoformat(),
        )

    def unsigned_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("signature")
        return payload

    def with_signature(self, signature: str) -> PromotionManifest:
        if not signature:
            raise ArtifactRegistryError("Manifest signature must not be empty.")
        return PromotionManifest(**(self.unsigned_payload() | {"signature": signature}))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PromotionManifest:
        try:
            manifest = cls(**payload)
        except (TypeError, ValueError) as exc:
            raise ArtifactRegistryError("Promotion manifest is malformed.") from exc
        _safe_identity(manifest.ticker, "ticker")
        _safe_identity(manifest.engine, "engine")
        _safe_identity(manifest.version, "version")
        _safe_identity(manifest.benchmark_id, "benchmark")
        _safe_identity(manifest.snapshot_id, "snapshot")
        if manifest.schema_version != 1:
            raise ArtifactRegistryError("Unsupported promotion manifest schema.")
        if not isinstance(manifest.evidence, dict) or not isinstance(manifest.files, dict):
            raise ArtifactRegistryError("Promotion evidence or file hashes are malformed.")
        return manifest


class LocalArtifactRegistry:
    """Filesystem implementation with atomic channel pointers."""

    def __init__(
        self,
        root: str | Path,
        *,
        require_signature: bool = False,
        verify_signature: Callable[[bytes, str], bool] | None = None,
    ):
        self.root = Path(root)
        self.require_signature = require_signature
        self.verify_signature = verify_signature

    def _engine_root(self, ticker: str, engine: str) -> Path:
        return self.root / _safe_identity(ticker, "ticker") / _safe_identity(engine, "engine")

    def _pointer(self, ticker: str, engine: str, channel: Channel) -> Path:
        return self._engine_root(ticker, engine) / f"{channel}.json"

    @staticmethod
    def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}-{uuid.uuid4().hex}.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)

    def _version_dir(self, manifest: PromotionManifest) -> Path:
        return (
            self._engine_root(manifest.ticker, manifest.engine)
            / "versions"
            / _safe_identity(manifest.version, "version")
        )

    def _validate_signature(self, manifest: PromotionManifest) -> None:
        if manifest.signature is None:
            if self.require_signature:
                raise ArtifactRegistryError("A signed promotion manifest is required.")
            return
        if self.verify_signature is None:
            if self.require_signature:
                raise ArtifactRegistryError("No promotion signature verifier is configured.")
            return
        if not self.verify_signature(
            _canonical_json(manifest.unsigned_payload()), manifest.signature
        ):
            raise ArtifactRegistryError("Promotion manifest signature is invalid.")

    def _validate_files(self, manifest: PromotionManifest, directory: Path) -> None:
        if not manifest.files:
            raise ArtifactRegistryError("Promotion manifest contains no artifact files.")
        for name, expected in manifest.files.items():
            if Path(name).name != name or not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise ArtifactRegistryError("Promotion file manifest is invalid.")
            path = directory / name
            if not path.is_file() or sha256_file(path) != expected:
                raise ArtifactRegistryError(f"Artifact integrity check failed for {name}.")

    def stage(
        self,
        source: str | Path,
        manifest: PromotionManifest,
        *,
        sign: Callable[[bytes], str] | None = None,
    ) -> PromotionManifest:
        """Copy a complete candidate into its immutable version directory."""

        source_path = Path(source)
        if not source_path.is_dir():
            raise ArtifactRegistryError("Candidate source directory does not exist.")
        if sign is not None:
            manifest = manifest.with_signature(sign(_canonical_json(manifest.unsigned_payload())))
        self._validate_signature(manifest)
        self._validate_files(manifest, source_path)
        destination = self._version_dir(manifest)
        if destination.exists():
            raise ArtifactRegistryError("Artifact version already exists.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}-{uuid.uuid4().hex}.tmp")
        shutil.copytree(source_path, temporary)
        (temporary / "promotion.json").write_text(
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, destination)
        self._atomic_json(
            self._pointer(manifest.ticker, manifest.engine, "candidate"),
            {"version": manifest.version},
        )
        return manifest

    def read_manifest(self, ticker: str, engine: str, version: str) -> PromotionManifest:
        path = self._engine_root(ticker, engine) / "versions" / version / "promotion.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactRegistryError("Promotion manifest is missing or corrupt.") from exc
        manifest = PromotionManifest.from_dict(payload)
        self._validate_signature(manifest)
        self._validate_files(manifest, path.parent)
        return manifest

    def resolve(self, ticker: str, engine: str, channel: Channel = "current") -> Path | None:
        pointer = self._pointer(ticker, engine, channel)
        if not pointer.exists():
            return None
        try:
            version = json.loads(pointer.read_text(encoding="utf-8"))["version"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ArtifactRegistryError(f"{channel} pointer is corrupt.") from exc
        _safe_identity(version, "version")
        directory = self._engine_root(ticker, engine) / "versions" / version
        if not directory.is_dir():
            raise ArtifactRegistryError(f"{channel} artifact version is incomplete.")
        self.read_manifest(ticker, engine, version)
        return directory

    def promote(
        self,
        ticker: str,
        engine: str,
        version: str,
        *,
        probe: Callable[[Path], None] | None = None,
    ) -> Path:
        """Promote an evidence-approved candidate and roll back a failed probe."""

        manifest = self.read_manifest(ticker, engine, version)
        if not manifest.promoted:
            self._atomic_json(
                self._pointer(ticker, engine, "rejected"),
                {"version": version, "reasons": manifest.evidence.get("reasons", [])},
            )
            raise ArtifactRegistryError("Candidate evidence rejected promotion.")

        destination = self._version_dir(manifest)
        current_pointer = self._pointer(ticker, engine, "current")
        previous_payload: dict[str, Any] | None = None
        if current_pointer.exists():
            try:
                previous_payload = json.loads(current_pointer.read_text(encoding="utf-8"))
                _safe_identity(previous_payload["version"], "version")
            except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ArtifactRegistryError("Current pointer is corrupt.") from exc

        self._atomic_json(
            self._pointer(ticker, engine, "eligible"),
            {"version": version, "benchmark_id": manifest.benchmark_id},
        )
        if previous_payload is not None:
            self._atomic_json(self._pointer(ticker, engine, "previous"), previous_payload)
        self._atomic_json(current_pointer, {"version": version})

        try:
            if probe is not None:
                probe(destination)
        except Exception as exc:
            if previous_payload is None:
                current_pointer.unlink(missing_ok=True)
            else:
                self._atomic_json(current_pointer, previous_payload)
            raise ArtifactRegistryError(
                "Post-activation probe failed; promotion rolled back."
            ) from exc
        return destination

    def rollback(self, ticker: str, engine: str) -> Path:
        previous = self.resolve(ticker, engine, "previous")
        if previous is None:
            raise ArtifactRegistryError("No previous artifact is available.")
        version = previous.name
        self._atomic_json(self._pointer(ticker, engine, "current"), {"version": version})
        return previous
