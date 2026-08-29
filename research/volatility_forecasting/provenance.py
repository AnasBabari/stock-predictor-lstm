"""Cryptographically bound immutable run manifests and provenance validation.

Every research training, evaluation, certification, and export operation must bind
to an exact, verified ImmutableRunManifest. Stale ledgers, checkpoints, or model
weights cannot be silently reused without an exact cryptographic provenance match.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class ProvenanceMismatchError(ValueError):
    """Raised when cached evidence or model weights do not match the expected run provenance."""


@dataclass(frozen=True)
class ImmutableRunManifest:
    run_id: str
    artifact_role: str
    git_sha: str
    protocol_id: str
    protocol_sha256: str
    universe_snapshot_id: str
    universe_sha256: str
    panel_snapshot_id: str
    panel_sha256: str
    split_manifest_sha256: str
    feature_schema_sha256: str
    news_snapshot_sha256: str | None
    dependency_lock_sha256: str
    candidate_registry_sha256: str
    hardware: dict[str, Any]
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_bytes(self) -> bytes:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return payload.encode("utf-8")

    def manifest_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def save(self, path: Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImmutableRunManifest:
        expected_keys = {
            "run_id",
            "artifact_role",
            "git_sha",
            "protocol_id",
            "protocol_sha256",
            "universe_snapshot_id",
            "universe_sha256",
            "panel_snapshot_id",
            "panel_sha256",
            "split_manifest_sha256",
            "feature_schema_sha256",
            "news_snapshot_sha256",
            "dependency_lock_sha256",
            "candidate_registry_sha256",
            "hardware",
            "created_at",
        }
        missing = expected_keys - set(data.keys())
        if missing:
            raise ProvenanceMismatchError(f"Manifest missing required fields: {sorted(missing)}")
        return cls(
            run_id=str(data["run_id"]),
            artifact_role=str(data["artifact_role"]),
            git_sha=str(data["git_sha"]),
            protocol_id=str(data["protocol_id"]),
            protocol_sha256=str(data["protocol_sha256"]),
            universe_snapshot_id=str(data["universe_snapshot_id"]),
            universe_sha256=str(data["universe_sha256"]),
            panel_snapshot_id=str(data["panel_snapshot_id"]),
            panel_sha256=str(data["panel_sha256"]),
            split_manifest_sha256=str(data["split_manifest_sha256"]),
            feature_schema_sha256=str(data["feature_schema_sha256"]),
            news_snapshot_sha256=str(data["news_snapshot_sha256"])
            if data.get("news_snapshot_sha256") is not None
            else None,
            dependency_lock_sha256=str(data["dependency_lock_sha256"]),
            candidate_registry_sha256=str(data["candidate_registry_sha256"]),
            hardware=dict(data.get("hardware", {})),
            created_at=str(data["created_at"]),
        )

    @classmethod
    def from_file(cls, path: Path) -> ImmutableRunManifest:
        target = Path(path)
        if not target.exists():
            raise ProvenanceMismatchError(f"Manifest does not exist at {target}")
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ProvenanceMismatchError(f"Malformed manifest at {target}: {exc}") from exc
        return cls.from_dict(data)

    def verify_matching(self, other: ImmutableRunManifest | dict[str, Any]) -> None:
        target_dict = other.to_dict() if isinstance(other, ImmutableRunManifest) else other
        self_dict = self.to_dict()
        critical_fields = [
            "git_sha",
            "protocol_id",
            "protocol_sha256",
            "universe_snapshot_id",
            "universe_sha256",
            "panel_snapshot_id",
            "panel_sha256",
            "split_manifest_sha256",
            "feature_schema_sha256",
            "news_snapshot_sha256",
            "dependency_lock_sha256",
            "candidate_registry_sha256",
        ]
        mismatches = []
        for field in critical_fields:
            v_self = self_dict.get(field)
            v_other = target_dict.get(field)
            if v_self != v_other:
                mismatches.append(f"{field}: expected {v_self!r}, got {v_other!r}")
        if mismatches:
            raise ProvenanceMismatchError(
                f"Provenance mismatch detected across {len(mismatches)} fields:\n"
                + "\n".join(f"  - {m}" for m in mismatches)
            )


def capture_runtime_hardware_info(device_name: str | None = None) -> dict[str, Any]:
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
    }
    if device_name:
        info["device"] = device_name
    return info


def compute_canonical_json_sha256(data: Any) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compute_ledger_evidence_sha256(records: list[dict[str, Any]]) -> str:
    def record_sort_key(r: dict[str, Any]) -> tuple:
        return (
            int(r.get("fold", 0)),
            str(r.get("family", "")),
            int(r.get("seed", 0)),
            int(r.get("horizon", 0)),
        )

    sorted_records = sorted(records, key=record_sort_key)
    return compute_canonical_json_sha256(sorted_records)


def compute_sha256(path: Path | str) -> str:
    """Compute SHA-256 digest of a file on disk."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found for sha256 computation: {p}")
    return hashlib.sha256(p.read_bytes()).hexdigest()


def generate_run_id(prefix: str = "run_v10") -> str:
    """Generate deterministic timestamped run ID."""
    import secrets
    from datetime import UTC, datetime

    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    rand_suffix = secrets.token_hex(3)
    return f"{prefix}_{ts}_{rand_suffix}"
