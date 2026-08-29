"""Content-addressed development candidate freeze for StockLSTM V10.

Packages winning candidate configurations, weights, scalers, baseline parameters,
feature schemas, and ledger digests into an immutable, verifiable package prior to
sealed certification.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class FreezeIntegrityError(ValueError):
    """Raised when candidate package fails pre-certification verification."""


@dataclass(frozen=True)
class FrozenHorizonCandidate:
    horizon: int
    family: str
    role: str  # "learned_candidate" | "development_baseline_candidate"
    config: dict[str, Any]
    selected_seed: int
    scaler_parameters: dict[str, Any]
    baseline_parameters: dict[str, Any] | None
    weights_relative_path: str | None
    weights_sha256: str | None


@dataclass(frozen=True)
class FrozenCandidatePackageV10:
    package_id: str
    protocol_id: str
    protocol_sha256: str
    git_sha: str
    feature_schema_sha256: str
    panel_snapshot_sha256: str
    development_ledger_sha256: str
    created_at_utc: str
    horizons: tuple[FrozenHorizonCandidate, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "package_id": self.package_id,
            "protocol_id": self.protocol_id,
            "protocol_sha256": self.protocol_sha256,
            "git_sha": self.git_sha,
            "feature_schema_sha256": self.feature_schema_sha256,
            "panel_snapshot_sha256": self.panel_snapshot_sha256,
            "development_ledger_sha256": self.development_ledger_sha256,
            "created_at_utc": self.created_at_utc,
            "horizons": [asdict(h) for h in self.horizons],
        }

    def canonical_bytes(self) -> bytes:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return payload.encode("utf-8")

    def package_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def save_package_atomic(
        self,
        base_dir: Path,
        weights_map: dict[str, bytes] | None = None,
    ) -> Path:
        """Atomically create and write the candidate package directory."""
        target_dir = Path(base_dir) / self.package_id
        if target_dir.exists():
            raise FreezeIntegrityError(
                f"Candidate package {self.package_id} already exists at {target_dir}. Overwrite strictly forbidden."
            )

        # Write to temporary directory first
        tmp_dir = Path(tempfile.mkdtemp(prefix="pkg_atomic_", dir=base_dir))
        try:
            manifest_path = tmp_dir / "candidate_manifest.json"
            manifest_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

            if weights_map:
                for rel_path, w_bytes in weights_map.items():
                    w_file = tmp_dir / rel_path
                    w_file.parent.mkdir(parents=True, exist_ok=True)
                    w_file.write_bytes(w_bytes)

            # Atomic directory rename
            tmp_dir.rename(target_dir)
            return target_dir
        except Exception:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

    @classmethod
    def from_package_dir(cls, package_dir: Path) -> FrozenCandidatePackageV10:
        manifest_path = Path(package_dir) / "candidate_manifest.json"
        if not manifest_path.exists():
            raise FreezeIntegrityError(f"Candidate manifest missing at {manifest_path}")
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        horizons = tuple(
            FrozenHorizonCandidate(
                horizon=int(h["horizon"]),
                family=str(h["family"]),
                role=str(h["role"]),
                config=dict(h.get("config", {})),
                selected_seed=int(h["selected_seed"]),
                scaler_parameters=dict(h.get("scaler_parameters", {})),
                baseline_parameters=dict(h.get("baseline_parameters", {}))
                if h.get("baseline_parameters")
                else None,
                weights_relative_path=str(h["weights_relative_path"])
                if h.get("weights_relative_path")
                else None,
                weights_sha256=str(h["weights_sha256"]) if h.get("weights_sha256") else None,
            )
            for h in data["horizons"]
        )
        return cls(
            package_id=str(data["package_id"]),
            protocol_id=str(data["protocol_id"]),
            protocol_sha256=str(data["protocol_sha256"]),
            git_sha=str(data["git_sha"]),
            feature_schema_sha256=str(data["feature_schema_sha256"]),
            panel_snapshot_sha256=str(data["panel_snapshot_sha256"]),
            development_ledger_sha256=str(data["development_ledger_sha256"]),
            created_at_utc=str(data["created_at_utc"]),
            horizons=horizons,
        )

    def verify_weights_integrity(self, package_dir: Path) -> None:
        """Verify checksums of all serialized weight files."""
        for h in self.horizons:
            if h.weights_relative_path:
                w_path = Path(package_dir) / h.weights_relative_path
                if not w_path.exists():
                    raise FreezeIntegrityError(
                        f"Weight file missing for horizon {h.horizon}: {w_path}"
                    )
                actual_sha = hashlib.sha256(w_path.read_bytes()).hexdigest()
                if actual_sha != h.weights_sha256:
                    raise FreezeIntegrityError(
                        f"Weight checksum mismatch for horizon {h.horizon}: expected {h.weights_sha256}, got {actual_sha}"
                    )
