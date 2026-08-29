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
from datetime import UTC, datetime
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
        target_dir = Path(base_dir)
        if target_dir.is_dir() and target_dir.name != self.package_id:
            target_dir = target_dir / self.package_id

        if target_dir.exists():
            raise FreezeIntegrityError(
                f"Candidate package target already exists at {target_dir}. Overwrite strictly forbidden."
            )

        # Write to temporary directory first
        parent_dir = target_dir.parent
        parent_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = Path(tempfile.mkdtemp(prefix="pkg_atomic_", dir=parent_dir))
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


def freeze_candidate_package(
    target_dir: Path,
    candidate_name: str,
    protocol_version: str,
    horizons: list[int],
    configuration: dict[str, Any],
    scalers_by_horizon: dict[int, Any],
    baseline_params_by_horizon: dict[int, Any],
    weights_by_horizon: dict[int, bytes],
    development_ledger_sha256: str,
    git_sha: str = "0" * 40,
    feature_schema_sha256: str = "0" * 64,
    panel_snapshot_sha256: str = "0" * 64,
) -> tuple[Path, FrozenCandidatePackageV10]:
    """Assemble and atomically save a candidate package directory."""
    weights_map: dict[str, bytes] = {}
    frozen_horizons = []

    for h in horizons:
        h_sel = configuration.get(str(h), configuration.get(h, {}))
        fam = h_sel.get("selected_family", "har")
        role = h_sel.get("selected_role", "development_baseline_candidate")
        seed = h_sel.get("selected_seed", 41)
        w_bytes = weights_by_horizon.get(h)

        if w_bytes:
            rel_w = f"weights_h{h}_{fam}.bin"
            w_sha = hashlib.sha256(w_bytes).hexdigest()
            weights_map[rel_w] = w_bytes
        else:
            rel_w = None
            w_sha = None

        cand = FrozenHorizonCandidate(
            horizon=h,
            family=fam,
            role=role,
            config=h_sel,
            selected_seed=seed,
            scaler_parameters=scalers_by_horizon.get(h, {}),
            baseline_parameters=baseline_params_by_horizon.get(h, {}),
            weights_relative_path=rel_w,
            weights_sha256=w_sha,
        )
        frozen_horizons.append(cand)

    pkg = FrozenCandidatePackageV10(
        package_id=candidate_name,
        protocol_id=protocol_version,
        protocol_sha256="0" * 64,
        git_sha=git_sha,
        feature_schema_sha256=feature_schema_sha256,
        panel_snapshot_sha256=panel_snapshot_sha256,
        development_ledger_sha256=development_ledger_sha256,
        created_at_utc=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        horizons=tuple(frozen_horizons),
    )

    out_dir = pkg.save_package_atomic(target_dir, weights_map=weights_map)
    return out_dir, pkg
