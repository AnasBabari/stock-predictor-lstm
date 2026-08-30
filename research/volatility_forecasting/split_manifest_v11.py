"""Canonical split manifest builder recording line-by-line security/origin partition assignments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from research.volatility_forecasting.chronological_partitions_v11 import (
    ChronologicalPartitionSplit,
)


@dataclass(frozen=True)
class CanonicalSplitManifest:
    nominal_split: str
    train_dates: tuple[str, str]
    val_dates: tuple[str, str]
    test_dates: tuple[str, str]
    total_rows: int
    train_rows: int
    val_rows: int
    test_rows: int
    purged_train_rows: int
    purged_val_rows: int
    embargo_sessions: int
    effective_train_pct: float
    effective_val_pct: float
    effective_test_pct: float
    split_sha256: str
    partition_assignments_digest: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SplitManifestBuilderV11:
    """Constructs line-by-line canonical split manifests with immutable SHA-256 fingerprinting."""

    @classmethod
    def build_and_save_manifest(
        cls,
        dates: list[str],
        security_ids: list[str],
        split: ChronologicalPartitionSplit,
        target_path: Path,
    ) -> CanonicalSplitManifest:
        n = len(dates)
        if len(security_ids) != n:
            raise ValueError(
                f"Mismatched dates ({n}) and security_ids ({len(security_ids)}) counts."
            )

        train_set = set(split.train_indices)
        val_set = set(split.val_indices)
        test_set = set(split.test_indices)

        assignment_lines: list[str] = []
        for i in range(n):
            sec_id = security_ids[i]
            d = dates[i]
            if i in train_set:
                part = "TRAIN"
            elif i in val_set:
                part = "VALIDATION"
            elif i in test_set:
                part = "SEALED_TEST"
            else:
                part = "PURGED_OR_EMBARGO"
            assignment_lines.append(f"{sec_id}|{d}|{part}")

        raw_canonical = "\n".join(assignment_lines)
        assignment_sha = hashlib.sha256(raw_canonical.encode("utf-8")).hexdigest()

        eff_t = round(len(split.train_indices) / n * 100.0, 2)
        eff_v = round(len(split.val_indices) / n * 100.0, 2)
        eff_s = round(len(split.test_indices) / n * 100.0, 2)

        manifest = CanonicalSplitManifest(
            nominal_split="70/15/15",
            train_dates=split.train_dates,
            val_dates=split.val_dates,
            test_dates=split.test_dates,
            total_rows=n,
            train_rows=len(split.train_indices),
            val_rows=len(split.val_indices),
            test_rows=len(split.test_indices),
            purged_train_rows=split.purged_train_count,
            purged_val_rows=split.purged_val_count,
            embargo_sessions=split.embargo_sessions,
            effective_train_pct=eff_t,
            effective_val_pct=eff_v,
            effective_test_pct=eff_s,
            split_sha256=split.split_digest,
            partition_assignments_digest=assignment_sha,
        )

        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
        return manifest
