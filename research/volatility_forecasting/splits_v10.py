"""Unique-origin session 70/15/15 partitioning and sealed target store for V10.

Partitions dates by unique forecast origin sessions, guarantees 30-session embargo
at boundaries, purges overlapping multi-step label windows, and isolates sealed test targets.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PartitionAssignment:
    train_sessions: list[str]
    val_sessions: list[str]
    test_sessions: list[str]
    embargo_sessions: int
    max_label_horizon: int
    train_rows: int
    val_rows: int
    test_rows: int
    transfer_rows: int
    purged_rows: int
    assignment_fingerprint_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SealedTargetMetadata:
    row_count: int
    security_count: int
    target_contract_version: str
    date_start: str
    date_end: str
    checksum_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SealedTargetStore:
    """Isolated target store for the single-use sealed test partition."""

    def __init__(self, target_file: Path) -> None:
        self.target_file = Path(target_file)

    @classmethod
    def create_sealed_store(
        cls,
        target_df: pd.DataFrame,
        output_file: Path,
        target_contract_version: str = "future-rv-total-v2",
    ) -> tuple[SealedTargetStore, SealedTargetMetadata]:
        """Create and write immutable sealed target store with checksum metadata."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Sort canonically by (SecurityID, Date)
        sorted_df = target_df.sort_values(by=["SecurityID", "Date"]).copy()
        records = sorted_df.to_dict(orient="records")
        payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
        output_path.write_bytes(payload)

        checksum = hashlib.sha256(payload).hexdigest()
        unique_secs = int(sorted_df["SecurityID"].nunique()) if len(sorted_df) > 0 else 0
        min_date = str(sorted_df["Date"].min()) if len(sorted_df) > 0 else ""
        max_date = str(sorted_df["Date"].max()) if len(sorted_df) > 0 else ""

        meta = SealedTargetMetadata(
            row_count=len(sorted_df),
            security_count=unique_secs,
            target_contract_version=target_contract_version,
            date_start=min_date,
            date_end=max_date,
            checksum_sha256=checksum,
        )
        return cls(output_path), meta

    def load_targets(self, expected_checksum: str | None = None) -> pd.DataFrame:
        """Load sealed targets after verifying content checksum."""
        if not self.target_file.exists():
            raise FileNotFoundError(f"Sealed target store file missing: {self.target_file}")
        content = self.target_file.read_bytes()
        actual_sha = hashlib.sha256(content).hexdigest()
        if expected_checksum and actual_sha != expected_checksum:
            raise ValueError(
                f"Sealed target store checksum mismatch: expected {expected_checksum}, got {actual_sha}"
            )
        records = json.loads(content.decode("utf-8"))
        return pd.DataFrame(records)


class UniqueOriginSplitterV10:
    """Mathematical 70/15/15 origin-session splitter with purge, embargo and permanent IDs."""

    def __init__(
        self,
        train_fraction: float = 0.70,
        val_fraction: float = 0.15,
        test_fraction: float = 0.15,
        embargo_sessions: int = 30,
        max_label_horizon: int = 30,
        required_transfer_security_ids: tuple[str, ...] = (
            "SEC_GOOGL_001",
            "SEC_MSFT_001",
            "SEC_NVDA_001",
            "SEC_PEP_001",
            "SEC_TXN_001",
        ),
    ) -> None:
        if not np.isclose(train_fraction + val_fraction + test_fraction, 1.0):
            raise ValueError("Split fractions must sum to 1.0")
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.embargo_sessions = embargo_sessions
        self.max_label_horizon = max_label_horizon
        self.required_transfer_security_ids = set(required_transfer_security_ids)

    def partition_sessions(
        self, all_dates: list[str] | pd.Index
    ) -> tuple[list[str], list[str], list[str]]:
        """Partition unique sessions applying 30-session embargo and label-window purge."""
        unique_dates = sorted(set(pd.to_datetime(all_dates).strftime("%Y-%m-%d")))
        n = len(unique_dates)
        min_required = 100 + self.embargo_sessions * 2 + self.max_label_horizon * 2
        if n < min_required:
            raise ValueError(
                f"Insufficient unique sessions for 70/15/15 split with embargo: {n} < {min_required}"
            )

        n_train_block = int(np.floor(self.train_fraction * n))
        n_val_block = int(np.floor(self.val_fraction * n))

        # Train block: origins must end before embargo + label horizon
        purge_train = self.embargo_sessions + self.max_label_horizon
        train_end_idx = max(1, n_train_block - purge_train)
        train_dates = unique_dates[:train_end_idx]

        # Validation block starts at n_train_block
        val_start_idx = n_train_block
        val_end_idx = max(val_start_idx + 1, n_train_block + n_val_block - purge_train)
        val_dates = unique_dates[val_start_idx:val_end_idx]

        # Test block starts at n_train_block + n_val_block
        test_start_idx = n_train_block + n_val_block
        test_end_idx = max(test_start_idx + 1, n - self.max_label_horizon)
        test_dates = unique_dates[test_start_idx:test_end_idx]

        return train_dates, val_dates, test_dates

    def build_assignment(
        self,
        df: pd.DataFrame,
        date_column: str = "Date",
        security_id_column: str = "SecurityID",
    ) -> tuple[pd.DataFrame, PartitionAssignment]:
        """Assign partition labels per row, enforce permanent IDs, and compute order-invariant fingerprint."""
        if security_id_column not in df.columns:
            raise ValueError(
                f"Required column '{security_id_column}' missing from DataFrame. Permanent security IDs are mandatory."
            )

        # Reject duplicate (SecurityID, Date) identities
        dups = df.duplicated(subset=[security_id_column, date_column])
        if dups.any():
            dup_count = int(dups.sum())
            raise ValueError(
                f"Duplicate (SecurityID, Date) rows detected in dataset: {dup_count} duplicate rows."
            )

        unique_dates = sorted(df[date_column].astype(str).unique())
        train_dates, val_dates, test_dates = self.partition_sessions(unique_dates)

        train_set = set(train_dates)
        val_set = set(val_dates)
        test_set = set(test_dates)

        df_out = df.copy()
        partitions = []
        for _, row in df_out.iterrows():
            sec = str(row[security_id_column])
            dt = str(row[date_column])[:10]

            if sec in self.required_transfer_security_ids:
                partitions.append("transfer_holdout")
            elif dt in train_set:
                partitions.append("train")
            elif dt in val_set:
                partitions.append("val")
            elif dt in test_set:
                partitions.append("test")
            else:
                partitions.append("label_overlap_purged")

        df_out["Partition"] = partitions

        train_rows = int((df_out["Partition"] == "train").sum())
        val_rows = int((df_out["Partition"] == "val").sum())
        test_rows = int((df_out["Partition"] == "test").sum())
        transfer_rows = int((df_out["Partition"] == "transfer_holdout").sum())
        purged_rows = int((df_out["Partition"] == "label_overlap_purged").sum())

        # Order-invariant deterministic fingerprint: sort canonical records
        sorted_records = sorted(
            f"{r[security_id_column]}_{str(r[date_column])[:10]}_{r['Partition']}"
            for _, r in df_out.iterrows()
        )
        fingerprint = hashlib.sha256("\n".join(sorted_records).encode("utf-8")).hexdigest()

        assignment = PartitionAssignment(
            train_sessions=train_dates,
            val_sessions=val_dates,
            test_sessions=test_dates,
            embargo_sessions=self.embargo_sessions,
            max_label_horizon=self.max_label_horizon,
            train_rows=train_rows,
            val_rows=val_rows,
            test_rows=test_rows,
            transfer_rows=transfer_rows,
            purged_rows=purged_rows,
            assignment_fingerprint_sha256=fingerprint,
        )

        return df_out, assignment
