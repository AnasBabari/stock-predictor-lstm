"""Unique-origin session 70/15/15 partitioning, expanding folds, and sealed target store for V10.

Partitions dates by unique forecast origin sessions, guarantees 30-session embargo
at boundaries, purges overlapping multi-step label windows, constructs 5 expanding
cross-validation folds, validates panel integrity, and isolates sealed test targets.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEPLOYABLE_FEATURE_COLUMNS_V5: tuple[str, ...] = (
    # Return Structure (13)
    "Return_1D",
    "Return_5D",
    "Return_10D",
    "Return_20D",
    "Overnight_Return",
    "OpenToClose_Return",
    "HL_Range_Log",
    "Downside_Semivar_20",
    "Realized_Skew_20",
    "Realized_Kurt_20",
    "Drawdown_From_Peak",
    "Up_Streak",
    "Down_Streak",
    # Volatility (7)
    "Vol_C2C_5",
    "Vol_C2C_10",
    "Vol_C2C_20",
    "Vol_C2C_60",
    "EWMA_Var",
    "Vol_Of_Vol_20",
    "Vol_Percentile_252",
    # Liquidity (6)
    "Log_Dollar_Volume",
    "Dollar_Volume_Median_20",
    "Volume_Surprise",
    "Amihud_Illiquidity_20",
    "Zero_Return_Fraction_20",
    "Stale_Price_Flag",
)


class PanelValidationError(ValueError):
    """Raised when panel dataset fails semantic or cryptographic integrity checks."""


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
class ExpandingFoldAssignment:
    fold_idx: int
    train_sessions: list[str]
    val_sessions: list[str]
    purged_sessions: list[str]
    train_rows: int
    val_rows: int

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
            raise PanelValidationError(
                f"Sealed target store checksum mismatch: expected {expected_checksum}, got {actual_sha}"
            )
        records = json.loads(content.decode("utf-8"))
        return pd.DataFrame(records)


@dataclass(frozen=True)
class SampleMetadataRecord:
    security_id: str
    origin_session: str
    label_end_session: str
    partition: str
    fold: int
    feature_schema_sha256: str
    target_contract_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StrictPanelLoader:
    """Strict panel dataset validator enforcing feature contracts and integrity invariants."""

    SECURITY_ID_PATTERN = re.compile(r"^SEC_[A-Z0-9]+_[0-9]+$")

    @classmethod
    def load_and_validate(
        cls,
        panel_source: Path | str | pd.DataFrame,
        required_horizons: list[int] = (1, 3, 5, 7, 14, 30),
        expected_features: tuple[str, ...] = DEPLOYABLE_FEATURE_COLUMNS_V5,
    ) -> pd.DataFrame:
        """Load and validate panel DataFrame; fails closed on any schema violation or corruption."""
        if isinstance(panel_source, (str, Path)):
            path = Path(panel_source)
            if not path.exists():
                raise FileNotFoundError(f"Panel source file not found: {path}")
            df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_json(path)
        elif isinstance(panel_source, pd.DataFrame):
            df = panel_source.copy()
        else:
            raise TypeError(f"Unsupported panel source type: {type(panel_source)}")

        # 1. Validate required primary identity columns
        if "Date" not in df.columns:
            raise PanelValidationError("Missing mandatory 'Date' column in panel data.")
        if "SecurityID" not in df.columns:
            raise PanelValidationError("Missing mandatory 'SecurityID' column in panel data.")

        # 2. Validate SecurityID syntax
        for sec in df["SecurityID"].unique():
            if not cls.SECURITY_ID_PATTERN.match(str(sec)):
                raise PanelValidationError(
                    f"SecurityID '{sec}' does not match required format SEC_<TICKER>_<ID>"
                )

        # 3. Check for duplicates
        dups = df.duplicated(subset=["SecurityID", "Date"])
        if dups.any():
            dup_count = int(dups.sum())
            raise PanelValidationError(
                f"Duplicate (SecurityID, Date) rows detected: {dup_count} duplicate rows."
            )

        # 4. Validate targets exist and are strictly positive and finite
        for h in required_horizons:
            col = f"target_h{h}"
            if col not in df.columns:
                raise PanelValidationError(f"Missing required target column '{col}' in panel data.")
            vals = df[col].to_numpy(dtype=float)
            if not np.all(np.isfinite(vals)):
                raise PanelValidationError(f"Non-finite values detected in target column '{col}'.")
            if np.any(vals <= 0.0):
                raise PanelValidationError(
                    f"Non-positive values detected in target column '{col}'. Volatility targets must be strictly positive."
                )

        # 5. Validate features exist and are strictly finite
        for feat in expected_features:
            if feat not in df.columns:
                raise PanelValidationError(
                    f"Missing required feature column '{feat}' in panel data."
                )
            vals = df[feat].to_numpy(dtype=float)
            if not np.all(np.isfinite(vals)):
                raise PanelValidationError(
                    f"Non-finite values detected in feature column '{feat}'."
                )

        # 6. Sort canonically
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
        df_sorted = df.sort_values(by=["SecurityID", "Date"]).reset_index(drop=True)
        return df_sorted


class ExpandingFoldSplitterV10:
    """Constructs 5 expanding temporal folds over development sessions with purge and embargo."""

    def __init__(
        self,
        n_folds: int = 5,
        embargo_sessions: int = 30,
        max_label_horizon: int = 30,
        min_train_sessions: int = 40,
    ) -> None:
        self.n_folds = n_folds
        self.embargo_sessions = embargo_sessions
        self.max_label_horizon = max_label_horizon
        self.min_train_sessions = min_train_sessions

    def split_sessions(self, dev_sessions: list[str]) -> list[ExpandingFoldAssignment]:
        """Generate expanding fold session assignments."""
        unique_dates = sorted(set(dev_sessions))
        total_sessions = len(unique_dates)

        purge_len = self.embargo_sessions + self.max_label_horizon
        min_required = self.min_train_sessions + self.n_folds * 10 + purge_len
        if total_sessions < min_required:
            raise ValueError(
                f"Insufficient development sessions for {self.n_folds} expanding folds: {total_sessions} < {min_required}"
            )

        # Reserve validation block length per fold
        avail_for_val = total_sessions - self.min_train_sessions - purge_len
        val_block_size = max(5, avail_for_val // self.n_folds)

        folds: list[ExpandingFoldAssignment] = []
        for k in range(self.n_folds):
            val_end_idx = total_sessions - (self.n_folds - 1 - k) * val_block_size
            val_start_idx = max(self.min_train_sessions + purge_len, val_end_idx - val_block_size)

            train_end_idx = val_start_idx - purge_len
            train_dates = unique_dates[:train_end_idx]
            val_dates = unique_dates[val_start_idx:val_end_idx]
            purged_dates = unique_dates[train_end_idx:val_start_idx]

            folds.append(
                ExpandingFoldAssignment(
                    fold_idx=k,
                    train_sessions=train_dates,
                    val_sessions=val_dates,
                    purged_sessions=purged_dates,
                    train_rows=0,
                    val_rows=0,
                )
            )

        return folds


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
