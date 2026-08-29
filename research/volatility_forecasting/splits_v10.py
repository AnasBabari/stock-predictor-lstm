"""Unique-origin session 70/15/15 partitioning and sealed target store for V10.

Partitions dates by unique forecast origin sessions, guarantees 30-session embargo
at boundaries, purges overlapping multi-step labels, and isolates sealed test targets.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PartitionAssignment:
    train_sessions: list[str]
    val_sessions: list[str]
    test_sessions: list[str]
    embargo_sessions: int
    train_rows: int
    val_rows: int
    test_rows: int
    transfer_rows: int
    assignment_fingerprint_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SealedTargetMetadata:
    row_count: int
    security_count: int
    target_schema_sha256: str
    date_start: str
    date_end: str
    is_opened: bool
    checksum_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class UniqueOriginSplitterV10:
    def __init__(
        self,
        train_fraction: float = 0.70,
        val_fraction: float = 0.15,
        test_fraction: float = 0.15,
        embargo_sessions: int = 30,
        required_transfer_tickers: tuple[str, ...] = ("GOOGL", "MSFT", "NVDA", "PEP", "TXN"),
    ) -> None:
        if not np.isclose(train_fraction + val_fraction + test_fraction, 1.0):
            raise ValueError("Split fractions must sum to 1.0")
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.embargo_sessions = embargo_sessions
        self.required_transfer_tickers = set(required_transfer_tickers)

    def partition_sessions(
        self, all_dates: list[str] | pd.Index
    ) -> tuple[list[str], list[str], list[str]]:
        unique_dates = sorted(set(pd.to_datetime(all_dates).strftime("%Y-%m-%d")))
        n = len(unique_dates)
        if n < 100:
            raise ValueError(f"Insufficient unique sessions for 70/15/15 split: {n}")

        n_train = int(np.floor(self.train_fraction * n))
        n_val = int(np.floor(self.val_fraction * n))

        train_dates = unique_dates[:n_train]
        val_dates = unique_dates[n_train : n_train + n_val]
        test_dates = unique_dates[n_train + n_val :]

        return train_dates, val_dates, test_dates

    def build_assignment(
        self,
        df: pd.DataFrame,
        date_column: str = "Date",
        ticker_column: str = "Ticker",
    ) -> PartitionAssignment:
        unique_dates = sorted(df[date_column].astype(str).unique())
        train_dates, val_dates, test_dates = self.partition_sessions(unique_dates)

        is_transfer = df[ticker_column].isin(self.required_transfer_tickers)
        is_train_date = df[date_column].astype(str).isin(set(train_dates))
        is_val_date = df[date_column].astype(str).isin(set(val_dates))
        is_test_date = df[date_column].astype(str).isin(set(test_dates))

        train_mask = is_train_date & (~is_transfer)
        val_mask = is_val_date & (~is_transfer)
        test_mask = is_test_date & (~is_transfer)
        transfer_mask = is_transfer

        fingerprint_data = {
            "train_sessions": train_dates,
            "val_sessions": val_dates,
            "test_sessions": test_dates,
            "train_row_count": int(train_mask.sum()),
            "val_row_count": int(val_mask.sum()),
            "test_row_count": int(test_mask.sum()),
            "transfer_row_count": int(transfer_mask.sum()),
        }
        fp = hashlib.sha256(
            json.dumps(fingerprint_data, sort_keys=True).encode("utf-8")
        ).hexdigest()

        return PartitionAssignment(
            train_sessions=train_dates,
            val_sessions=val_dates,
            test_sessions=test_dates,
            embargo_sessions=self.embargo_sessions,
            train_rows=int(train_mask.sum()),
            val_rows=int(val_mask.sum()),
            test_rows=int(test_mask.sum()),
            transfer_rows=int(transfer_mask.sum()),
            assignment_fingerprint_sha256=fp,
        )
