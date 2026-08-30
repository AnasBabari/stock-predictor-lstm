"""Strict chronological 70/15/15 partitioning with label purging and 30-session embargo discipline."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ChronologicalPartitionSplit:
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    train_dates: tuple[str, str]
    val_dates: tuple[str, str]
    test_dates: tuple[str, str]
    purged_train_count: int
    purged_val_count: int
    embargo_sessions: int
    split_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_dates": self.train_dates,
            "val_dates": self.val_dates,
            "test_dates": self.test_dates,
            "train_size": len(self.train_indices),
            "val_size": len(self.val_indices),
            "test_size": len(self.test_indices),
            "purged_train_count": self.purged_train_count,
            "purged_val_count": self.purged_val_count,
            "embargo_sessions": self.embargo_sessions,
            "split_digest": self.split_digest,
        }


class ChronologicalPartitionManager:
    """Manages 70/15/15 temporal splitting, boundary label purging, and expanding CV folds."""

    @staticmethod
    def create_70_15_15_split(
        dates: list[str],
        max_horizon_days: int = 7,
        embargo_sessions: int = 30,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
    ) -> ChronologicalPartitionSplit:
        n = len(dates)
        if n < 100:
            raise ValueError(f"Insufficient historical sessions for 70/15/15 split: {n} < 100")

        # 1. Monotonic chronological sort
        unique_dates = sorted(set(dates))
        n_dates = len(unique_dates)

        # 2. Raw boundary indices
        train_end_idx = int(n_dates * train_ratio)
        val_end_idx = int(n_dates * (train_ratio + val_ratio))

        # 3. Apply target purge and frozen embargo at Train -> Val boundary
        # Purge training rows whose target horizon crosses into validation
        effective_train_end = max(0, train_end_idx - max_horizon_days)
        val_start_idx = min(n_dates - 1, train_end_idx + embargo_sessions)

        # Apply target purge and frozen embargo at Val -> Test boundary
        effective_val_end = max(val_start_idx, val_end_idx - max_horizon_days)
        test_start_idx = min(n_dates - 1, val_end_idx + embargo_sessions)

        train_date_set = set(unique_dates[:effective_train_end])
        val_date_set = set(unique_dates[val_start_idx:effective_val_end])
        test_date_set = set(unique_dates[test_start_idx:])

        # Map back to original sample indices
        train_idx = np.array([i for i, d in enumerate(dates) if d in train_date_set], dtype=int)
        val_idx = np.array([i for i, d in enumerate(dates) if d in val_date_set], dtype=int)
        test_idx = np.array([i for i, d in enumerate(dates) if d in test_date_set], dtype=int)

        purged_train = train_end_idx - effective_train_end
        purged_val = val_end_idx - effective_val_end

        # Cryptographic digest of assignment
        raw_sig = (
            f"{len(train_idx)}:{len(val_idx)}:{len(test_idx)}:{unique_dates[0]}:{unique_dates[-1]}"
        )
        split_hash = hashlib.sha256(raw_sig.encode()).hexdigest()

        return ChronologicalPartitionSplit(
            train_indices=train_idx,
            val_indices=val_idx,
            test_indices=test_idx,
            train_dates=(unique_dates[0], unique_dates[effective_train_end - 1]),
            val_dates=(unique_dates[val_start_idx], unique_dates[effective_val_end - 1]),
            test_dates=(unique_dates[test_start_idx], unique_dates[-1]),
            purged_train_count=purged_train,
            purged_val_count=purged_val,
            embargo_sessions=embargo_sessions,
            split_digest=split_hash,
        )

    @staticmethod
    def create_expanding_folds(
        train_val_dates: list[str],
        n_folds: int = 5,
        max_horizon_days: int = 7,
        embargo_sessions: int = 15,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate expanding cross-validation folds within the first 85% development data."""
        unique_dates = sorted(set(train_val_dates))
        n_dates = len(unique_dates)
        min_train_sessions = int(n_dates * 0.40)
        remaining = n_dates - min_train_sessions
        fold_step = remaining // n_folds

        folds: list[tuple[np.ndarray, np.ndarray]] = []
        for f in range(n_folds):
            train_cut = min_train_sessions + f * fold_step
            val_cut = train_cut + fold_step if f < n_folds - 1 else n_dates

            eff_train_cut = max(0, train_cut - max_horizon_days)
            val_start = min(n_dates - 1, train_cut + embargo_sessions)

            if val_start >= val_cut:
                continue

            train_d = set(unique_dates[:eff_train_cut])
            val_d = set(unique_dates[val_start:val_cut])

            t_idx = np.array([i for i, d in enumerate(train_val_dates) if d in train_d], dtype=int)
            v_idx = np.array([i for i, d in enumerate(train_val_dates) if d in val_d], dtype=int)

            if len(t_idx) > 0 and len(v_idx) > 0:
                folds.append((t_idx, v_idx))

        return folds
