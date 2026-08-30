"""Physically sealed dataset store enforcing persistent on-disk one-shot test partition access."""

from __future__ import annotations

import datetime
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class SealedPartitionAccessError(RuntimeError):
    """Raised when an unauthorized access or multiple unseal of the sealed test partition is attempted."""

    __test__ = False


@dataclass(frozen=True)
class DevelopmentDatasetPayload:
    train_numeric: np.ndarray
    train_news: np.ndarray
    train_returns: np.ndarray
    train_rv: np.ndarray
    train_dates: list[str]
    val_numeric: np.ndarray
    val_news: np.ndarray
    val_returns: np.ndarray
    val_rv: np.ndarray
    val_dates: list[str]
    split_digest: str


@dataclass(frozen=True)
class SealedTestPayload:
    test_numeric: np.ndarray
    test_news: np.ndarray
    test_returns: np.ndarray
    test_rv: np.ndarray
    test_dates: list[str]
    unseal_token: str
    split_digest: str


class SealedDatasetStoreV11:
    """Manages physical, on-disk sealed storage of 70/15/15 datasets, strictly blocking test access."""

    def __init__(
        self,
        dates: list[str],
        numeric_features: np.ndarray,
        news_features: np.ndarray,
        returns_targets: np.ndarray,
        rv_targets: np.ndarray,
        train_indices: np.ndarray,
        val_indices: np.ndarray,
        test_indices: np.ndarray,
        split_digest: str,
        lock_dir: Path | None = None,
    ) -> None:
        self._dates = dates
        self._numeric = numeric_features
        self._news = news_features
        self._returns = returns_targets
        self._rv = rv_targets

        self._train_idx = train_indices
        self._val_idx = val_indices
        self._test_idx = test_indices
        self._split_digest = split_digest

        self._lock_dir = lock_dir or (
            Path(__file__).resolve().parents[2] / "artifacts" / "sealed_test_locks"
        )
        self._lock_dir.mkdir(parents=True, exist_ok=True)
        self._lock_file = self._lock_dir / f"SEALED_TEST_OPENED_{split_digest[:16]}.json"

    def load_development_dataset(self) -> DevelopmentDatasetPayload:
        """Returns train and validation partitions. Test set remains strictly inaccessible."""
        return DevelopmentDatasetPayload(
            train_numeric=self._numeric[self._train_idx],
            train_news=self._news[self._train_idx],
            train_returns=self._returns[self._train_idx],
            train_rv=self._rv[self._train_idx],
            train_dates=[self._dates[i] for i in self._train_idx],
            val_numeric=self._numeric[self._val_idx],
            val_news=self._news[self._val_idx],
            val_returns=self._returns[self._val_idx],
            val_rv=self._rv[self._val_idx],
            val_dates=[self._dates[i] for i in self._val_idx],
            split_digest=self._split_digest,
        )

    def unseal_test_partition(self, candidate_freeze_digest: str) -> SealedTestPayload:
        """Unseals the sacred 15% test partition once and only once using the frozen candidate digest.

        Enforces persistent on-disk locking across process restarts.
        """
        if len(candidate_freeze_digest) < 32:
            raise SealedPartitionAccessError(
                "Invalid candidate freeze digest: must be a valid 32+ character SHA digest."
            )

        # Check persistent on-disk lock marker
        if self._lock_file.exists():
            try:
                locked_meta = json.loads(self._lock_file.read_text(encoding="utf-8"))
            except Exception:
                locked_meta = {}
            raise SealedPartitionAccessError(
                f"Test partition already permanently unsealed on disk! Lock file: {self._lock_file}, "
                f"originally unsealed for candidate: {locked_meta.get('candidate_digest')}, "
                f"unsealed_at: {locked_meta.get('unsealed_at')}."
            )

        unseal_token = hashlib.sha256(
            f"{candidate_freeze_digest}:{self._split_digest}".encode()
        ).hexdigest()

        # Write immutable on-disk lock marker
        lock_data = {
            "candidate_digest": candidate_freeze_digest,
            "split_digest": self._split_digest,
            "unsealed_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "unseal_token": unseal_token,
            "test_sample_count": len(self._test_idx),
        }
        self._lock_file.write_text(json.dumps(lock_data, indent=2), encoding="utf-8")

        return SealedTestPayload(
            test_numeric=self._numeric[self._test_idx],
            test_news=self._news[self._test_idx],
            test_returns=self._returns[self._test_idx],
            test_rv=self._rv[self._test_idx],
            test_dates=[self._dates[i] for i in self._test_idx],
            unseal_token=unseal_token,
            split_digest=self._split_digest,
        )
