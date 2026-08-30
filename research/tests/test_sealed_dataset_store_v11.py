"""Unit tests for persistent SealedDatasetStoreV11."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from research.volatility_forecasting.sealed_dataset_store_v11 import (
    SealedDatasetStoreV11,
    SealedPartitionAccessError,
)


def test_sealed_dataset_store_persistent_disk_locking():
    n_samples = 200
    dates = [f"2023-01-{i:02d}" for i in range(1, 101)] * 2
    numeric = np.random.randn(n_samples, 34)
    news = np.random.randn(n_samples, 19)
    rets = np.random.randn(n_samples, 4)
    rv = np.random.rand(n_samples, 4)

    train_idx = np.arange(0, 140)
    val_idx = np.arange(140, 170)
    test_idx = np.arange(170, 200)

    with tempfile.TemporaryDirectory() as tmpdir:
        lock_path = Path(tmpdir)
        store = SealedDatasetStoreV11(
            dates=dates,
            numeric_features=numeric,
            news_features=news,
            returns_targets=rets,
            rv_targets=rv,
            train_indices=train_idx,
            val_indices=val_idx,
            test_indices=test_idx,
            split_digest="split123testdigest",
            lock_dir=lock_path,
        )

        # 1. Development load succeeds
        dev_payload = store.load_development_dataset()
        assert len(dev_payload.train_numeric) == 140
        assert len(dev_payload.val_numeric) == 30

        # 2. First unseal succeeds and creates disk marker
        cand_digest = "a" * 64
        test_payload = store.unseal_test_partition(cand_digest)
        assert len(test_payload.test_numeric) == 30

        # 3. Second unseal on same store object fails
        with pytest.raises(SealedPartitionAccessError):
            store.unseal_test_partition(cand_digest)

        # 4. Constructing a completely new store object in another Python instance also fails (persistent disk lock)
        new_store = SealedDatasetStoreV11(
            dates=dates,
            numeric_features=numeric,
            news_features=news,
            returns_targets=rets,
            rv_targets=rv,
            train_indices=train_idx,
            val_indices=val_idx,
            test_indices=test_idx,
            split_digest="split123testdigest",
            lock_dir=lock_path,
        )
        with pytest.raises(SealedPartitionAccessError):
            new_store.unseal_test_partition(cand_digest)
