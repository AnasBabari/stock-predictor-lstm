"""Unit tests for SealedDatasetStoreV11."""

import numpy as np
import pytest

from research.volatility_forecasting.sealed_dataset_store_v11 import (
    SealedDatasetStoreV11,
    SealedPartitionAccessError,
)


def test_sealed_dataset_store_development_and_unseal_lifecycle():
    n_samples = 200
    dates = [f"2023-01-{i:02d}" for i in range(1, 101)] * 2
    numeric = np.random.randn(n_samples, 34)
    news = np.random.randn(n_samples, 19)
    rets = np.random.randn(n_samples, 4)
    rv = np.random.rand(n_samples, 4)

    train_idx = np.arange(0, 140)
    val_idx = np.arange(140, 170)
    test_idx = np.arange(170, 200)

    store = SealedDatasetStoreV11(
        dates=dates,
        numeric_features=numeric,
        news_features=news,
        returns_targets=rets,
        rv_targets=rv,
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        split_digest="abc123splitdigest",
    )

    # 1. Development load succeeds
    dev_payload = store.load_development_dataset()
    assert len(dev_payload.train_numeric) == 140
    assert len(dev_payload.val_numeric) == 30

    # 2. Invalid unseal fails
    with pytest.raises(SealedPartitionAccessError):
        store.unseal_test_partition("short_token")

    # 3. Valid unseal succeeds
    cand_digest = "a" * 64
    test_payload = store.unseal_test_partition(cand_digest)
    assert len(test_payload.test_numeric) == 30
    assert len(test_payload.unseal_token) == 64

    # 4. Attempting to re-unseal with another candidate fails (anti-tamper)
    with pytest.raises(SealedPartitionAccessError):
        store.unseal_test_partition("b" * 64)
