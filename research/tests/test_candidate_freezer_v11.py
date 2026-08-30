"""Unit tests for CandidateFreezerV11."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from research.volatility_forecasting.candidate_freezer_v11 import (
    CandidateFreezerV11,
)
from research.volatility_forecasting.chronological_partitions_v11 import (
    ChronologicalPartitionManager,
)
from research.volatility_forecasting.global_multimodal_trainer_v11 import (
    GlobalMultimodalTrainerV11,
)
from research.volatility_forecasting.sealed_dataset_store_v11 import (
    SealedDatasetStoreV11,
)


def test_candidate_freezer_cryptographic_integrity():
    n_samples = 300
    dates = pd.date_range("2021-01-01", periods=n_samples, freq="B").strftime("%Y-%m-%d").tolist()
    rng = np.random.default_rng(42)

    x_num = rng.normal(0.0, 1.0, size=(n_samples, 34))
    x_num[:, [23, 24, 25]] = np.abs(x_num[:, [23, 24, 25]]) * 0.015 + 0.005
    x_news = rng.normal(0.0, 1.0, size=(n_samples, 19))
    shuffled_news = rng.normal(0.0, 1.0, size=(n_samples, 19))
    delayed_news = rng.normal(0.0, 1.0, size=(n_samples, 19))
    y_rets = rng.normal(0.001, 0.02, size=(n_samples, 4))
    y_rv = np.abs(rng.normal(0.0004, 0.0001, size=(n_samples, 4)))

    split = ChronologicalPartitionManager.create_70_15_15_split(
        dates=dates, max_horizon_days=7, embargo_sessions=30
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        store = SealedDatasetStoreV11(
            dates=dates,
            numeric_features=x_num,
            news_features=x_news,
            same_origin_shuffled_news=shuffled_news,
            causal_delayed_news=delayed_news,
            returns_targets=y_rets,
            rv_targets=y_rv,
            train_indices=split.train_indices,
            val_indices=split.val_indices,
            test_indices=split.test_indices,
            split_digest=split.split_digest,
            lock_dir=Path(tmpdir),
        )

        dev_payload = store.load_development_dataset()
        bundle = GlobalMultimodalTrainerV11.develop_and_freeze_bundle(
            dev_payload=dev_payload, max_epochs=2, patience=1, lr=0.005, n_expanding_folds=2
        )

        m_file, s_file, digest1 = CandidateFreezerV11.freeze_and_save_bundle(
            bundle=bundle,
            output_dir=Path(tmpdir),
            git_sha="abc123git",
            panel_sha="panel123sha",
            split_sha="split123sha",
        )

        assert len(digest1) == 64
        assert m_file.exists()
        assert s_file.exists()

        # Altering a parameter changes the digest
        bundle.num_scaler_mean[0, 0] += 0.5
        _, _, digest2 = CandidateFreezerV11.freeze_and_save_bundle(
            bundle=bundle,
            output_dir=Path(tmpdir),
            git_sha="abc123git",
            panel_sha="panel123sha",
            split_sha="split123sha",
        )
        assert digest1 != digest2
