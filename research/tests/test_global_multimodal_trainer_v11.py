"""Unit tests for hardened GlobalMultimodalTrainerV11 exercising confirmatory lifecycle and bootstrap gates."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from research.volatility_forecasting.chronological_partitions_v11 import (
    ChronologicalPartitionManager,
)
from research.volatility_forecasting.global_multimodal_trainer_v11 import (
    GlobalMultimodalTrainerV11,
)
from research.volatility_forecasting.sealed_dataset_store_v11 import (
    SealedDatasetStoreV11,
)


def test_global_multimodal_trainer_v11_confirmatory_lifecycle():
    n_samples = 400
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
        lock_dir = Path(tmpdir)
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
            lock_dir=lock_dir,
        )

        dev_payload = store.load_development_dataset()
        bundle = GlobalMultimodalTrainerV11.develop_and_freeze_bundle(
            dev_payload=dev_payload,
            max_epochs=3,
            patience=2,
            lr=0.005,
            n_expanding_folds=3,
        )

        assert bundle.manifest.selected_candidate_family in ["M1_NUMERIC", "M2_MULTIMODAL_NEWS"]
        assert len(bundle.manifest.manifest_sha256) == 64

        cert_result = GlobalMultimodalTrainerV11.evaluate_frozen_bundle_once(
            bundle=bundle,
            sealed_store=store,
        )

        assert cert_result.preselected_family == bundle.manifest.selected_candidate_family
        assert "M2_vs_M1" in cert_result.paired_bootstrap_cis
        assert "M1_vs_M0" in cert_result.paired_bootstrap_cis
        assert cert_result.certification_decision in [
            "SEALED_TEST_PASS_M2_MULTIMODAL",
            "SEALED_TEST_FAIL_M2_MULTIMODAL",
            "SEALED_TEST_PASS_M1_NUMERIC",
            "SEALED_TEST_FAIL_M1_NUMERIC",
        ]
