"""Unit tests for hardened GlobalMultimodalTrainerV11 exercising full candidate bundle and negative controls."""

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


def test_global_multimodal_trainer_v11_bundle_and_controls():
    n_samples = 400
    dates = pd.date_range("2021-01-01", periods=n_samples, freq="B").strftime("%Y-%m-%d").tolist()

    rng = np.random.default_rng(42)
    # Exact 34 numeric + 19 news features (53 total)
    x_num = rng.normal(0.0, 1.0, size=(n_samples, 34))
    # Make har components (features 23, 24, 25) positive daily volatility
    x_num[:, [23, 24, 25]] = np.abs(x_num[:, [23, 24, 25]]) * 0.015 + 0.005
    x_news = rng.normal(0.0, 1.0, size=(n_samples, 19))

    # Exact 4 required target horizons (1, 3, 5, 7)
    y_rets = rng.normal(0.001, 0.02, size=(n_samples, 4))
    y_rv = np.abs(rng.normal(0.0004, 0.0001, size=(n_samples, 4)))

    # Chronological partition with 7-session purge
    split = ChronologicalPartitionManager.create_70_15_15_split(
        dates=dates, max_horizon_days=7, embargo_sessions=30
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        lock_dir = Path(tmpdir)
        store = SealedDatasetStoreV11(
            dates=dates,
            numeric_features=x_num,
            news_features=x_news,
            returns_targets=y_rets,
            rv_targets=y_rv,
            train_indices=split.train_indices,
            val_indices=split.val_indices,
            test_indices=split.test_indices,
            split_digest=split.split_digest,
            lock_dir=lock_dir,
        )

        # 1. Development, Expanding Folds, and Bundle Freeze
        dev_payload = store.load_development_dataset()
        assert len(dev_payload.train_numeric) > 100
        assert len(dev_payload.val_numeric) > 20

        bundle = GlobalMultimodalTrainerV11.develop_and_freeze_bundle(
            dev_payload=dev_payload,
            max_epochs=4,
            lr=0.005,
            n_expanding_folds=3,
        )

        assert bundle.manifest.winning_model_family in ["M1_NUMERIC", "M2_MULTIMODAL_NEWS"]
        assert "M0_HAR_BASELINE" in bundle.manifest.validation_oof_metrics
        assert "M1_NUMERIC" in bundle.manifest.validation_oof_metrics
        assert "M2_MULTIMODAL_NEWS" in bundle.manifest.validation_oof_metrics

        # 2. Sacred One-Shot Sealed Test Evaluation
        cert_result = GlobalMultimodalTrainerV11.evaluate_frozen_bundle_once(
            bundle=bundle,
            sealed_store=store,
        )

        assert "M0_HAR_BASELINE" in cert_result.sealed_test_metrics
        assert "M1_NUMERIC" in cert_result.sealed_test_metrics
        assert "M2_MULTIMODAL_NEWS" in cert_result.sealed_test_metrics
        assert "M3_SHUFFLE_CONTROL" in cert_result.sealed_test_metrics
        assert "M3_DELAY_CONTROL" in cert_result.sealed_test_metrics
        assert len(cert_result.audit_trail["unseal_token"]) == 64
        assert cert_result.certification_decision in [
            "CERTIFIED_M2_PROMOTED",
            "CERTIFIED_M1_NUMERIC_CHAMPION",
            "CERTIFIED_INFERIOR",
        ]
