import numpy as np
import pandas as pd
import pytest

from experiments.contracts import FoldPlan, build_experiment_dataset


def test_experiment_dataset_and_fold_plan_share_raw_origins_without_target_overlap():
    close = np.arange(1.0, 301.0)
    dataset = build_experiment_dataset(
        np.column_stack([close, close * 2]),
        close,
        dates=pd.date_range("2024-01-01", periods=len(close), freq="B"),
        feature_names=["Close", "Double"],
        lookback=5,
        horizons=(1, 5, 20),
        snapshot_id="snapshot-one",
    )
    plan = FoldPlan.create(dataset, folds=2, min_train_size=100, validation_size=40, gap=20)
    assert dataset.snapshot_id == "snapshot-one"
    assert plan.gap == 20
    for fold in plan.folds:
        assert (
            dataset.origin_indices[fold.training_indices[-1]] + 20
            < dataset.origin_indices[fold.validation_indices[0]]
        )


def test_fold_plan_refuses_gap_smaller_than_maximum_horizon():
    close = np.arange(1.0, 201.0)
    dataset = build_experiment_dataset(
        close[:, None],
        close,
        dates=pd.date_range("2024-01-01", periods=len(close), freq="B"),
        feature_names=["Close"],
        lookback=5,
        horizons=(1, 20),
    )
    with pytest.raises(ValueError, match="gap"):
        FoldPlan.create(dataset, folds=1, min_train_size=100, validation_size=20, gap=19)
