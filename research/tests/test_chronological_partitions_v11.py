"""Unit tests for ChronologicalPartitionManager."""

import pandas as pd

from research.volatility_forecasting.chronological_partitions_v11 import (
    ChronologicalPartitionManager,
)


def test_chronological_70_15_15_split_and_purge():
    dates = pd.date_range("2020-01-01", periods=1000, freq="B").strftime("%Y-%m-%d").tolist()
    split = ChronologicalPartitionManager.create_70_15_15_split(
        dates=dates,
        max_horizon_days=7,
        embargo_sessions=30,
    )

    # 1. Verify train dates precede val dates
    assert split.train_dates[1] < split.val_dates[0]
    # 2. Verify val dates precede test dates
    assert split.val_dates[1] < split.test_dates[0]

    # 3. Verify zero index overlap between all three sets
    t_set = set(split.train_indices)
    v_set = set(split.val_indices)
    s_set = set(split.test_indices)

    assert len(t_set.intersection(v_set)) == 0
    assert len(v_set.intersection(s_set)) == 0
    assert len(t_set.intersection(s_set)) == 0

    # 4. Verify expanding folds generated
    train_val_dates = [dates[i] for i in sorted(list(t_set.union(v_set)))]
    folds = ChronologicalPartitionManager.create_expanding_folds(train_val_dates, n_folds=4)
    assert len(folds) >= 3
    for t_idx, v_idx in folds:
        assert len(set(t_idx).intersection(set(v_idx))) == 0
