from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from stock_autoresearch.candidates import PersistenceCandidate
from stock_autoresearch.config import EvaluationPolicy
from stock_autoresearch.data import Snapshot, build_examples, expanding_folds
from stock_autoresearch.evaluation import evaluate_candidate


def make_snapshot(rows: int = 700) -> Snapshot:
    index = pd.date_range("2020-01-01", periods=rows, freq="B")
    close = 100.0 * np.exp(np.cumsum(np.full(rows, 0.001)))
    frame = pd.DataFrame(
        {"Close": close, "feature_a": np.linspace(0.0, 1.0, rows)},
        index=index,
    )
    return Snapshot(frame=frame, snapshot_id="test-snapshot", feature_names=("feature_a",))


def test_targets_are_cumulative_returns_from_origin() -> None:
    snapshot = make_snapshot()
    features, targets, origins = build_examples(snapshot, window=60, horizon=5)
    assert features.shape[1:] == (60, 1)
    assert origins[0] == 60
    np.testing.assert_allclose(targets[0], 0.005, atol=1e-12)


def test_folds_have_a_purge_gap() -> None:
    folds = list(
        expanding_folds(
            635,
            folds=5,
            minimum_train_rows=300,
            validation_rows=60,
            purge=4,
        )
    )
    assert len(folds) == 5
    for train, validation in folds:
        assert train[-1] < validation[0] - 4


def test_persistence_is_zero_return_and_complete() -> None:
    snapshot = make_snapshot()
    policy = EvaluationPolicy()
    result = evaluate_candidate(
        snapshot,
        lambda seed: PersistenceCandidate(),
        horizon=5,
        policy=policy,
    )
    assert result.complete
    assert len(result.folds) == policy.folds
    assert result.summary(policy)["median_relative_mae"] == pytest.approx(1.0)
