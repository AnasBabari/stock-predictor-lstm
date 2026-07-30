import numpy as np
import pytest

from evaluation.metrics import (
    evaluate_forecast_horizons,
    evaluate_probability_forecast,
    regression_metrics,
)
from evaluation.promotion import PromotionPolicy, assess_promotion
from evaluation.splits import generate_walk_forward_splits, purged_tail_split


def test_direction_accuracy_is_anchored_to_each_forecast_origin():
    origins = np.array([100.0, 200.0])
    actual = np.array([[101.0, 99.0], [198.0, 203.0]])
    predicted = np.array([[102.0, 98.0], [199.0, 204.0]])

    report = evaluate_forecast_horizons(actual, predicted, origins, horizons=[1, 5])

    assert report["per_horizon"]["1"]["direction_accuracy"] == 1.0
    assert report["per_horizon"]["5"]["direction_accuracy"] == 1.0
    assert report["pooled"]["direction_accuracy"] == 1.0


def test_horizon_report_uses_persistence_from_the_same_origins():
    origins = np.array([100.0, 200.0])
    actual = np.array([[102.0], [198.0]])
    predicted = np.array([[101.0], [199.0]])

    metrics = evaluate_forecast_horizons(
        actual,
        predicted,
        origins,
        horizons=[1],
        scale_series=np.array([95.0, 96.0, 98.0, 100.0]),
    )["per_horizon"]["1"]

    assert metrics["mae"] == 1.0
    assert metrics["rmse"] == 1.0
    assert metrics["relative_mae"] == 0.5
    assert metrics["relative_rmse"] == 0.5
    assert metrics["mase"] is not None
    assert metrics["rmsse"] is not None


def test_regression_metrics_reject_misaligned_origins():
    with pytest.raises(ValueError, match="Origin values"):
        regression_metrics([1.0, 2.0], [1.0, 2.0], origin=[1.0])


def test_probability_report_includes_calibration_metrics_and_majority_baseline():
    report = evaluate_probability_forecast(
        [1, 0, 1, 0],
        [0.9, 0.1, 0.8, 0.2],
        training_targets=[1, 1, 1, 0],
    )
    assert report["accuracy"] == 1.0
    assert report["balanced_accuracy"] == 1.0
    assert report["majority_baseline"] == 0.5
    assert report["brier_score"] < 0.05
    assert report["log_loss"] > 0


def test_walk_forward_splits_preserve_gap_and_method():
    expanding = generate_walk_forward_splits(
        240, folds=3, min_train_size=100, validation_size=30, gap=5
    )
    rolling = generate_walk_forward_splits(
        240,
        folds=3,
        min_train_size=100,
        validation_size=30,
        gap=5,
        method="rolling",
    )
    assert [(x[0][0], x[0][-1], x[1][0], x[1][-1]) for x in expanding] == [
        (0, 144, 150, 179),
        (0, 174, 180, 209),
        (0, 204, 210, 239),
    ]
    assert [(x[0][0], x[0][-1], x[1][0], x[1][-1]) for x in rolling] == [
        (45, 144, 150, 179),
        (75, 174, 180, 209),
        (105, 204, 210, 239),
    ]


def test_purged_tail_split_removes_overlapping_target_boundary():
    fitting, validation = purged_tail_split(100, validation_fraction=0.1, purge=4)
    assert fitting[-1] == 85
    assert validation[0] == 90
    assert validation[0] - fitting[-1] - 1 == 4


def test_promotion_gate_requires_persistence_improvement_and_fold_stability():
    policy = PromotionPolicy(minimum_winning_folds=2)
    pooled = {"relative_mae": 0.8, "relative_rmse": 0.85, "mase": 0.8, "rmsse": 0.9}
    folds = [
        {"relative_mae": 0.8, "relative_rmse": 0.9},
        {"relative_mae": 0.9, "relative_rmse": 0.95},
    ]
    assert assess_promotion(pooled, folds, policy=policy).promoted

    rejected = assess_promotion(
        {**pooled, "relative_rmse": 1.1},
        folds,
        policy=policy,
    )
    assert not rejected.promoted
    assert any("relative RMSE" in reason for reason in rejected.reasons)
