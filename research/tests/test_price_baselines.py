from types import SimpleNamespace

import numpy as np
import pytest

from research.price_forecasting.baselines import (
    evaluate_predictions,
    fit_ridge_validation,
    training_majority,
)


def test_matched_errors_and_pooled_rmse():
    actual = np.log([[1.1, 1.2], [0.9, 1.0]])
    predicted = np.zeros((2, 2))
    result = evaluate_predictions(actual, predicted, np.array([1, 1]), ["A", "B"])
    assert result["pooled"]["mae_percent"] == pytest.approx(10)
    assert result["pooled"]["rmse_percent"] == pytest.approx(np.sqrt(150))
    assert result["pooled"]["relative_rmse_vs_persistence"] == pytest.approx(1)
    assert result["pooled"]["direction_accuracy"] == 0
    assert result["pooled"]["majority_direction_accuracy"] == 0.5
    assert result["per_ticker"]["B"]["per_horizon"][1]["relative_mae_vs_persistence"] is None


def test_majority_training_ties_and_validation_shift():
    majority = training_majority(np.array([[1, -1, 0], [-1, -1, 0], [0, 1, 0]]))
    np.testing.assert_array_equal(majority, [1, -1, 1])
    result = evaluate_predictions(np.array([[-1.0, 1.0, 0.0]]), np.zeros((1, 3)), majority, ["A"])
    assert result["pooled"]["majority_direction_accuracy"] == 0


def test_ridge_latest_slice_train_only_scaling_and_no_test_use():
    rng = np.random.default_rng(42)
    sequences = rng.normal(size=(20, 4, 3))
    targets = rng.normal(size=(20, 2))
    data = SimpleNamespace(
        sequences=sequences,
        targets=targets,
        split_train=np.arange(12),
        split_validation=np.arange(12, 16),
        ticker_indices=np.zeros(20, dtype=int),
        ticker_names=("A",),
    )  # Deliberately no split_test attribute.
    report = fit_ridge_validation(data)
    np.testing.assert_allclose(report["scaler_mean"], sequences[:12, -1].mean(axis=0))
    sequences[:, :-1] = 1e6
    sequences[16:] = np.nan
    targets[16:] = np.nan
    repeated = fit_ridge_validation(data)
    assert repeated == report
    assert report["input_slice"] == "sequences[:, -1, :]"
    assert not report["test_evaluated"]


def test_nonfinite_metrics_fail():
    with pytest.raises(ValueError):
        evaluate_predictions([[np.nan]], [[0]], [1], ["A"])
