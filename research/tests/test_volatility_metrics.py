from __future__ import annotations

import numpy as np
from volatility_forecasting.metrics import (
    DistributionPredictions,
    gaussian_crps,
    horizon_distribution_metrics,
    qlike_losses,
)


def test_qlike_is_zero_for_perfect_positive_variance_forecast() -> None:
    realized = np.array([0.1, 0.2, 0.3])
    np.testing.assert_allclose(qlike_losses(realized, realized), 0.0, atol=1e-12)


def test_gaussian_crps_is_lower_when_location_is_closer() -> None:
    observation = np.array([0.1])
    variance = np.array([0.04])
    close = gaussian_crps(observation, np.array([0.1]), variance)
    far = gaussian_crps(observation, np.array([1.0]), variance)
    assert close[0] < far[0]


def test_horizon_metrics_report_proper_scores_calibration_and_direction() -> None:
    realized_var = np.array([[0.01, 0.02], [0.02, 0.04], [0.03, 0.06]])
    baseline_var = realized_var * 2.0
    returns = np.array([[-0.1, -0.2], [0.0, 0.0], [0.1, 0.2]])
    classes = np.array([[0, 0], [1, 1], [2, 2]])
    probabilities = np.eye(3)[classes]
    predictions = DistributionPredictions(
        variance=realized_var,
        return_location=returns,
        direction_probabilities=probabilities,
    )
    metrics = horizon_distribution_metrics(
        predictions=predictions,
        baseline_variance=baseline_var,
        realized_variance=realized_var,
        cumulative_returns=returns,
        direction_classes=classes,
        horizons=(1, 7),
    )

    assert len(metrics) == 2
    assert metrics[0]["relative_qlike"] == 0.0
    assert metrics[0]["direction_accuracy"] == 1.0
    assert metrics[0]["direction_balanced_accuracy"] == 1.0
    assert metrics[0]["direction_brier"] == 0.0
    assert metrics[0]["relative_return_mae"] == 0.0
    assert "relative_variance_only_gaussian_crps" in metrics[0]
    assert "variance_only_coverage_80" in metrics[0]
    assert 0.0 <= metrics[0]["coverage_80"] <= 1.0


def test_distribution_prediction_contract_rejects_bad_probabilities() -> None:
    try:
        DistributionPredictions(
            variance=np.ones((2, 1)),
            return_location=np.zeros((2, 1)),
            direction_probabilities=np.ones((2, 1, 3)),
        )
    except ValueError as error:
        assert "sum to one" in str(error)
    else:
        raise AssertionError("bad probability rows should fail")
