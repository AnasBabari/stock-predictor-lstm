import numpy as np

from evaluation.conformal import calibrate_intervals, interval_diagnostics, prediction_intervals
from evaluation.evidence import (
    benjamini_hochberg,
    moving_block_bootstrap_interval,
    paired_loss_evidence,
)
from evaluation.metrics import evaluate_probability_forecast, regression_metrics


def test_regression_metrics_include_robust_and_symmetric_percentage_errors():
    report = regression_metrics([100, 110], [90, 120])
    assert report["median_absolute_error"] == 10.0
    assert report["smape"] > 0


def test_bootstrap_and_paired_loss_show_candidate_improvement():
    values = np.linspace(1.0, 2.0, 30)
    interval = moving_block_bootstrap_interval(values, resamples=100, block_length=5)
    assert interval["lower"] <= interval["estimate"] <= interval["upper"]
    evidence = paired_loss_evidence(
        values, values - 0.1, values - 1.0, resamples=100, block_length=5
    )
    assert evidence["mean_improvement"] > 0


def test_bh_control_and_conformal_intervals_are_horizon_aware():
    assert benjamini_hochberg([0.001, 0.02, 0.9], q=0.1) == [True, True, False]
    actual = np.array([[101.0, 103.0], [99.0, 101.0], [102.0, 104.0]])
    prediction = np.array([[100.0, 100.0], [100.0, 100.0], [100.0, 100.0]])
    calibration = calibrate_intervals(actual, prediction, coverages=(0.8,))
    intervals = prediction_intervals(prediction, calibration, coverage=0.8)
    diagnostics = interval_diagnostics(actual, intervals)
    assert intervals["lower"].shape == actual.shape
    assert diagnostics["average_width"] > 0


def test_probability_report_contains_reliability_information():
    report = evaluate_probability_forecast([0, 1], [0.1, 0.9], training_targets=[0, 1])
    assert report["expected_calibration_error"] >= 0
    assert sum(item["count"] for item in report["reliability_bins"]) == 2
