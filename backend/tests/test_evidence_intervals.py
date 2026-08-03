import numpy as np
import pytest

from evaluation.conformal import calibrate_intervals, interval_diagnostics, prediction_intervals
from evaluation.evidence import (
    benjamini_hochberg,
    moving_block_bootstrap_interval,
    paired_loss_evidence,
    relative_ratio_evidence,
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


def test_relative_ratio_evidence_identical_candidate_and_baseline():
    rng = np.random.default_rng(7)
    actual = 100.0 + np.cumsum(rng.normal(0.0, 1.0, 200))
    noisy = actual + rng.normal(0.0, 0.5, 200)
    result = relative_ratio_evidence(actual, noisy, noisy, resamples=200, block_length=10)
    assert result["metric"] == "mae"
    assert result["ratio"] == pytest.approx(1.0)
    interval = result["confidence_interval"]
    assert interval["lower"] <= 1.0 <= interval["upper"]
    assert result["sample_count"] == 200


def test_relative_ratio_evidence_better_candidate_wins_for_mae_and_rmse():
    rng = np.random.default_rng(11)
    actual = 100.0 + np.cumsum(rng.normal(0.0, 1.0, 300))
    noise = rng.normal(0.0, 0.8, 300)
    baseline = actual + noise
    better = actual + 0.4 * noise
    for metric in ("mae", "rmse"):
        result = relative_ratio_evidence(
            actual, better, baseline, metric=metric, resamples=200, block_length=10
        )
        assert result["ratio"] < 1.0
        assert result["confidence_interval"]["upper"] < 1.0


def test_relative_ratio_evidence_reduces_multi_horizon_inputs():
    rng = np.random.default_rng(13)
    actual = 100.0 + np.cumsum(rng.normal(0.0, 1.0, 150))[:, None] + rng.normal(0.0, 0.2, (150, 3))
    noise = rng.normal(0.0, 0.8, (150, 3))
    baseline = actual + noise
    better = actual + 0.5 * noise
    result = relative_ratio_evidence(
        actual, better, baseline, horizon=3, resamples=100, block_length=10
    )
    assert result["sample_count"] == 150
    assert result["ratio"] < 1.0
    assert result["confidence_interval"]["block_length"] == 10


def test_relative_ratio_evidence_default_block_length_tracks_horizon():
    rng = np.random.default_rng(17)
    actual = 100.0 + np.cumsum(rng.normal(0.0, 1.0, 60))
    noisy = actual + rng.normal(0.0, 0.5, 60)
    result = relative_ratio_evidence(actual, noisy, noisy, horizon=25, resamples=50)
    assert result["confidence_interval"]["block_length"] == 25
    capped = relative_ratio_evidence(actual, noisy, noisy, horizon=1, resamples=50)
    assert capped["confidence_interval"]["block_length"] == 20


def test_relative_ratio_evidence_drops_zero_baseline_resamples():
    # Regression: resamples whose drawn blocks are entirely zero-baseline-error
    # rows must be filtered out instead of producing NaN CI bounds. The first
    # 20 of 25 rows have zero baseline error, so with block length 5 many
    # resamples consist only of zero-error rows and get dropped.
    rng = np.random.default_rng(29)
    n = 25
    actual = 100.0 + np.cumsum(rng.normal(0.0, 1.0, n))
    candidate = actual + rng.normal(0.0, 0.5, n)
    baseline = candidate.copy()
    baseline[:20] = actual[:20]
    result = relative_ratio_evidence(actual, candidate, baseline, resamples=300, block_length=5)
    interval = result["confidence_interval"]
    assert np.isfinite(result["ratio"])
    assert np.isfinite(interval["lower"])
    assert np.isfinite(interval["upper"])
    assert interval["lower"] > 0.0
    assert result["sample_count"] == n


def test_relative_ratio_evidence_fully_zero_baseline_raises():
    rng = np.random.default_rng(31)
    actual = 100.0 + np.cumsum(rng.normal(0.0, 1.0, 30))
    with pytest.raises(ValueError):
        relative_ratio_evidence(actual, actual + 0.5, actual)


def test_relative_ratio_evidence_rejects_invalid_inputs():
    rng = np.random.default_rng(19)
    actual = 100.0 + np.cumsum(rng.normal(0.0, 1.0, 40))
    noisy = actual + rng.normal(0.0, 0.5, 40)
    with pytest.raises(ValueError):
        relative_ratio_evidence(actual, noisy[:-1], noisy)
    contaminated = noisy.copy()
    contaminated[0] = np.nan
    with pytest.raises(ValueError):
        relative_ratio_evidence(actual, contaminated, noisy)
    with pytest.raises(ValueError):
        relative_ratio_evidence(actual, noisy, noisy, metric="smape")
    with pytest.raises(ValueError):
        relative_ratio_evidence(actual, actual, actual)
    with pytest.raises(ValueError):
        relative_ratio_evidence(actual[:1], noisy[:1], noisy[:1])
