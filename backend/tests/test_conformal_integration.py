import numpy as np

from evaluation.conformal import calibrate_intervals, interval_diagnostics, prediction_intervals
from experiments.runner import ExperimentConfig, run_baseline_experiment


def test_conformal_intervals_cover_a_predictable_series():
    rng = np.random.default_rng(7)
    n = 200
    actual = 100.0 + 0.5 * np.arange(n) + rng.normal(0.0, 0.10, n)
    # A near-perfect predictor leaves only tiny calibration residuals.
    predicted = actual + rng.normal(0.0, 0.05, n)

    calibration = calibrate_intervals(actual, predicted, coverages=(0.9,))
    radii = calibration["radii"]["0.9"]
    assert len(radii) == 1
    assert all(np.isfinite(radius) and radius >= 0 for radius in radii)

    intervals = prediction_intervals(predicted, calibration, coverage=0.9)
    diagnostics = interval_diagnostics(actual.reshape(-1, 1), intervals)
    # Finite-sample conformal ranks guarantee near-nominal in-sample coverage.
    assert diagnostics["empirical_coverage"] >= 0.88
    assert diagnostics["average_width"] > 0


def test_runner_attaches_intervals_to_every_model_except_persistence():
    close = np.arange(1.0, 181.0)
    features = np.column_stack([close, np.ones_like(close)])
    report = run_baseline_experiment(
        features,
        close,
        feature_names=["Close", "Constant"],
        config=ExperimentConfig(
            lookback=5, horizons=(1, 3), folds=2, min_train_size=50, validation_size=20
        ),
    )

    assert "intervals" not in report["models"]["persistence"]
    for model_name, model_report in report["models"].items():
        if model_name == "persistence":
            continue
        intervals = model_report["intervals"]
        assert all(np.isfinite(radius) for radius in intervals["radius"].values())
        assert intervals["empirical_coverage"] >= 0.88
        assert intervals["sample_count"] == 40
