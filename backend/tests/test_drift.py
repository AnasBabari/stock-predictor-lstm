import numpy as np
import pytest

from evaluation.drift import feature_divergence, population_stability_index, residual_drift


def test_psi_is_zero_for_identical_distributions():
    generator = np.random.default_rng(7)
    values = generator.normal(size=400)
    psi = population_stability_index(values, values)
    assert psi == pytest.approx(0.0, abs=1e-6)
    same_distribution = generator.normal(size=400)
    assert population_stability_index(values, same_distribution) < 0.05


def test_psi_is_large_for_shifted_distribution():
    generator = np.random.default_rng(11)
    reference = generator.normal(0.0, 1.0, size=500)
    shifted = generator.normal(3.0, 1.0, size=500)
    psi = population_stability_index(reference, shifted)
    assert np.isfinite(psi)
    assert psi > 0.5


def test_psi_handles_constant_reference():
    """A degenerate reference has undefined PSI — None, never a fake 0.0."""
    constant_reference = np.full(50, 2.5)
    assert population_stability_index(constant_reference, np.full(30, 2.5)) is None
    assert population_stability_index(constant_reference, np.arange(30.0)) is None


def test_feature_divergence_reports_degenerate_columns():
    rng = np.random.default_rng(0)
    train = rng.normal(size=(40, 1, 2))
    validation = rng.normal(size=(30, 1, 2))
    # Second feature constant in training => degenerate PSI for that column.
    train[:, :, 1] = 7.0
    result = feature_divergence(train, validation)
    assert result["degenerate_columns"] == [1]
    assert result["max_psi"] == pytest.approx(result["psi_by_column"][0])
    fully_degenerate = np.full((10, 1, 1), 3.0)
    empty = feature_divergence(fully_degenerate, np.full((8, 1, 1), 4.0))
    assert empty["max_psi"] is None and empty["mean_psi"] is None


def test_psi_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        population_stability_index([], [1.0, 2.0])
    with pytest.raises(ValueError):
        population_stability_index([1.0], [])
    with pytest.raises(ValueError):
        population_stability_index([1.0, 2.0], [1.0, np.nan])
    with pytest.raises(ValueError):
        population_stability_index([[1.0, 2.0]], [1.0, 2.0])
    with pytest.raises(ValueError):
        population_stability_index([1.0, 2.0, 3.0], [1.0, 2.0], bins=1)


def test_feature_divergence_reports_one_psi_per_feature_column():
    generator = np.random.default_rng(13)
    train_windows = generator.normal(size=(40, 5, 3))
    validation_windows = generator.normal(size=(20, 5, 3))
    validation_windows[:, :, 2] += 2.0
    report = feature_divergence(train_windows, validation_windows, bins=5)
    assert len(report["psi_by_column"]) == 3
    assert report["max_psi"] == pytest.approx(max(report["psi_by_column"]))
    assert report["mean_psi"] == pytest.approx(float(np.mean(report["psi_by_column"])))
    assert report["psi_by_column"][2] > report["psi_by_column"][0]


def test_feature_divergence_accepts_two_dimensional_input_and_validates_shapes():
    generator = np.random.default_rng(17)
    train_rows = generator.normal(size=(60, 2))
    validation_rows = generator.normal(size=(30, 2))
    report = feature_divergence(train_rows, validation_rows, bins=4)
    assert len(report["psi_by_column"]) == 2
    with pytest.raises(ValueError):
        feature_divergence(generator.normal(size=(10, 5, 3)), generator.normal(size=(8, 5, 2)))
    with pytest.raises(ValueError):
        feature_divergence(generator.normal(size=(10, 3)), np.full((5, 3), np.inf))


def test_residual_drift_flagged_only_when_late_shift_present():
    generator = np.random.default_rng(19)
    stationary = np.abs(np.random.default_rng(3).normal(size=400))
    stationary_report = residual_drift(stationary, resamples=150)
    assert stationary_report["drift_detected"] is False
    assert stationary_report["first_half_ci"]["lower"] <= stationary_report["first_half_mae"]

    drifted = np.abs(generator.normal(0.0, 1.0, size=120))
    drifted[60:] += 3.0
    drifted_report = residual_drift(drifted, resamples=150)
    assert drifted_report["drift_detected"] is True
    assert drifted_report["difference"] == pytest.approx(
        drifted_report["second_half_mae"] - drifted_report["first_half_mae"]
    )

    variance_shift = np.abs(generator.normal(0.0, 1.0, size=120))
    variance_shift[60:] *= 6.0
    assert residual_drift(variance_shift, resamples=150)["drift_detected"] is True


def test_residual_drift_rejects_tiny_or_invalid_input():
    with pytest.raises(ValueError):
        residual_drift(np.arange(9.0))
    with pytest.raises(ValueError):
        residual_drift(np.full(12, np.nan))
    with pytest.raises(ValueError):
        residual_drift(np.arange(20.0), confidence=1.5)
