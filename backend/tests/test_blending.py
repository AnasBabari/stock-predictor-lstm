import numpy as np
import pytest

from evaluation.blending import fit_constrained_blend, fit_shrinkage_alpha


def test_shrinkage_alpha_stays_within_unit_interval():
    rng = np.random.default_rng(0)
    actual = rng.normal(0.0, 0.01, size=120)
    model = actual * 0.7 + rng.normal(0.0, 0.01, size=120)
    alpha = fit_shrinkage_alpha(model, actual)
    assert 0.0 <= alpha <= 1.0


def test_shrinkage_alpha_selects_one_for_perfect_model():
    actual = np.linspace(-0.02, 0.03, 50)
    assert fit_shrinkage_alpha(actual.copy(), actual) == 1.0


def test_shrinkage_alpha_near_zero_for_pure_noise_model():
    rng = np.random.default_rng(1)
    actual = rng.normal(0.0, 0.01, size=400)
    noise = rng.normal(0.0, 0.01, size=400)
    assert fit_shrinkage_alpha(noise, actual) <= 0.1


def test_shrinkage_alpha_respects_custom_grid():
    actual = np.linspace(-0.01, 0.02, 40)
    alpha = fit_shrinkage_alpha(actual, actual, grid=(0.25, 0.5, 0.75))
    assert alpha == 0.75


def test_constrained_blend_weights_are_non_negative_and_bounded():
    rng = np.random.default_rng(2)
    actual = rng.normal(0.0, 0.01, size=100)
    members = np.column_stack([actual * 2.0, actual * 0.5 + rng.normal(0.0, 0.005, 100), -actual])
    weights = fit_constrained_blend(members, actual)
    assert weights.shape == (3,)
    assert np.all(weights >= 0.0)
    assert weights.sum() <= 1.0 + 1e-12


def test_constrained_blend_known_hand_checked_solution():
    # NNLS of [[2,0],[0,2]] against [1,1] is exactly (0.5, 0.5), sum 1.
    weights = fit_constrained_blend(np.array([[2.0, 0.0], [0.0, 2.0]]), np.array([1.0, 1.0]))
    assert np.allclose(weights, [0.5, 0.5])


def test_constrained_blend_renormalizes_weights_above_one():
    # NNLS solution is (2, 2); renormalization divides by the sum of four.
    weights = fit_constrained_blend(np.eye(2), np.array([2.0, 2.0]))
    assert np.allclose(weights, [0.5, 0.5])
    assert weights.sum() <= 1.0


def test_constrained_blend_falls_back_to_equal_weights_for_zero_solution():
    # Opposite-signed members cannot reduce squared error, NNLS returns zeros,
    # and the documented fallback is equal weights summing to one.
    weights = fit_constrained_blend(-np.eye(3), np.ones(3))
    assert np.allclose(weights, np.full(3, 1.0 / 3.0))
    assert weights.sum() <= 1.0


def test_shrinkage_alpha_validation_errors():
    with pytest.raises(ValueError):
        fit_shrinkage_alpha([0.01, 0.02], [0.01])
    with pytest.raises(ValueError):
        fit_shrinkage_alpha([], [])
    with pytest.raises(ValueError):
        fit_shrinkage_alpha([np.nan], [0.01])
    with pytest.raises(ValueError):
        fit_shrinkage_alpha([0.01], [0.01], grid=[])
    with pytest.raises(ValueError):
        fit_shrinkage_alpha([0.01], [0.01], grid=[0.5, 1.5])


def test_constrained_blend_validation_errors():
    with pytest.raises(ValueError):
        fit_constrained_blend(np.ones((4, 2)), np.ones(3))
    with pytest.raises(ValueError):
        fit_constrained_blend(np.ones(4), np.ones(4))
    with pytest.raises(ValueError):
        fit_constrained_blend(np.empty((0, 2)), np.empty(0))
    with pytest.raises(ValueError):
        fit_constrained_blend(np.array([[np.nan, 0.0]]), np.ones(1))
    with pytest.raises(ValueError):
        fit_constrained_blend(np.ones((2, 2)), np.array([np.inf, 1.0]))
