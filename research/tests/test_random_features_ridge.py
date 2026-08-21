"""Structural and behavioural tests for the random_features_ridge candidate.

These tests exist because the candidate was formerly named ``small_tcn`` while
containing no convolution whatsoever. They pin the actual mechanism so the
name can never silently drift away from the implementation again.
"""

from __future__ import annotations

import numpy as np
import pytest
from stock_autoresearch.candidates import (
    LEGACY_FAMILY_ALIASES,
    RandomFeaturesRidgeCandidate,
    canonical_family,
)


def _window(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 60, 5))


def test_family_name_describes_the_mechanism() -> None:
    assert RandomFeaturesRidgeCandidate.name == "random_features_ridge"
    candidate = RandomFeaturesRidgeCandidate()
    assert candidate.describe()["family"] == "random_features_ridge"
    # The former misleading name must not reappear in the description.
    assert "tcn" not in candidate.describe()["family"]


def test_legacy_alias_maps_only_at_boundaries() -> None:
    assert LEGACY_FAMILY_ALIASES == {"small_tcn": "random_features_ridge"}
    assert canonical_family("small_tcn") == "random_features_ridge"
    assert canonical_family("ridge") == "ridge"
    assert canonical_family("random_features_ridge") == "random_features_ridge"


def test_ignores_everything_except_the_final_window_row() -> None:
    """A convolution over the window would react to earlier timesteps.

    The implementation pools only the latest temporal state of two dense
    projections, so perturbing the entire history except the final row must
    leave predictions bit-identical. This is the opposite of temporal
    convolution and is exactly what the rename documents.
    """
    x = _window(16, seed=7)
    y = np.linspace(-1.0, 1.0, 16)
    baseline = RandomFeaturesRidgeCandidate(seed=3).fit(x, y).predict(x)

    perturbed = x.copy()
    perturbed[:, :-1, :] = rng_permutation(perturbed[:, :-1, :])
    scrambled = RandomFeaturesRidgeCandidate(seed=3).fit(perturbed, y).predict(perturbed)

    np.testing.assert_array_equal(baseline, scrambled)


def rng_permutation(array: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(11)
    flat = array.reshape(array.shape[0], -1)
    return rng.permutation(flat).reshape(array.shape)


def test_fixed_random_projection_is_seeded_and_ridge_is_trained() -> None:
    x = _window(12, seed=1)
    y = np.linspace(0.0, 1.0, 12)

    a = RandomFeaturesRidgeCandidate(seed=5).fit(x.copy(), y)
    b = RandomFeaturesRidgeCandidate(seed=5).fit(x.copy(), y)
    np.testing.assert_array_equal(a.predict(x), b.predict(x))

    c = RandomFeaturesRidgeCandidate(seed=6).fit(x.copy(), y)
    assert not np.array_equal(a.predict(x), c.predict(x))

    # A Ridge readout was actually fitted on 2*channels random features.
    assert a._model.coef_ is not None
    assert a._model.coef_.size == 2 * a.channels


def test_parameter_count_reports_only_trained_parameters() -> None:
    candidate = RandomFeaturesRidgeCandidate(channels=4).fit(_window(8), np.zeros(8))
    # Ridge intercept + coefficients; the random projection is not trained.
    assert candidate.parameter_count() == 2 * candidate.channels + 1


@pytest.mark.parametrize("seed", [42, 7])
def test_predict_shape_and_dtype(seed: int) -> None:
    x = _window(10, seed=seed)
    model = RandomFeaturesRidgeCandidate(seed=seed).fit(x, np.zeros(10))
    out = model.predict(x)
    assert out.shape == (10,)
    assert out.dtype == np.float64
