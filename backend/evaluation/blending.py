"""Constrained blending primitives for leakage-safe forecast combination.

Both functions are pure: callers own the leakage-safe split of rows. Inputs
are log-return predictions (persistence corresponds to a zero log return), so
blend weights and shrinkage factors can be audited directly.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls


def _aligned_one_dimensional(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")
    return array


def fit_shrinkage_alpha(model_returns, actual_returns, *, grid=None) -> float:
    """Return the shrinkage factor alpha in [0, 1] for one model's predictions.

    The blended forecast is ``alpha * model`` shrunk toward persistence, which
    is a zero log return. Alpha is chosen from ``grid`` (default
    ``np.linspace(0, 1, 21)``) by minimizing the MAE of ``alpha * model``
    against the realized returns on a held-out calibration region. Ties keep
    the smallest alpha.
    """
    model_array = _aligned_one_dimensional(model_returns, "model_returns")
    actual_array = _aligned_one_dimensional(actual_returns, "actual_returns")
    if model_array.shape != actual_array.shape:
        raise ValueError("model_returns and actual_returns must be aligned.")
    if grid is None:
        candidates = np.linspace(0.0, 1.0, 21)
    else:
        candidates = np.asarray(grid, dtype=float).reshape(-1)
        if candidates.size == 0 or not np.isfinite(candidates).all():
            raise ValueError("grid must be a non-empty finite vector.")
        if np.any((candidates < 0.0) | (candidates > 1.0)):
            raise ValueError("grid values must lie within [0, 1].")
    errors = np.abs(candidates[:, None] * model_array[None, :] - actual_array[None, :])
    mean_absolute_errors = errors.mean(axis=1)
    return float(candidates[int(np.argmin(mean_absolute_errors))])


def fit_constrained_blend(member_predictions, actuals) -> np.ndarray:
    """Return non-negative blend weights over member log-return predictions.

    ``member_predictions`` has shape ``(n_samples, n_members)`` aligned with
    ``actuals``. Weights solve ``scipy.optimize.nnls``, so they are already
    non-negative (any numerical residue is clipped to zero) and are then
    renormalized so the weights sum to at most one: when the NNLS sum exceeds
    one every weight is divided by that sum, which keeps the blend inside the
    convex hull of persistence and the members. When NNLS returns an all-zero
    solution (no member reduces squared error) the honest fallback is pure
    persistence: all weights stay zero, so the blend degenerates to the
    baseline instead of committing to members that provably do not help.
    """
    members = np.asarray(member_predictions, dtype=float)
    observed = np.asarray(actuals, dtype=float).reshape(-1)
    if members.ndim != 2 or members.shape[0] == 0:
        raise ValueError("member_predictions must be a non-empty (n_samples, n_members) array.")
    if members.shape[0] != len(observed):
        raise ValueError("member_predictions and actuals must be aligned.")
    if not np.isfinite(members).all() or not np.isfinite(observed).all():
        raise ValueError("Blending inputs must be finite.")

    weights, _ = nnls(members, observed)
    weights = np.clip(weights, 0.0, None)
    total = float(weights.sum())
    if total > 1.0:
        weights = weights / total
    elif total == 0.0:
        # No member helps: stay with persistence (all-zero weights).
        weights = np.zeros(members.shape[1])
    return weights.astype(float)
