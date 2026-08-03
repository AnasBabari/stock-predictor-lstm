"""Drift and out-of-distribution diagnostics for model inputs and residuals."""

from __future__ import annotations

import numpy as np

from evaluation.evidence import moving_block_bootstrap_interval

_EPSILON = 1e-4


def _one_dimensional(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")
    return array


def _bin_proportions(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    indices = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)
    counts = np.bincount(indices, minlength=len(edges) - 1).astype(float)
    return (counts + _EPSILON) / (values.size + _EPSILON * (len(edges) - 1))


def population_stability_index(reference, comparison, *, bins: int = 10) -> float:
    """Quantile-binned PSI of ``comparison`` relative to ``reference``.

    Bin edges are fitted on the reference distribution only. A constant
    reference collapses to a single effective bin, which yields a finite PSI
    of zero because both distributions occupy the same bin.
    """
    reference_array = _one_dimensional(reference, "reference")
    comparison_array = _one_dimensional(comparison, "comparison")
    if not isinstance(bins, int) or bins < 2:
        raise ValueError("bins must be an integer of at least two.")

    edges = np.unique(np.quantile(reference_array, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 2:
        # Degenerate reference: a single effective bin cannot diverge.
        return 0.0

    reference_proportions = _bin_proportions(reference_array, edges)
    comparison_proportions = _bin_proportions(comparison_array, edges)
    psi = np.sum(
        (comparison_proportions - reference_proportions)
        * np.log(comparison_proportions / reference_proportions)
    )
    return float(psi)


def feature_divergence(train_windows, validation_windows, *, bins: int = 10) -> dict:
    """Per-feature PSI between training and validation feature columns.

    Accepts ``(samples, lookback, features)`` or ``(rows, features)`` arrays;
    all rows are pooled per feature column before computing PSI.
    """
    train_array = np.asarray(train_windows, dtype=float)
    validation_array = np.asarray(validation_windows, dtype=float)
    for name, array in (("train_windows", train_array), ("validation_windows", validation_array)):
        if array.ndim not in (2, 3):
            raise ValueError(f"{name} must be a two- or three-dimensional array.")
        if array.size == 0:
            raise ValueError(f"{name} must not be empty.")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains non-finite values.")
    if train_array.shape[-1] != validation_array.shape[-1]:
        raise ValueError("train_windows and validation_windows must share the feature dimension.")

    feature_count = train_array.shape[-1]
    train_flat = train_array.reshape(-1, feature_count)
    validation_flat = validation_array.reshape(-1, feature_count)
    psi_by_column = [
        population_stability_index(train_flat[:, column], validation_flat[:, column], bins=bins)
        for column in range(feature_count)
    ]
    return {
        "psi_by_column": psi_by_column,
        "max_psi": float(max(psi_by_column)),
        "mean_psi": float(np.mean(psi_by_column)),
    }


def residual_drift(
    pooled_residuals,
    *,
    resamples: int = 250,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict:
    """Chronological half-split drift check on pooled error magnitudes.

    Drift is flagged when the bootstrap confidence intervals of the two
    half-means are disjoint.
    """
    residuals = _one_dimensional(pooled_residuals, "pooled_residuals")
    if residuals.size < 10:
        raise ValueError("pooled_residuals must contain at least ten finite values.")
    if resamples < 1:
        raise ValueError("resamples must be at least one.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between zero and one.")

    midpoint = residuals.size // 2
    first_half = residuals[:midpoint]
    second_half = residuals[midpoint:]
    block_length = max(1, min(20, min(len(first_half), len(second_half)) // 4))
    options = {"resamples": resamples, "seed": seed, "confidence": confidence}
    first_interval = moving_block_bootstrap_interval(
        first_half, block_length=block_length, **options
    )
    second_interval = moving_block_bootstrap_interval(
        second_half, block_length=block_length, **options
    )
    drift_detected = bool(
        first_interval["upper"] < second_interval["lower"]
        or second_interval["upper"] < first_interval["lower"]
    )
    return {
        "first_half_mae": float(np.mean(first_half)),
        "second_half_mae": float(np.mean(second_half)),
        "difference": float(np.mean(second_half) - np.mean(first_half)),
        "first_half_ci": first_interval,
        "second_half_ci": second_interval,
        "drift_detected": drift_detected,
    }
