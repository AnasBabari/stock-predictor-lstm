"""Self-contained statistical evidence helpers for the research harness.

This module is intentionally isolated from the locked harness modules: it
imports only numpy and the standard library, and operates on plain arrays of
fold-level or row-level losses that callers derive from untouched
``evaluate_candidate`` results.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np


def block_bootstrap_interval(
    values,
    *,
    confidence: float = 0.95,
    resamples: int = 1000,
    block_length: int = 20,
    seed: int = 0,
) -> dict[str, float | int]:
    """Moving-block bootstrap percentile CI for the mean of a time-ordered series.

    Blocks of consecutive observations are resampled with replacement to
    preserve local dependence, truncated to the series length, and the mean of
    each resample is recorded. Percentile quantiles of the resampled means form
    the interval.
    """
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size < 2 or not np.isfinite(array).all():
        raise ValueError("values must contain at least two finite values.")
    if not 0 < confidence < 1 or resamples < 1 or block_length < 1:
        raise ValueError("Invalid bootstrap settings.")
    n = len(array)
    block = min(int(block_length), n)
    needed = math.ceil(n / block)
    starts = n - block + 1
    generator = np.random.default_rng(seed)
    sample_means = np.empty(resamples)
    chunk_size = 256
    for offset in range(0, resamples, chunk_size):
        end = min(offset + chunk_size, resamples)
        selected = generator.integers(starts, size=(end - offset, needed))
        positions = selected[:, :, None] + np.arange(block)
        gathered = array[positions].reshape(end - offset, needed * block)
        sample_means[offset:end] = gathered[:, :n].sum(axis=1) / n
    alpha = (1 - confidence) / 2
    return {
        "estimate": float(np.mean(array)),
        "lower": float(np.quantile(sample_means, alpha)),
        "upper": float(np.quantile(sample_means, 1 - alpha)),
        "confidence": confidence,
        "resamples": resamples,
        "block_length": block,
    }


def dm_style_statistic(
    candidate_losses, baseline_losses, *, max_lag: int | None = None
) -> dict[str, float | int]:
    """Paired Diebold-Mariano-style statistic with Newey-West HAC variance.

    The loss differential is ``candidate - baseline`` (negative means the
    candidate has lower loss). The variance of the differential mean uses the
    Newey-West estimator with ``lag_max = floor(4 * (n / 100) ** (2 / 9))``
    unless ``max_lag`` is given, and a two-sided normal p-value is reported.

    Overlapping multi-step losses (h-step-ahead evaluated at every origin)
    are autocorrelated up to roughly ``h - 1`` lags; the default rule
    undersmooths for large ``h``, so callers comparing overlapping horizons
    should pass ``max_lag >= h``.
    """
    contender = np.asarray(candidate_losses, dtype=float).reshape(-1)
    reference = np.asarray(baseline_losses, dtype=float).reshape(-1)
    if contender.shape != reference.shape or contender.size < 2:
        raise ValueError(
            "candidate_losses and baseline_losses must be aligned with at least two rows."
        )
    if not (np.isfinite(contender).all() and np.isfinite(reference).all()):
        raise ValueError("Loss inputs must be finite.")
    difference = contender - reference
    n = len(difference)
    mean_diff = float(np.mean(difference))
    lag_max = (
        max(1, int(max_lag))
        if max_lag is not None
        else math.floor(4 * ((n / 100) ** (2 / 9)))
    )
    variance = float(np.var(difference, ddof=0))
    for lag in range(1, min(lag_max, n - 1) + 1):
        covariance = float(
            np.mean((difference[:-lag] - mean_diff) * (difference[lag:] - mean_diff))
        )
        weight = 1.0 - (lag / (lag_max + 1.0))
        variance += 2.0 * weight * covariance
    standard_error = math.sqrt(max(variance, 0.0) / n)
    if standard_error <= 0.0:
        return {
            "statistic": 0.0,
            "two_sided_p_value": 1.0,
            "mean_difference": mean_diff,
            "sample_count": n,
        }
    statistic = mean_diff / standard_error
    p_value = float(2 * (1 - NormalDist().cdf(abs(statistic))))
    return {
        "statistic": float(statistic),
        "two_sided_p_value": p_value,
        "mean_difference": mean_diff,
        "sample_count": n,
    }


def fold_metric_evidence(
    fold_metrics_sequence,
    *,
    metric: str = "relative_rmse",
    confidence: float = 0.95,
    resamples: int = 1000,
    block_length: int = 20,
    seed: int = 0,
) -> dict[str, object]:
    """Bootstrap evidence for the mean of a per-fold metric sequence.

    Accepts either a sequence of metric dicts (e.g. the ``metrics`` payloads of
    ``FoldResult`` objects from ``evaluate_candidate``) or a plain sequence of
    floats. With fewer than five folds the estimate is unreliable, so the point
    estimate is returned with a ``None`` interval and ``reliable=False``.
    """
    values: list[float] = []
    for item in fold_metrics_sequence:
        if isinstance(item, dict):
            if metric not in item:
                raise ValueError(f"Fold metrics dict is missing key '{metric}'.")
            values.append(float(item[metric]))
        else:
            values.append(float(item))
    if not values:
        raise ValueError("fold_metrics_sequence must not be empty.")
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("Fold metric values must be finite.")
    estimate = float(np.mean(array))
    if len(array) < 5:
        return {
            "metric": metric,
            "estimate": estimate,
            "confidence_interval": None,
            "fold_count": len(array),
            "reliable": False,
        }
    interval = block_bootstrap_interval(
        array,
        confidence=confidence,
        resamples=resamples,
        block_length=block_length,
        seed=seed,
    )
    return {
        "metric": metric,
        "estimate": estimate,
        "confidence_interval": interval,
        "fold_count": len(array),
        "reliable": True,
    }
