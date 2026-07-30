"""Statistical evidence helpers for leakage-safe forecast comparisons."""

from __future__ import annotations

from statistics import NormalDist

import numpy as np


def _aligned_losses(actual, candidate, baseline, loss: str) -> tuple[np.ndarray, np.ndarray]:
    observed = np.asarray(actual, dtype=float).reshape(-1)
    contender = np.asarray(candidate, dtype=float).reshape(-1)
    reference = np.asarray(baseline, dtype=float).reshape(-1)
    if not (observed.shape == contender.shape == reference.shape) or observed.size < 2:
        raise ValueError("actual, candidate, and baseline must be aligned with at least two rows.")
    if not all(np.isfinite(value).all() for value in (observed, contender, reference)):
        raise ValueError("Loss inputs must be finite.")
    if loss == "absolute":
        return np.abs(observed - contender), np.abs(observed - reference)
    if loss == "squared":
        return (observed - contender) ** 2, (observed - reference) ** 2
    raise ValueError("loss must be 'absolute' or 'squared'.")


def moving_block_bootstrap_interval(
    values,
    *,
    confidence: float = 0.95,
    resamples: int = 1000,
    block_length: int = 20,
    seed: int = 42,
) -> dict[str, float | int]:
    """Return a percentile CI of a mean while respecting local time dependence."""
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size < 2 or not np.isfinite(array).all():
        raise ValueError("values must contain at least two finite values.")
    if not 0 < confidence < 1 or resamples < 1 or block_length < 1:
        raise ValueError("Invalid bootstrap settings.")
    block = min(block_length, len(array))
    starts = np.arange(len(array) - block + 1)
    generator = np.random.default_rng(seed)
    sample_means = np.empty(resamples)
    needed = int(np.ceil(len(array) / block))
    for number in range(resamples):
        selected = generator.choice(starts, size=needed, replace=True)
        sample = np.concatenate([array[start : start + block] for start in selected])[: len(array)]
        sample_means[number] = np.mean(sample)
    alpha = (1 - confidence) / 2
    return {
        "estimate": float(np.mean(array)),
        "lower": float(np.quantile(sample_means, alpha)),
        "upper": float(np.quantile(sample_means, 1 - alpha)),
        "confidence": confidence,
        "resamples": resamples,
        "block_length": block,
    }


def paired_loss_evidence(
    actual, candidate, baseline, *, loss: str = "absolute", horizon: int = 1, **bootstrap_options
) -> dict:
    """Paired loss difference evidence; positive improvement means candidate wins."""
    candidate_loss, baseline_loss = _aligned_losses(actual, candidate, baseline, loss)
    difference = baseline_loss - candidate_loss
    options = dict(bootstrap_options)
    options.setdefault("block_length", max(int(horizon), 20))
    interval = moving_block_bootstrap_interval(difference, **options)
    # A normal approximation is a transparent DM-style supporting signal. It
    # is not used as the sole promotion criterion.
    standard_error = float(np.std(difference, ddof=1) / np.sqrt(len(difference)))
    statistic = float(np.mean(difference) / standard_error) if standard_error else 0.0
    p_value = float(2 * (1 - NormalDist().cdf(abs(statistic)))) if standard_error else 1.0
    return {
        "loss": loss,
        "mean_improvement": interval["estimate"],
        "confidence_interval": interval,
        "dm_style_statistic": statistic,
        "two_sided_p_value": p_value,
        "sample_count": int(len(difference)),
    }


def benjamini_hochberg(p_values, *, q: float = 0.10) -> list[bool]:
    """Return rejection decisions in original order using BH FDR control."""
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or not len(values) or np.any((values < 0) | (values > 1)) or not 0 < q < 1:
        raise ValueError("p_values must be a non-empty [0, 1] vector and q must be in (0, 1).")
    order = np.argsort(values)
    thresholds = q * (np.arange(1, len(values) + 1) / len(values))
    passing = values[order] <= thresholds
    decisions = np.zeros(len(values), dtype=bool)
    if np.any(passing):
        decisions[order[: np.where(passing)[0][-1] + 1]] = True
    return decisions.tolist()
