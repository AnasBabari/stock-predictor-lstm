"""Statistical evidence helpers for leakage-safe forecast comparisons."""

from __future__ import annotations

from collections.abc import Iterator
from statistics import NormalDist

import numpy as np


def _aligned_losses(actual, candidate, baseline, loss: str) -> tuple[np.ndarray, np.ndarray]:
    observed = np.asarray(actual, dtype=float)
    contender = np.asarray(candidate, dtype=float)
    reference = np.asarray(baseline, dtype=float)
    if not (observed.shape == contender.shape == reference.shape) or observed.size < 2:
        raise ValueError("actual, candidate, and baseline must be aligned with at least two rows.")
    if not all(np.isfinite(value).all() for value in (observed, contender, reference)):
        raise ValueError("Loss inputs must be finite.")
    if loss == "absolute":
        candidate_loss = np.abs(observed - contender)
        baseline_loss = np.abs(observed - reference)
    elif loss == "squared":
        candidate_loss = (observed - contender) ** 2
        baseline_loss = (observed - reference) ** 2
    else:
        raise ValueError("loss must be 'absolute' or 'squared'.")
    if candidate_loss.ndim > 1:
        candidate_loss = np.mean(candidate_loss, axis=tuple(range(1, candidate_loss.ndim)))
        baseline_loss = np.mean(baseline_loss, axis=tuple(range(1, baseline_loss.ndim)))
    return candidate_loss.reshape(-1), baseline_loss.reshape(-1)


def _block_resample_chunks(
    n: int, block_length: int, resamples: int, generator: np.random.Generator
) -> Iterator[np.ndarray]:
    """Yield moving-block resampled index matrices of shape (chunk, n)."""
    block = min(block_length, n)
    needed = int(np.ceil(n / block))
    starts = n - block + 1
    chunk_size = 256
    for offset in range(0, resamples, chunk_size):
        end = min(offset + chunk_size, resamples)
        selected = generator.integers(starts, size=(end - offset, needed))
        positions = selected[:, :, None] + np.arange(block)
        yield positions.reshape(end - offset, needed * block)[:, :n]


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
    n = len(array)
    starts = np.arange(n - block + 1)
    generator = np.random.default_rng(seed)
    needed = int(np.ceil(n / block))
    sample_means = np.empty(resamples)
    chunk_size = 256
    for offset in range(0, resamples, chunk_size):
        end = min(offset + chunk_size, resamples)
        selected = generator.integers(len(starts), size=(end - offset, needed))
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


def paired_loss_evidence(
    actual, candidate, baseline, *, loss: str = "absolute", horizon: int = 1, **bootstrap_options
) -> dict:
    """Paired loss difference evidence; positive improvement means candidate wins."""
    candidate_loss, baseline_loss = _aligned_losses(actual, candidate, baseline, loss)
    difference = baseline_loss - candidate_loss
    options = dict(bootstrap_options)
    options.setdefault("block_length", max(int(horizon), 20))
    interval = moving_block_bootstrap_interval(difference, **options)
    # Newey-West HAC variance
    diff_arr = np.asarray(difference, dtype=float)
    n = len(diff_arr)
    if n > 1:
        mean_diff = float(np.mean(diff_arr))
        lag_max = int(np.floor(4 * ((n / 100) ** (2 / 9))))
        var = float(np.var(diff_arr, ddof=0))
        for lag in range(1, lag_max + 1):
            if lag < n:
                cov = float(np.mean((diff_arr[:-lag] - mean_diff) * (diff_arr[lag:] - mean_diff)))
                weight = 1.0 - (lag / (lag_max + 1.0))
                var += 2.0 * weight * cov
        standard_error = float(np.sqrt(max(var, 0.0) / n))
    else:
        standard_error = 0.0

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


def relative_ratio_evidence(
    actual,
    candidate,
    baseline,
    *,
    metric: str = "mae",
    horizon: int = 1,
    confidence: float = 0.95,
    resamples: int = 1000,
    seed: int = 42,
    block_length: int | None = None,
) -> dict:
    """Relative error-ratio evidence with a block-bootstrap confidence interval.

    The point statistic is ``mean(candidate_error) / mean(baseline_error)``;
    values below 1.0 mean the candidate beats the baseline. For ``metric="mae"``
    errors are absolute; for ``metric="rmse"`` errors are squared and the ratio
    is square-rooted, so the reported ratio is the RMSE ratio (the ratio of
    mean squared errors taken under the square root, matching the usual RMSE
    convention).

    The confidence interval is computed by bootstrapping the ratio directly:
    each moving-block resample draws aligned index blocks for both paired error
    series, recomputes the per-resample means, and evaluates the ratio on that
    resample. This keeps the interval on the ratio scale (skew preserved) and
    avoids anchoring approximations on the baseline error mean. Percentile
    quantiles of the resampled ratios form the interval.

    Multi-horizon 2-D inputs are reduced to a per-origin mean error exactly
    like :func:`paired_loss_evidence` (via ``_aligned_losses``).
    """
    if metric not in {"mae", "rmse"}:
        raise ValueError("metric must be 'mae' or 'rmse'.")
    if not 0 < confidence < 1 or resamples < 1:
        raise ValueError("Invalid bootstrap settings.")
    candidate_error, baseline_error = _aligned_losses(
        actual, candidate, baseline, "squared" if metric == "rmse" else "absolute"
    )
    baseline_mean = float(np.mean(baseline_error))
    if baseline_mean <= 0.0:
        raise ValueError("Baseline error mean must be positive to form a ratio.")
    candidate_mean = float(np.mean(candidate_error))
    ratio = candidate_mean / baseline_mean
    if metric == "rmse":
        ratio = float(np.sqrt(ratio))
    block = int(block_length) if block_length is not None else max(int(horizon), 20)
    generator = np.random.default_rng(seed)
    kept_ratios: list[np.ndarray] = []
    for indices in _block_resample_chunks(len(candidate_error), block, resamples, generator):
        baseline_means = baseline_error[indices].mean(axis=1)
        # Resamples whose drawn blocks are entirely zero-baseline-error rows
        # would divide by zero; drop them instead of emitting NaN bounds.
        valid = baseline_means > 0.0
        if not valid.any():
            continue
        valid_indices = indices[valid]
        candidate_means = candidate_error[valid_indices].mean(axis=1)
        chunk_ratios = candidate_means / baseline_means[valid]
        if metric == "rmse":
            chunk_ratios = np.sqrt(chunk_ratios)
        kept_ratios.append(chunk_ratios)
    if not kept_ratios:
        raise ValueError(
            "Every bootstrap resample had a non-positive baseline error mean; "
            "the ratio confidence interval is undefined for this baseline."
        )
    sample_ratios = np.concatenate(kept_ratios)
    alpha = (1 - confidence) / 2
    return {
        "metric": metric,
        "ratio": float(ratio),
        "confidence_interval": {
            "lower": float(np.quantile(sample_ratios, alpha)),
            "upper": float(np.quantile(sample_ratios, 1 - alpha)),
            "confidence": confidence,
            "resamples": resamples,
            "block_length": min(block, len(candidate_error)),
        },
        "sample_count": int(len(candidate_error)),
    }


def benjamini_hochberg(p_values, *, q: float = 0.10) -> list[bool]:
    """Return rejection decisions in original order using BH FDR control."""
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or not len(values) or np.any((values < 0) | (values > 1)) or not 0 < q < 1:
        raise ValueError("p_values must be a non-empty [0, 1] vector and q must be in (0, 1).")
    order = np.argsort(values)
    thresholds = q * (np.arange(1, len(values) + 1) / len(values))
    passing = values[order] <= thresholds
    decisions: np.ndarray = np.zeros(len(values), dtype=bool)
    if np.any(passing):
        decisions[order[: np.where(passing)[0][-1] + 1]] = True
    return decisions.tolist()
