"""Aggregation of per-seed walk-forward evaluation summaries.

Multi-seed evaluation re-fits stochastic models once per seed while keeping
the dataset and fold plan fixed. This module collapses the resulting per-seed
metric summaries into mean/median/std/best/worst aggregates without touching
the promotion policy, which still operates on a single fold plan.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def _finite_metric_value(summary: Mapping[str, object] | None, metric: str) -> float | None:
    """Return the metric as a finite float, or None when the seed run failed."""
    if summary is None:
        return None
    try:
        value = summary.get(metric)
    except AttributeError:
        return None
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def aggregate_seed_runs(
    per_seed_summaries: Sequence[Mapping[str, float] | None],
    *,
    metrics: Sequence[str] = ("relative_mae", "relative_rmse"),
) -> dict:
    """Aggregate per-seed metric summaries into robust cross-seed statistics.

    Each entry contributes one scalar per requested metric. Entries that are
    ``None`` or that contain a missing or non-finite value for any requested
    metric count as failed seed runs: they increment ``failure_count`` and are
    excluded from every aggregate. When every seed run failed, each aggregate
    value is ``None`` and ``failure_count`` equals the number of entries.

    Returns ``{metric: {"mean", "median", "std", "best", "worst"}, ...,
    "failure_count": int}`` where ``best`` is the minimum and ``worst`` the
    maximum observed value.
    """
    if len(per_seed_summaries) == 0:
        raise ValueError("per_seed_summaries must contain at least one seed run.")
    metric_names = tuple(metrics)
    if not metric_names:
        raise ValueError("metrics must identify at least one metric.")

    valid_rows: list[dict[str, float]] = []
    for summary in per_seed_summaries:
        row: dict[str, float] = {}
        for metric in metric_names:
            value = _finite_metric_value(summary, metric)
            if value is None:
                row = {}
                break
            row[metric] = value
        if row:
            valid_rows.append(row)

    failure_count = len(per_seed_summaries) - len(valid_rows)
    result: dict[str, object] = {}
    for metric in metric_names:
        if not valid_rows:
            result[metric] = {
                "mean": None,
                "median": None,
                "std": None,
                "best": None,
                "worst": None,
            }
            continue
        values = np.asarray([row[metric] for row in valid_rows], dtype=float)
        result[metric] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "best": float(np.min(values)),
            "worst": float(np.max(values)),
        }
    result["failure_count"] = failure_count
    return result
