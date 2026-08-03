"""Multi-seed wrapper around the locked single-seed candidate evaluator.

The locked ``evaluate_candidate`` remains the single source of fold-aware,
persistence-relative metrics. This module simply repeats it once per seed on
the same snapshot and policy, counts per-seed failures (exceptions, timeouts,
or incomplete summaries), and aggregates the scalar summary fields into a
ledger-compatible record.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from .config import EvaluationPolicy
from .data import Snapshot
from .evaluation import evaluate_candidate

SCALAR_METRICS = (
    "median_relative_mae",
    "median_relative_rmse",
    "worst_fold_relative_rmse",
    "folds_beating_persistence",
)

# Error-style metrics are lower-is-better; these counts are higher-is-better.
_HIGHER_IS_BETTER = frozenset({"folds_beating_persistence"})


def _finite_summary_values(summaries: Sequence[dict[str, Any]], metric: str) -> list[float]:
    values: list[float] = []
    for summary in summaries:
        raw = summary.get(metric)
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return values


def _aggregate(values: list[float], *, higher_is_better: bool = False) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "std": None, "best": None, "worst": None}
    array = np.asarray(values, dtype=float)
    best = float(np.max(array)) if higher_is_better else float(np.min(array))
    worst = float(np.min(array)) if higher_is_better else float(np.max(array))
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "best": best,
        "worst": worst,
    }


def evaluate_multi_seed(
    snapshot: Snapshot,
    candidate_factory: Callable[[int], Any],
    *,
    horizon: int,
    policy: EvaluationPolicy,
    seeds: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Evaluate one candidate factory across multiple seeds.

    Defaults to ``seeds = tuple(range(policy.seed_count))``. Each seed invokes
    the locked ``evaluate_candidate`` exactly once; per-seed exceptions are
    caught and counted as failures instead of aborting the remaining seeds.
    """
    seed_values = tuple(seeds) if seeds is not None else tuple(range(policy.seed_count))
    if not seed_values:
        raise ValueError("seeds must contain at least one seed.")

    per_seed: list[dict[str, Any]] = []
    successful_summaries: list[dict[str, Any]] = []
    for seed in seed_values:
        try:
            result = evaluate_candidate(
                snapshot,
                candidate_factory,
                horizon=horizon,
                policy=policy,
                seed=seed,
            )
            summary = result.summary(policy)
        except Exception as exc:  # Per-seed failures are counted, not raised.
            per_seed.append(
                {
                    "seed": seed,
                    "status": "failed",
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
            )
            continue
        successful_summaries.append(summary)
        per_seed.append({"seed": seed, "status": "success", "summary": summary})

    failure_count = len(seed_values) - len(successful_summaries)
    aggregate = {
        metric: _aggregate(
            _finite_summary_values(successful_summaries, metric),
            higher_is_better=metric in _HIGHER_IS_BETTER,
        )
        for metric in SCALAR_METRICS
    }
    promotable_seed_count = int(
        sum(bool(summary.get("promotable")) for summary in successful_summaries)
    )

    record: dict[str, Any] = {
        "seeds": list(seed_values),
        "failure_count": failure_count,
        "per_seed": per_seed,
        "seed_aggregate": aggregate,
        "promotable_seed_count": promotable_seed_count,
        "status": "success"
        if failure_count == 0
        else ("crash" if failure_count == len(seed_values) else "partial"),
    }
    # Top-level scalars use the cross-seed median so the record remains
    # directly ledger-compatible (the ledger reads median_* keys verbatim).
    for metric in SCALAR_METRICS:
        record[metric] = aggregate[metric]["median"]
    record["promotable"] = failure_count == 0 and promotable_seed_count == len(seed_values)
    # Surface the first distinct per-seed error so callers (and the locked
    # ledger schema, which drops per_seed detail) can record why seeds failed.
    record["failure_reason"] = "; ".join(
        f"seed {entry['seed']}: {entry.get('error_type', 'Error')}: {entry.get('error', '')}"
        for entry in per_seed
        if entry["status"] == "failed"
    )
    return record
