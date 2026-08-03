"""Locked, fold-aware evaluation of candidates against persistence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .candidates import Candidate, PersistenceCandidate
from .config import EvaluationPolicy
from .data import Snapshot, build_examples, expanding_folds
from .metrics import regression_metrics


@dataclass(frozen=True)
class FoldResult:
    fold: int
    metrics: dict[str, float]
    rows: int


@dataclass(frozen=True)
class EvaluationResult:
    candidate: dict[str, object]
    snapshot_id: str
    horizon: int
    folds: tuple[FoldResult, ...]
    complete: bool = True

    def summary(self, policy: EvaluationPolicy) -> dict[str, object]:
        relative_mae = [f.metrics["relative_mae"] for f in self.folds]
        relative_rmse = [f.metrics["relative_rmse"] for f in self.folds]
        return {
            "candidate": self.candidate,
            "snapshot_id": self.snapshot_id,
            "horizon": self.horizon,
            "folds": len(self.folds),
            "complete": self.complete,
            "median_relative_mae": float(np.median(relative_mae)),
            "median_relative_rmse": float(np.median(relative_rmse)),
            "worst_fold_relative_rmse": float(np.max(relative_rmse)),
            "folds_beating_persistence": int(sum(value < 1.0 for value in relative_rmse)),
            "promotable": bool(
                self.complete
                and np.median(relative_mae) < policy.relative_error_gate
                and np.median(relative_rmse) < policy.relative_error_gate
                and np.max(relative_rmse) <= policy.worst_fold_gate
                and sum(value < 1.0 for value in relative_rmse) >= max(1, policy.folds - 1)
            ),
        }


def evaluate_candidate(
    snapshot: Snapshot,
    candidate_factory,
    *,
    horizon: int,
    policy: EvaluationPolicy,
    seed: int = 0,
) -> EvaluationResult:
    x, y, _ = build_examples(snapshot, window=policy.window, horizon=horizon)
    purge = horizon - 1
    fold_results: list[FoldResult] = []
    for fold_no, (train_idx, validation_idx) in enumerate(
        expanding_folds(
            len(x),
            folds=policy.folds,
            minimum_train_rows=policy.minimum_train_rows,
            validation_rows=policy.minimum_validation_rows,
            purge=purge,
        ),
        start=1,
    ):
        candidate: Candidate = candidate_factory(seed)
        candidate.fit(x[train_idx], y[train_idx])
        predicted = candidate.predict(x[validation_idx])
        baseline = PersistenceCandidate().predict(x[validation_idx])
        metrics = regression_metrics(y[validation_idx], predicted, baseline)
        fold_results.append(FoldResult(fold=fold_no, metrics=metrics, rows=len(validation_idx)))
    return EvaluationResult(
        candidate=candidate_factory(seed).describe(),
        snapshot_id=snapshot.snapshot_id,
        horizon=horizon,
        folds=tuple(fold_results),
    )
