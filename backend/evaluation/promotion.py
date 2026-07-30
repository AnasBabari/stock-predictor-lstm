"""Deterministic model-promotion policy based on out-of-fold evidence."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromotionPolicy:
    minimum_relative_improvement: float = 0.05
    minimum_winning_folds: int = 4
    maximum_fold_relative_rmse: float = 1.25
    require_scaled_error_below_one: bool = True


@dataclass(frozen=True)
class PromotionDecision:
    promoted: bool
    reasons: tuple[str, ...]


def assess_promotion(
    pooled_metrics: dict,
    fold_metrics: list[dict],
    *,
    policy: PromotionPolicy | None = None,
) -> PromotionDecision:
    """Require meaningful pooled and fold-level improvement over persistence."""

    selected = policy or PromotionPolicy()
    reasons: list[str] = []
    relative_mae = pooled_metrics.get("relative_mae")
    relative_rmse = pooled_metrics.get("relative_rmse")
    required_ratio = 1 - selected.minimum_relative_improvement

    if relative_mae is None or relative_mae > required_ratio:
        reasons.append(f"pooled relative MAE must be <= {required_ratio:.3f}")
    if relative_rmse is None or relative_rmse > required_ratio:
        reasons.append(f"pooled relative RMSE must be <= {required_ratio:.3f}")

    if selected.require_scaled_error_below_one:
        if pooled_metrics.get("mase") is None or pooled_metrics["mase"] >= 1:
            reasons.append("pooled MASE must be below 1")
        if pooled_metrics.get("rmsse") is None or pooled_metrics["rmsse"] >= 1:
            reasons.append("pooled RMSSE must be below 1")

    winning_folds = sum(
        metric.get("relative_mae", float("inf")) < 1
        and metric.get("relative_rmse", float("inf")) < 1
        for metric in fold_metrics
    )
    if winning_folds < selected.minimum_winning_folds:
        reasons.append(
            f"model must beat persistence in at least {selected.minimum_winning_folds} folds"
        )

    if any(
        metric.get("relative_rmse", float("inf")) > selected.maximum_fold_relative_rmse
        for metric in fold_metrics
    ):
        reasons.append(
            f"no fold may exceed {selected.maximum_fold_relative_rmse:.2f}x persistence RMSE"
        )

    return PromotionDecision(promoted=not reasons, reasons=tuple(reasons))
