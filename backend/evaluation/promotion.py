"""Deterministic model-promotion policy based on out-of-fold evidence."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median


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


@dataclass(frozen=True)
class UniversePromotionPolicy:
    minimum_assets: int = 12
    minimum_qualifying_assets: int = 8
    maximum_median_relative_mae: float = 0.95
    maximum_median_relative_rmse: float = 0.95
    catastrophic_relative_rmse: float = 1.25
    maximum_catastrophic_assets: int = 1


@dataclass(frozen=True)
class DirectionPromotionPolicy:
    minimum_balanced_accuracy: float = 0.53
    minimum_probability_improvement: float = 0.02
    minimum_winning_folds: int = 4
    maximum_expected_calibration_error: float = 0.10


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


def assess_universe_promotion(
    ticker_reports: dict[str, dict],
    *,
    policy: UniversePromotionPolicy | None = None,
) -> PromotionDecision:
    """Require broad, stable evidence before an architecture family is eligible."""

    selected = policy or UniversePromotionPolicy()
    reasons: list[str] = []
    if len(ticker_reports) < selected.minimum_assets:
        reasons.append(f"universe requires at least {selected.minimum_assets} assets")

    ratios: list[tuple[float, float]] = []
    qualifying = 0
    catastrophic = 0
    for report in ticker_reports.values():
        pooled = report.get("pooled", report.get("aggregate", {}).get("pooled", {}))
        relative_mae = pooled.get("relative_mae")
        relative_rmse = pooled.get("relative_rmse")
        if isinstance(relative_mae, (int, float)) and isinstance(relative_rmse, (int, float)):
            ratios.append((float(relative_mae), float(relative_rmse)))
            catastrophic += relative_rmse > selected.catastrophic_relative_rmse
        if report.get("promoted", report.get("promotion", {}).get("promoted", False)):
            qualifying += 1

    if len(ratios) != len(ticker_reports):
        reasons.append("every universe asset must provide relative MAE and RMSE")
    if qualifying < selected.minimum_qualifying_assets:
        reasons.append(
            f"at least {selected.minimum_qualifying_assets} assets must pass the ticker gate"
        )
    if ratios:
        if median(value[0] for value in ratios) > selected.maximum_median_relative_mae:
            reasons.append(
                f"median relative MAE must be <= {selected.maximum_median_relative_mae:.3f}"
            )
        if median(value[1] for value in ratios) > selected.maximum_median_relative_rmse:
            reasons.append(
                f"median relative RMSE must be <= {selected.maximum_median_relative_rmse:.3f}"
            )
    if catastrophic > selected.maximum_catastrophic_assets:
        reasons.append(
            f"no more than {selected.maximum_catastrophic_assets} assets may be catastrophic"
        )
    return PromotionDecision(promoted=not reasons, reasons=tuple(reasons))


def assess_direction_promotion(
    pooled_metrics: dict,
    fold_metrics: list[dict],
    *,
    baseline_metrics: dict,
    policy: DirectionPromotionPolicy | None = None,
) -> PromotionDecision:
    """Gate direction candidates on discrimination, calibration, and fold stability."""

    selected = policy or DirectionPromotionPolicy()
    reasons: list[str] = []
    balanced_accuracy = pooled_metrics.get("balanced_accuracy")
    if (
        not isinstance(balanced_accuracy, (int, float))
        or balanced_accuracy < selected.minimum_balanced_accuracy
    ):
        reasons.append(f"balanced accuracy must be >= {selected.minimum_balanced_accuracy:.3f}")
    for metric in ("brier_score", "log_loss"):
        candidate = pooled_metrics.get(metric)
        baseline = baseline_metrics.get(metric)
        if (
            not isinstance(candidate, (int, float))
            or not isinstance(baseline, (int, float))
            or baseline <= 0
            or candidate / baseline > 1 - selected.minimum_probability_improvement
        ):
            reasons.append(
                f"{metric} must improve over the base-rate forecast by at least "
                f"{selected.minimum_probability_improvement:.1%}"
            )
    calibration_error = pooled_metrics.get("expected_calibration_error")
    if (
        not isinstance(calibration_error, (int, float))
        or calibration_error > selected.maximum_expected_calibration_error
    ):
        reasons.append(
            "expected calibration error must be <= "
            f"{selected.maximum_expected_calibration_error:.3f}"
        )
    winning_folds = sum(
        fold.get("balanced_accuracy", 0) > 0.5
        and fold.get("brier_score", float("inf")) < fold.get("baseline_brier_score", float("-inf"))
        for fold in fold_metrics
    )
    required_wins = min(selected.minimum_winning_folds, len(fold_metrics))
    if not fold_metrics or winning_folds < required_wins:
        reasons.append(f"direction model must beat its baseline in {required_wins} folds")
    return PromotionDecision(promoted=not reasons, reasons=tuple(reasons))
