"""Plausibility, calibration, and abstention gate for multi-horizon forecasts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

DecisionState = Literal[
    "promoted_model",
    "promoted_blend",
    "abstain_extreme_unconfirmed",
    "abstain_model_disagreement",
    "abstain_failed_baseline_gate",
    "abstain_uncalibrated",
]


@dataclass(frozen=True)
class GateEvaluationResult:
    decision: DecisionState
    is_promoted: bool
    jump_score: float
    model_disagreement_pct: float
    relative_loss_vs_baseline: float
    coverage_error_pct: float
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PlausibilityAbstentionGate:
    """Evaluates first-step jump plausibility, candidate agreement, baseline superiority, and interval calibration."""

    def __init__(
        self,
        max_jump_score: float = 3.5,
        max_disagreement_pct: float = 8.0,
        sigma_floor: float = 0.005,
    ) -> None:
        self.max_jump_score = max_jump_score
        self.max_disagreement_pct = max_disagreement_pct
        self.sigma_floor = sigma_floor

    def evaluate(
        self,
        predicted_day1_log_return: float,
        predicted_day1_volatility: float,
        candidate_day1_returns: list[float] | None = None,
        relative_loss_vs_baseline: float = 0.90,
        coverage_80pct: float = 0.80,
        has_confirmed_catalyst: bool = False,
    ) -> GateEvaluationResult:
        # 1. Jump Score
        sigma = max(predicted_day1_volatility, self.sigma_floor)
        jump_score = abs(predicted_day1_log_return) / sigma

        # 2. Disagreement
        disagreement_pct = 0.0
        if candidate_day1_returns and len(candidate_day1_returns) > 1:
            disagreement_pct = (max(candidate_day1_returns) - min(candidate_day1_returns)) * 100.0

        # 3. Calibration Error
        coverage_error = abs(coverage_80pct - 0.80) * 100.0

        # Gate Checks
        if relative_loss_vs_baseline >= 1.0:
            return GateEvaluationResult(
                decision="abstain_failed_baseline_gate",
                is_promoted=False,
                jump_score=round(jump_score, 2),
                model_disagreement_pct=round(disagreement_pct, 2),
                relative_loss_vs_baseline=round(relative_loss_vs_baseline, 4),
                coverage_error_pct=round(coverage_error, 2),
                rationale=(
                    f"Candidate failed baseline gate with relative loss {relative_loss_vs_baseline:.4f} >= 1.0"
                ),
            )

        if disagreement_pct > self.max_disagreement_pct:
            return GateEvaluationResult(
                decision="abstain_model_disagreement",
                is_promoted=False,
                jump_score=round(jump_score, 2),
                model_disagreement_pct=round(disagreement_pct, 2),
                relative_loss_vs_baseline=round(relative_loss_vs_baseline, 4),
                coverage_error_pct=round(coverage_error, 2),
                rationale=(
                    f"Candidate models exhibit extreme Day-1 disagreement ({disagreement_pct:.2f}% > {self.max_disagreement_pct:.1f}%)"
                ),
            )

        if jump_score > self.max_jump_score and not has_confirmed_catalyst:
            return GateEvaluationResult(
                decision="abstain_extreme_unconfirmed",
                is_promoted=False,
                jump_score=round(jump_score, 2),
                model_disagreement_pct=round(disagreement_pct, 2),
                relative_loss_vs_baseline=round(relative_loss_vs_baseline, 4),
                coverage_error_pct=round(coverage_error, 2),
                rationale=(
                    f"Day-1 implied jump score ({jump_score:.2f} sigma) exceeds plausibility ceiling ({self.max_jump_score:.1f}) without causal catalyst"
                ),
            )

        if coverage_error > 20.0:
            return GateEvaluationResult(
                decision="abstain_uncalibrated",
                is_promoted=False,
                jump_score=round(jump_score, 2),
                model_disagreement_pct=round(disagreement_pct, 2),
                relative_loss_vs_baseline=round(relative_loss_vs_baseline, 4),
                coverage_error_pct=round(coverage_error, 2),
                rationale=f"Forecast intervals are uncalibrated (coverage error {coverage_error:.1f}% > 20%)",
            )

        return GateEvaluationResult(
            decision="promoted_model",
            is_promoted=True,
            jump_score=round(jump_score, 2),
            model_disagreement_pct=round(disagreement_pct, 2),
            relative_loss_vs_baseline=round(relative_loss_vs_baseline, 4),
            coverage_error_pct=round(coverage_error, 2),
            rationale="Candidate passed plausibility, agreement, calibration, and baseline gates.",
        )
