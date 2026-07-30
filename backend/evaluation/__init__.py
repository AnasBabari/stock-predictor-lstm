"""Leakage-aware forecast evaluation primitives."""

from .conformal import calibrate_intervals, interval_diagnostics, prediction_intervals
from .evidence import benjamini_hochberg, moving_block_bootstrap_interval, paired_loss_evidence
from .metrics import (
    evaluate_forecast_horizons,
    evaluate_probability_forecast,
    regression_metrics,
)
from .promotion import (
    DirectionPromotionPolicy,
    PromotionDecision,
    PromotionPolicy,
    UniversePromotionPolicy,
    assess_direction_promotion,
    assess_promotion,
    assess_universe_promotion,
)
from .splits import generate_walk_forward_splits, purged_tail_split

__all__ = [
    "PromotionDecision",
    "PromotionPolicy",
    "DirectionPromotionPolicy",
    "UniversePromotionPolicy",
    "assess_direction_promotion",
    "assess_promotion",
    "assess_universe_promotion",
    "evaluate_forecast_horizons",
    "evaluate_probability_forecast",
    "generate_walk_forward_splits",
    "purged_tail_split",
    "regression_metrics",
    "benjamini_hochberg",
    "calibrate_intervals",
    "interval_diagnostics",
    "moving_block_bootstrap_interval",
    "paired_loss_evidence",
    "prediction_intervals",
]
