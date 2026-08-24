"""Frozen contracts for the global volatility-distribution candidate.

The primary task is cumulative realized variance, not dollar price. Return
location and three-way direction are auxiliary outputs and can be shrunk to
their matched baselines independently when they fail their own evidence gate.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.panel.features import DEPLOYABLE_FEATURE_COLUMNS_V5

VOLATILITY_PROTOCOL_VERSION = "global-volatility-distribution-v5"
MODEL_ARCHITECTURE_VERSION = "baseline-residual-tcn-v2"
TARGET_VERSION = "future-rv-total-v1"
DEFAULT_HORIZONS = (1, 3, 5, 7, 14, 30)


@dataclass(frozen=True)
class VolatilityForecastProtocol:
    """Immutable model and evaluation settings for one research run."""

    protocol_version: str = VOLATILITY_PROTOCOL_VERSION
    architecture_version: str = MODEL_ARCHITECTURE_VERSION
    target_version: str = TARGET_VERSION
    horizons: tuple[int, ...] = DEFAULT_HORIZONS
    feature_names: tuple[str, ...] = DEPLOYABLE_FEATURE_COLUMNS_V5
    window_size: int = 60
    folds: int = 5
    embargo_sessions: int = 30
    minimum_train_sessions: int = 756
    validation_sessions: int = 126
    early_stopping_sessions: int = 63
    temporal_holdout_sessions: int = 252
    asset_holdout_fraction: float = 0.20
    seeds: tuple[int, ...] = (41, 42, 43)
    realized_variance_proxy: str = "overnight_plus_rogers_satchell"
    baseline_family: str = "causal_log_har"

    def __post_init__(self) -> None:
        if not self.horizons or tuple(sorted(set(self.horizons))) != self.horizons:
            raise ValueError("horizons must be non-empty, unique, and increasing")
        if self.window_size < 22:
            raise ValueError("window_size must cover at least one HAR monthly window")
        if self.embargo_sessions < max(self.horizons):
            raise ValueError("embargo_sessions must be at least the maximum forecast horizon")
        if self.folds < 3:
            raise ValueError("at least three expanding folds are required")
        if self.early_stopping_sessions < 5:
            raise ValueError("early_stopping_sessions must contain at least five sessions")

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)


@dataclass(frozen=True)
class VolatilityPromotionGate:
    """Conservative initial promotion defaults; the frozen run records them.

    These values are not universal claims. They are initial guardrails which
    may only change before certification, with a new protocol version and a
    complete benchmark rerun.
    """

    maximum_relative_qlike: float = 0.98
    maximum_relative_variance_only_crps: float = 0.99
    maximum_relative_return_mae: float = 0.99
    maximum_relative_return_rmse: float = 0.99
    minimum_return_folds_beating_baseline: int = 4
    minimum_folds_beating_baseline: int = 4
    maximum_worst_fold_relative_qlike: float = 1.10
    minimum_interval_coverage_80: float = 0.74
    maximum_interval_coverage_80: float = 0.86
    maximum_subgroup_relative_qlike: float = 1.05
    significance_level: float = 0.05

    def __post_init__(self) -> None:
        if not 0 < self.maximum_relative_qlike < 1:
            raise ValueError("relative QLIKE gate must require an improvement")
        if not 0 < self.maximum_relative_variance_only_crps < 1:
            raise ValueError("variance-only CRPS gate must require an improvement")
        if not 0 < self.maximum_relative_return_mae < 1:
            raise ValueError("return MAE gate must require an improvement")
        if not 0 < self.maximum_relative_return_rmse < 1:
            raise ValueError("return RMSE gate must require an improvement")
        if not 0 < self.minimum_interval_coverage_80 < self.maximum_interval_coverage_80 < 1:
            raise ValueError("invalid 80% interval coverage bounds")
