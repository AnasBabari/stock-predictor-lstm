"""Immutable research policy and the local RTX 2060 resource profile."""

from __future__ import annotations

from dataclasses import dataclass


HORIZONS = (1, 3, 5, 10, 20)
DEFAULT_WINDOW = 60
PURGE_VERSION = "purged-expanding-v1"


@dataclass(frozen=True)
class RuntimeBudget:
    """Hard limits for one experiment process on the user's RTX 2060 6 GB."""

    name: str = "rtx2060-6gb"
    vram_warning_mb: int = 5_200
    vram_kill_mb: int = 5_500
    screen_seconds: int = 120
    confirm_seconds: int = 1_200
    smoke_seconds: int = 30
    max_parameters: int = 20_000_000


@dataclass(frozen=True)
class EvaluationPolicy:
    window: int = DEFAULT_WINDOW
    horizons: tuple[int, ...] = HORIZONS
    folds: int = 5
    minimum_train_rows: int = 300
    minimum_validation_rows: int = 60
    seed_count: int = 3
    purge_version: str = PURGE_VERSION
    relative_error_gate: float = 0.98
    worst_fold_gate: float = 1.25


RUNTIME_BUDGET = RuntimeBudget()
EVALUATION_POLICY = EvaluationPolicy()
