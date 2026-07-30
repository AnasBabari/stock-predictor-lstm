"""Shared, immutable dataset and walk-forward fold contracts for experiments.

These contracts deliberately contain raw values only.  Candidate implementations
must fit any transforms inside the training partition described by ``FoldPlan``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from evaluation.splits import generate_walk_forward_splits
from experiments.targets import TargetType, build_supervised_dataset


@dataclass(frozen=True)
class ExperimentDataset:
    """A direct-horizon dataset shared unchanged by every experiment candidate."""

    features: np.ndarray
    targets: np.ndarray
    actual_prices: np.ndarray
    origins: np.ndarray
    origin_indices: np.ndarray
    origin_dates: pd.DatetimeIndex
    feature_names: tuple[str, ...]
    horizons: tuple[int, ...]
    target_type: TargetType
    snapshot_id: str | None = None

    def __post_init__(self) -> None:
        samples = len(self.features)
        arrays = (
            self.targets,
            self.actual_prices,
            self.origins,
            self.origin_indices,
            self.origin_dates,
        )
        if not samples or any(len(value) != samples for value in arrays):
            raise ValueError("Experiment dataset arrays must have the same non-zero sample count.")
        if self.features.ndim != 3 or self.targets.ndim != 2:
            raise ValueError("Experiment features must be 3D and targets must be 2D.")
        if self.targets.shape[1] != len(self.horizons):
            raise ValueError("Every target row must contain every requested horizon.")
        if self.features.shape[2] != len(self.feature_names):
            raise ValueError("feature_names must match the experiment feature matrix.")
        if not np.isfinite(self.features).all() or not np.isfinite(self.targets).all():
            raise ValueError("Experiment dataset values must be finite.")


def build_experiment_dataset(
    feature_values,
    close_values,
    *,
    dates,
    feature_names,
    lookback: int,
    horizons: tuple[int, ...] = (1, 5, 20),
    target_type: TargetType = "log_return",
    snapshot_id: str | None = None,
) -> ExperimentDataset:
    """Build the canonical raw experiment dataset from a single market snapshot."""

    names = tuple(str(name) for name in feature_names)
    date_index = pd.DatetimeIndex(dates)
    if len(date_index) != len(close_values):
        raise ValueError("dates must align with close_values.")
    supervised = build_supervised_dataset(
        feature_values,
        close_values,
        lookback=lookback,
        horizons=horizons,
        target_type=target_type,
    )
    return ExperimentDataset(
        features=supervised.features,
        targets=supervised.targets,
        actual_prices=supervised.actual_prices,
        origins=supervised.origins,
        origin_indices=supervised.origin_indices,
        origin_dates=date_index[supervised.origin_indices],
        feature_names=names,
        horizons=supervised.horizons,
        target_type=supervised.target_type,
        snapshot_id=snapshot_id,
    )


@dataclass(frozen=True)
class Fold:
    """One leakage-safe, chronological outer fold expressed in sample indices."""

    number: int
    training_indices: np.ndarray
    validation_indices: np.ndarray


@dataclass(frozen=True)
class FoldPlan:
    """The one fold definition that every candidate must receive."""

    folds: tuple[Fold, ...]
    gap: int
    method: Literal["expanding", "rolling"]
    min_train_size: int
    validation_size: int
    maximum_horizon: int

    @classmethod
    def create(
        cls,
        dataset: ExperimentDataset,
        *,
        folds: int = 5,
        min_train_size: int = 300,
        validation_size: int = 60,
        gap: int | None = None,
        method: Literal["expanding", "rolling"] = "expanding",
    ) -> FoldPlan:
        effective_gap = max(dataset.horizons) if gap is None else gap
        if effective_gap < max(dataset.horizons):
            raise ValueError("Fold gap must be at least the maximum forecast horizon.")
        raw_folds = generate_walk_forward_splits(
            len(dataset.features),
            folds=folds,
            min_train_size=min_train_size,
            validation_size=validation_size,
            gap=effective_gap,
            method=method,
        )
        plan_folds = tuple(
            Fold(number=index, training_indices=training, validation_indices=validation)
            for index, (training, validation) in enumerate(raw_folds, start=1)
        )
        for fold in plan_folds:
            last_training_origin = dataset.origin_indices[fold.training_indices[-1]]
            first_validation_origin = dataset.origin_indices[fold.validation_indices[0]]
            if last_training_origin + max(dataset.horizons) >= first_validation_origin:
                raise ValueError("Fold plan permits overlapping training and validation targets.")
        return cls(
            folds=plan_folds,
            gap=effective_gap,
            method=method,
            min_train_size=min_train_size,
            validation_size=validation_size,
            maximum_horizon=max(dataset.horizons),
        )
