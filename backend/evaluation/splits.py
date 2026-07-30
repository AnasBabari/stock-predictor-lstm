"""Chronological validation splits with explicit purge boundaries."""

from __future__ import annotations

import numpy as np


def generate_walk_forward_splits(
    n_rows: int,
    *,
    folds: int,
    min_train_size: int,
    validation_size: int,
    gap: int = 0,
    method: str = "expanding",
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate deterministic expanding or rolling walk-forward splits."""

    if folds < 1 or min_train_size < 1 or validation_size < 1 or gap < 0:
        raise ValueError("Fold, training, validation, and gap settings are invalid.")
    if method not in {"expanding", "rolling"}:
        raise ValueError("method must be 'expanding' or 'rolling'.")
    required = min_train_size + gap + folds * validation_size
    if n_rows < required:
        raise ValueError(f"Not enough rows for validation: need {required}, received {n_rows}.")

    first_validation = n_rows - folds * validation_size
    splits = []
    for fold in range(folds):
        validation_start = first_validation + fold * validation_size
        training_end = validation_start - gap
        training_start = 0 if method == "expanding" else training_end - min_train_size
        training = np.arange(training_start, training_end)
        validation = np.arange(validation_start, validation_start + validation_size)
        if len(training) < min_train_size:
            raise ValueError("Validation fold violates min_train_size.")
        splits.append((training, validation))
    return splits


def purged_tail_split(
    n_samples: int,
    *,
    validation_fraction: float = 0.1,
    purge: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a training fold for early stopping without overlapping target windows."""

    if n_samples < 3:
        raise ValueError("At least three samples are required.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one.")
    if purge < 0:
        raise ValueError("purge must not be negative.")

    validation_size = max(1, int(n_samples * validation_fraction))
    validation_start = n_samples - validation_size
    training_end = validation_start - purge
    if training_end < 1:
        raise ValueError("Purge leaves no fitting observations.")
    return np.arange(0, training_end), np.arange(validation_start, n_samples)
