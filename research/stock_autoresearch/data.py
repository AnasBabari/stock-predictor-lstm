"""Snapshot loading and target construction.

The evaluator receives an already frozen table. It deliberately has no network
client and never mutates the supplied frame. Snapshot creation belongs to the
backend feature pipeline and is a separate, reviewed operation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Snapshot:
    frame: pd.DataFrame
    snapshot_id: str
    feature_names: tuple[str, ...]


def validate_snapshot(snapshot: Snapshot) -> None:
    frame = snapshot.frame
    if frame.empty:
        raise ValueError("research snapshot is empty")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("snapshot index must be unique and strictly ordered")
    missing = [name for name in snapshot.feature_names if name not in frame.columns]
    if missing:
        raise ValueError(f"snapshot is missing features: {missing}")
    if "Close" not in frame.columns:
        raise ValueError("snapshot must contain Close for target reconstruction")
    values = frame[list(snapshot.feature_names) + ["Close"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("snapshot contains non-finite values")
    if (frame["Close"].to_numpy(dtype=float) <= 0).any():
        raise ValueError("Close must be positive")


def build_examples(
    snapshot: Snapshot,
    *,
    window: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build windows and cumulative log-return labels without future leakage."""

    validate_snapshot(snapshot)
    if window < 1 or horizon < 1:
        raise ValueError("window and horizon must be positive")
    features = snapshot.frame[list(snapshot.feature_names)].to_numpy(dtype=np.float64)
    close = snapshot.frame["Close"].to_numpy(dtype=np.float64)
    end = len(close) - horizon
    if end <= window:
        raise ValueError("snapshot does not contain enough rows")
    x = np.stack([features[i - window : i] for i in range(window, end)])
    y = np.log(close[window + horizon : window + horizon + len(x)] / close[window : window + len(x)])
    origins = np.arange(window, window + len(x), dtype=np.int64)
    return x, y, origins


def expanding_folds(
    n_rows: int,
    *,
    folds: int,
    minimum_train_rows: int,
    validation_rows: int,
    purge: int,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """Yield ordered train/validation indices with an explicit purge gap."""

    if folds < 1 or minimum_train_rows < 1 or validation_rows < 1:
        raise ValueError("invalid fold policy")
    usable = n_rows - minimum_train_rows - purge
    if usable < folds * validation_rows:
        raise ValueError("not enough rows for requested expanding folds")
    for fold in range(folds):
        validation_start = minimum_train_rows + purge + fold * validation_rows
        validation_end = validation_start + validation_rows
        train_end = validation_start - purge
        yield np.arange(0, train_end), np.arange(validation_start, validation_end)
