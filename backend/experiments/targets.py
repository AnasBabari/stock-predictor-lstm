"""Leakage-safe direct-horizon target construction for offline experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

TargetType = Literal["price_level", "simple_return", "log_return", "persistence_residual"]


@dataclass(frozen=True)
class SupervisedDataset:
    features: np.ndarray
    targets: np.ndarray
    actual_prices: np.ndarray
    origins: np.ndarray
    origin_indices: np.ndarray
    horizons: tuple[int, ...]
    target_type: TargetType


def _validate_horizons(horizons) -> tuple[int, ...]:
    values = tuple(int(value) for value in horizons)
    if not values or len(set(values)) != len(values) or any(value < 1 for value in values):
        raise ValueError("Horizons must be unique positive integers.")
    return values


def transform_price_targets(origins, future_prices, target_type: TargetType) -> np.ndarray:
    """Transform future prices while retaining an exact inverse for display."""

    origin_array = np.asarray(origins, dtype=float).reshape(-1)
    future_array = np.asarray(future_prices, dtype=float)
    if future_array.ndim == 1:
        future_array = future_array.reshape(-1, 1)
    if future_array.shape[0] != len(origin_array):
        raise ValueError("One origin is required for every future-price row.")
    if not np.isfinite(origin_array).all() or not np.isfinite(future_array).all():
        raise ValueError("Target prices must be finite.")
    if target_type == "price_level":
        return future_array.copy()
    if target_type == "simple_return":
        if np.any(origin_array == 0):
            raise ValueError("Simple returns require non-zero origin prices.")
        return future_array / origin_array[:, None] - 1
    if target_type == "log_return":
        if np.any(origin_array <= 0) or np.any(future_array <= 0):
            raise ValueError("Log returns require positive prices.")
        return np.log(future_array / origin_array[:, None])
    if target_type == "persistence_residual":
        return future_array - origin_array[:, None]
    raise ValueError(f"Unsupported target type: {target_type}")


def reconstruct_prices(origins, targets, target_type: TargetType) -> np.ndarray:
    """Convert model outputs back to price units."""

    origin_array = np.asarray(origins, dtype=float).reshape(-1)
    target_array = np.asarray(targets, dtype=float)
    if target_array.ndim == 1:
        target_array = target_array.reshape(-1, 1)
    if target_array.shape[0] != len(origin_array):
        raise ValueError("One origin is required for every target row.")
    if target_type == "price_level":
        return target_array.copy()
    if target_type == "simple_return":
        return origin_array[:, None] * (1 + target_array)
    if target_type == "log_return":
        return origin_array[:, None] * np.exp(target_array)
    if target_type == "persistence_residual":
        return origin_array[:, None] + target_array
    raise ValueError(f"Unsupported target type: {target_type}")


def build_supervised_dataset(
    feature_values,
    close_values,
    *,
    lookback: int,
    horizons=(1, 5, 20),
    target_type: TargetType = "log_return",
) -> SupervisedDataset:
    """Create direct-horizon samples whose feature window ends at the origin."""

    feature_array = np.asarray(feature_values, dtype=float)
    close_array = np.asarray(close_values, dtype=float).reshape(-1)
    horizon_values = _validate_horizons(horizons)
    if feature_array.ndim != 2 or len(feature_array) != len(close_array):
        raise ValueError("Features must be a 2D array aligned with close prices.")
    if lookback < 2:
        raise ValueError("lookback must be at least two observations.")
    if not np.isfinite(feature_array).all() or not np.isfinite(close_array).all():
        raise ValueError("Supervised input data contains non-finite values.")

    maximum_horizon = max(horizon_values)
    first_origin = lookback - 1
    last_origin = len(close_array) - maximum_horizon - 1
    if last_origin < first_origin:
        raise ValueError("Not enough observations for the lookback and forecast horizons.")

    origin_indices = np.arange(first_origin, last_origin + 1)
    feature_windows = np.asarray(
        [feature_array[index - lookback + 1 : index + 1] for index in origin_indices]
    )
    origins = close_array[origin_indices]
    actual_prices = np.column_stack(
        [close_array[origin_indices + horizon] for horizon in horizon_values]
    )
    targets = transform_price_targets(origins, actual_prices, target_type)

    return SupervisedDataset(
        features=feature_windows,
        targets=targets,
        actual_prices=actual_prices,
        origins=origins,
        origin_indices=origin_indices,
        horizons=horizon_values,
        target_type=target_type,
    )
