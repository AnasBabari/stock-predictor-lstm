"""Metrics that preserve forecast origins and horizons.

Every row represents one forecast origin. Columns represent explicit horizons.
This avoids treating a flattened multi-horizon matrix as one chronological
series, which would create invalid direction comparisons between unrelated
forecast origins.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import r2_score


def _one_dimensional(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")
    return array


def _same_shape(actual, predicted) -> tuple[np.ndarray, np.ndarray]:
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    if actual_array.shape != predicted_array.shape:
        raise ValueError("Actual and predicted values must have identical shapes.")
    if actual_array.size == 0:
        raise ValueError("Forecast arrays must not be empty.")
    if not np.isfinite(actual_array).all() or not np.isfinite(predicted_array).all():
        raise ValueError("Forecast arrays contain non-finite values.")
    return actual_array, predicted_array


def _direction_accuracy(actual: np.ndarray, predicted: np.ndarray, origin: np.ndarray) -> float:
    actual_direction = np.sign(actual - origin)
    predicted_direction = np.sign(predicted - origin)
    return float(np.mean(actual_direction == predicted_direction))


def regression_metrics(
    actual,
    predicted,
    *,
    origin=None,
    scale_series=None,
    persistence=None,
) -> dict[str, float | None]:
    """Evaluate one forecast horizon in the target's original units.

    ``origin`` is the price known when the forecast was made and is required
    for valid directional accuracy. ``scale_series`` must contain training-only
    observations when MASE/RMSSE are requested.
    """

    actual_array, predicted_array = _same_shape(actual, predicted)
    actual_array = actual_array.reshape(-1)
    predicted_array = predicted_array.reshape(-1)
    errors = actual_array - predicted_array
    absolute_errors = np.abs(errors)
    squared_errors = errors**2

    nonzero = actual_array != 0
    mape = (
        float(np.mean(np.abs(errors[nonzero] / actual_array[nonzero])) * 100)
        if np.any(nonzero)
        else None
    )
    r2 = float(r2_score(actual_array, predicted_array)) if len(actual_array) > 1 else None

    direction_accuracy = None
    if origin is not None:
        origin_array = _one_dimensional(origin, "origin")
        if origin_array.shape != actual_array.shape:
            raise ValueError("Origin values must match the forecast observations.")
        direction_accuracy = _direction_accuracy(actual_array, predicted_array, origin_array)

    mase = None
    rmsse = None
    if scale_series is not None:
        scale_array = _one_dimensional(scale_series, "scale_series")
        if len(scale_array) < 2:
            raise ValueError("scale_series needs at least two observations.")
        scale_differences = np.diff(scale_array)
        mae_scale = float(np.mean(np.abs(scale_differences)))
        mse_scale = float(np.mean(scale_differences**2))
        if mae_scale > 0:
            mase = float(np.mean(absolute_errors) / mae_scale)
        if mse_scale > 0:
            rmsse = float(np.sqrt(np.mean(squared_errors) / mse_scale))

    relative_mae = None
    relative_rmse = None
    if persistence is not None:
        persistence_array = _one_dimensional(persistence, "persistence")
        if persistence_array.shape != actual_array.shape:
            raise ValueError("Persistence values must match the forecast observations.")
        persistence_errors = actual_array - persistence_array
        persistence_mae = float(np.mean(np.abs(persistence_errors)))
        persistence_rmse = float(np.sqrt(np.mean(persistence_errors**2)))
        if persistence_mae > 0:
            relative_mae = float(np.mean(absolute_errors) / persistence_mae)
        if persistence_rmse > 0:
            relative_rmse = float(np.sqrt(np.mean(squared_errors)) / persistence_rmse)

    return {
        "mae": float(np.mean(absolute_errors)),
        "median_absolute_error": float(np.median(absolute_errors)),
        "mse": float(np.mean(squared_errors)),
        "rmse": float(np.sqrt(np.mean(squared_errors))),
        "mape": mape,
        "smape": float(
            np.mean(
                np.divide(
                    200 * absolute_errors,
                    np.abs(actual_array) + np.abs(predicted_array),
                    out=np.zeros_like(absolute_errors),
                    where=(np.abs(actual_array) + np.abs(predicted_array)) > 0,
                )
            )
        ),
        "bias": float(np.mean(errors)),
        "r2": r2,
        "direction_accuracy": direction_accuracy,
        "mase": mase,
        "rmsse": rmsse,
        "relative_mae": relative_mae,
        "relative_rmse": relative_rmse,
        "sample_count": int(len(actual_array)),
    }


def pinball_loss(actual, quantile_pred, tau) -> float:
    """Return the quantile (pinball) loss of a quantile forecast.

    The loss for each observation is ``max(tau * (y - q), (tau - 1) * (y - q))``
    where ``y`` is the realised value and ``q`` the quantile forecast. At
    ``tau = 0.5`` this equals half the mean absolute error. Actual and
    quantile predictions must be finite, non-empty, aligned arrays.
    """

    actual_array, quantile_array = _same_shape(actual, quantile_pred)
    try:
        tau_value = float(tau)
    except (TypeError, ValueError) as exc:
        raise ValueError("tau must be a numeric value.") from exc
    if not np.isfinite(tau_value) or not 0.0 < tau_value < 1.0:
        raise ValueError("tau must lie strictly between zero and one.")
    errors = actual_array - quantile_array
    losses = np.maximum(tau_value * errors, (tau_value - 1.0) * errors)
    return float(np.mean(losses))


def evaluate_forecast_horizons(
    actual,
    predicted,
    origins,
    *,
    horizons: Sequence[int] | None = None,
    scale_series=None,
) -> dict:
    """Return per-horizon and pooled metrics for an ``(origins, horizons)`` matrix."""

    actual_array, predicted_array = _same_shape(actual, predicted)
    if actual_array.ndim == 1:
        actual_array = actual_array.reshape(-1, 1)
        predicted_array = predicted_array.reshape(-1, 1)
    if actual_array.ndim != 2:
        raise ValueError("Multi-horizon forecasts must be a one- or two-dimensional array.")

    origin_array = _one_dimensional(origins, "origins")
    if len(origin_array) != actual_array.shape[0]:
        raise ValueError("One origin value is required for each forecast row.")

    if horizons is None:
        horizon_values = list(range(1, actual_array.shape[1] + 1))
    else:
        horizon_values = [int(value) for value in horizons]
        if len(horizon_values) != actual_array.shape[1]:
            raise ValueError("The number of horizons must match the forecast columns.")
        if len(set(horizon_values)) != len(horizon_values) or any(
            value < 1 for value in horizon_values
        ):
            raise ValueError("Horizons must be unique positive integers.")

    per_horizon = {}
    for column, horizon in enumerate(horizon_values):
        per_horizon[str(horizon)] = regression_metrics(
            actual_array[:, column],
            predicted_array[:, column],
            origin=origin_array,
            scale_series=scale_series,
            persistence=origin_array,
        )

    pooled_actual = actual_array.reshape(-1)
    pooled_predicted = predicted_array.reshape(-1)
    pooled_origins = np.repeat(origin_array, actual_array.shape[1])
    pooled = regression_metrics(
        pooled_actual,
        pooled_predicted,
        origin=pooled_origins,
        scale_series=scale_series,
        persistence=pooled_origins,
    )
    pooled["metric_scope"] = "forecast_origin_horizon_pairs"

    return {
        "horizons": horizon_values,
        "per_horizon": per_horizon,
        "pooled": pooled,
    }


def evaluate_probability_forecast(
    actual, probabilities, *, training_targets, calibration_bins: int = 10
) -> dict[str, Any]:
    """Evaluate a binary probability forecast against a training-only majority baseline."""

    actual_array = np.asarray(actual, dtype=int).reshape(-1)
    probability_array = np.asarray(probabilities, dtype=float).reshape(-1)
    training_array = np.asarray(training_targets, dtype=int).reshape(-1)
    if actual_array.shape != probability_array.shape or actual_array.size == 0:
        raise ValueError("Actual labels and probabilities must be non-empty and aligned.")
    if calibration_bins < 2:
        raise ValueError("calibration_bins must be at least two.")
    if not set(np.unique(actual_array)).issubset({0, 1}) or not set(
        np.unique(training_array)
    ).issubset({0, 1}):
        raise ValueError("Direction labels must be binary.")
    if not np.isfinite(probability_array).all() or np.any(
        (probability_array < 0) | (probability_array > 1)
    ):
        raise ValueError("Probabilities must be finite values between zero and one.")

    predictions = (probability_array >= 0.5).astype(int)
    accuracy = float(np.mean(predictions == actual_array))
    majority = int(np.bincount(training_array, minlength=2).argmax())
    majority_accuracy = float(np.mean(actual_array == majority))
    positive_recall = (
        float(np.mean(predictions[actual_array == 1] == 1)) if np.any(actual_array == 1) else 0.0
    )
    negative_recall = (
        float(np.mean(predictions[actual_array == 0] == 0)) if np.any(actual_array == 0) else 0.0
    )
    clipped = np.clip(probability_array, 1e-7, 1 - 1e-7)
    # Equal-width bins are deliberately deterministic and retain empty bins as
    # omitted rather than pretending they carry calibration evidence.
    bin_index = np.minimum((probability_array * calibration_bins).astype(int), calibration_bins - 1)
    reliability_bins = []
    expected_calibration_error = 0.0
    for index in range(calibration_bins):
        mask = bin_index == index
        if not np.any(mask):
            continue
        confidence = float(np.mean(probability_array[mask]))
        observed = float(np.mean(actual_array[mask]))
        weight = float(np.mean(mask))
        expected_calibration_error += weight * abs(confidence - observed)
        reliability_bins.append(
            {
                "bin": index,
                "count": int(np.sum(mask)),
                "confidence": confidence,
                "observed_rate": observed,
            }
        )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": (positive_recall + negative_recall) / 2,
        "majority_baseline": majority_accuracy,
        "brier_score": float(np.mean((probability_array - actual_array) ** 2)),
        "log_loss": float(
            -np.mean(actual_array * np.log(clipped) + (1 - actual_array) * np.log(1 - clipped))
        ),
        "expected_calibration_error": expected_calibration_error,
        "reliability_bins": reliability_bins,
        "sample_count": int(len(actual_array)),
    }
