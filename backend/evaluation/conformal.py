"""Split-conformal prediction intervals calibrated only on held-out residuals."""

from __future__ import annotations

import numpy as np


def calibrate_intervals(actual, predicted, *, coverages=(0.80, 0.95)) -> dict:
    """Fit symmetric per-horizon conformal radii from calibration residuals."""
    observed = np.asarray(actual, dtype=float)
    forecast = np.asarray(predicted, dtype=float)
    if observed.shape != forecast.shape or observed.ndim not in (1, 2) or observed.size < 2:
        raise ValueError(
            "actual and predicted must be matching non-empty one or two-dimensional arrays."
        )
    if not np.isfinite(observed).all() or not np.isfinite(forecast).all():
        raise ValueError("Conformal inputs must be finite.")
    if observed.ndim == 1:
        observed, forecast = observed[:, None], forecast[:, None]
    normalized_coverages = tuple(float(value) for value in coverages)
    if not normalized_coverages or any(not 0 < value < 1 for value in normalized_coverages):
        raise ValueError("coverages must be values in (0, 1).")
    residuals = np.abs(observed - forecast)
    # Finite-sample conformal rank: ceil((n + 1) * coverage) / n.
    radii = {}
    for coverage in normalized_coverages:
        rank = min(int(np.ceil((len(residuals) + 1) * coverage)), len(residuals))
        radii[str(coverage)] = np.sort(residuals, axis=0)[rank - 1].tolist()
    return {
        "method": "split_conformal_absolute_residual",
        "calibration_count": int(len(residuals)),
        "radii": radii,
    }


def prediction_intervals(predicted, calibration: dict, *, coverage: float = 0.95) -> dict:
    forecast = np.asarray(predicted, dtype=float)
    if forecast.ndim == 1:
        forecast = forecast[:, None]
    if forecast.ndim != 2 or not np.isfinite(forecast).all():
        raise ValueError("predicted must be a finite one- or two-dimensional array.")
    try:
        radius = np.asarray(calibration["radii"][str(float(coverage))], dtype=float)
    except (KeyError, TypeError) as exc:
        raise ValueError("Calibration does not contain the requested coverage.") from exc
    if radius.shape != (forecast.shape[1],):
        raise ValueError("Calibration horizon count does not match predictions.")
    return {"lower": forecast - radius, "upper": forecast + radius, "coverage": float(coverage)}


def interval_diagnostics(actual, intervals: dict) -> dict[str, float]:
    observed = np.asarray(actual, dtype=float)
    lower, upper = (
        np.asarray(intervals["lower"], dtype=float),
        np.asarray(intervals["upper"], dtype=float),
    )
    if observed.shape != lower.shape or observed.shape != upper.shape:
        raise ValueError("Actual and interval arrays must have the same shape.")
    return {
        "empirical_coverage": float(np.mean((observed >= lower) & (observed <= upper))),
        "average_width": float(np.mean(upper - lower)),
        "nominal_coverage": float(intervals["coverage"]),
    }
