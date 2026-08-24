"""Proper scoring rules and calibration metrics for volatility distributions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

_EPSILON = 1e-12
_NORMAL_INTERVAL_Z = {
    "50": 0.6744897501960817,
    "80": 1.2815515655446004,
    "95": 1.959963984540054,
}


@dataclass(frozen=True)
class DistributionPredictions:
    variance: np.ndarray
    return_location: np.ndarray
    direction_probabilities: np.ndarray
    return_variance: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.variance.ndim != 2 or self.return_location.shape != self.variance.shape:
            raise ValueError("variance and return location must be [rows, horizons]")
        if self.direction_probabilities.shape != (*self.variance.shape, 3):
            raise ValueError("direction probabilities must be [rows, horizons, 3]")
        if self.return_variance is None:
            object.__setattr__(self, "return_variance", self.variance)
        if self.return_variance.shape != self.variance.shape:
            raise ValueError("return variance must match realized-variance forecast shape")
        if not (
            np.isfinite(self.variance).all()
            and np.isfinite(self.return_location).all()
            and np.isfinite(self.direction_probabilities).all()
            and np.isfinite(self.return_variance).all()
        ):
            raise ValueError("predictions contain non-finite values")
        if (self.variance <= 0).any() or (self.return_variance <= 0).any():
            raise ValueError("realized and return forecast variances must be positive")
        if not np.allclose(self.direction_probabilities.sum(axis=-1), 1.0, atol=1e-5):
            raise ValueError("direction probabilities must sum to one")


def qlike_losses(forecast_variance: np.ndarray, realized_variance: np.ndarray) -> np.ndarray:
    forecast = np.maximum(np.asarray(forecast_variance, dtype=np.float64), _EPSILON)
    realized = np.maximum(np.asarray(realized_variance, dtype=np.float64), _EPSILON)
    if forecast.shape != realized.shape:
        raise ValueError("QLIKE arrays must have matching shapes")
    ratio = realized / forecast
    return ratio - np.log(ratio) - 1.0


def gaussian_crps(
    observation: np.ndarray,
    location: np.ndarray,
    variance: np.ndarray,
) -> np.ndarray:
    """Closed-form continuous ranked probability score for a Gaussian."""
    observed = np.asarray(observation, dtype=np.float64)
    mean = np.asarray(location, dtype=np.float64)
    sigma = np.sqrt(np.maximum(np.asarray(variance, dtype=np.float64), _EPSILON))
    if observed.shape != mean.shape or mean.shape != sigma.shape:
        raise ValueError("CRPS arrays must have matching shapes")
    z = (observed - mean) / sigma
    density = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
    return sigma * (z * (2.0 * ndtr(z) - 1.0) + 2.0 * density - 1.0 / np.sqrt(np.pi))


def gaussian_nll(
    observation: np.ndarray,
    location: np.ndarray,
    variance: np.ndarray,
) -> np.ndarray:
    observed = np.asarray(observation, dtype=np.float64)
    mean = np.asarray(location, dtype=np.float64)
    var = np.maximum(np.asarray(variance, dtype=np.float64), _EPSILON)
    return 0.5 * (np.log(2.0 * np.pi * var) + (observed - mean) ** 2 / var)


def fit_crps_variance_scale(
    forecast_variance: np.ndarray,
    cumulative_returns: np.ndarray,
    *,
    minimum_scale: float = 0.25,
    maximum_scale: float = 4.0,
    grid_points: int = 81,
) -> np.ndarray:
    """Fit one positive CRPS scale per horizon on a pre-evaluation set."""
    variance = np.asarray(forecast_variance, dtype=np.float64)
    returns = np.asarray(cumulative_returns, dtype=np.float64)
    if variance.shape != returns.shape or variance.ndim != 2 or len(variance) == 0:
        raise ValueError("variance calibration arrays must be matched non-empty matrices")
    if not 0 < minimum_scale < maximum_scale or grid_points < 3:
        raise ValueError("invalid variance calibration grid")
    grid = np.exp(np.linspace(np.log(minimum_scale), np.log(maximum_scale), grid_points))
    scales = np.empty(variance.shape[1], dtype=np.float64)
    zero = np.zeros(len(variance), dtype=np.float64)
    for column in range(variance.shape[1]):
        scores = [
            float(np.mean(gaussian_crps(returns[:, column], zero, variance[:, column] * factor)))
            for factor in grid
        ]
        scales[column] = grid[int(np.argmin(scores))]
    return scales


def fit_qlike_variance_scale(
    forecast_variance: np.ndarray,
    realized_variance: np.ndarray,
    *,
    session_labels: np.ndarray | None = None,
    minimum_scale: float = 0.25,
    maximum_scale: float = 4.0,
) -> np.ndarray:
    """Fit one multiplicative QLIKE calibration per horizon.

    For positive forecast ``f`` and realization ``y``, the QLIKE-optimal
    multiplicative scale is mean(y / f).  The caller supplies only a
    pre-evaluation calibration region; clipping prevents unstable extremes.
    """
    forecast = np.asarray(forecast_variance, dtype=np.float64)
    realized = np.asarray(realized_variance, dtype=np.float64)
    if forecast.ndim != 2 or forecast.shape != realized.shape or len(forecast) < 5:
        raise ValueError("QLIKE calibration requires matched [rows, horizons] arrays")
    if not 0 < minimum_scale <= maximum_scale:
        raise ValueError("QLIKE calibration scale bounds are invalid")
    if not (
        np.isfinite(forecast).all()
        and np.isfinite(realized).all()
        and (forecast > 0).all()
        and (realized > 0).all()
    ):
        raise ValueError("QLIKE calibration requires finite positive variances")
    ratios = realized / forecast
    if session_labels is None:
        scales = np.mean(ratios, axis=0)
    else:
        labels = np.asarray(session_labels)
        if labels.ndim != 1 or len(labels) != len(forecast):
            raise ValueError("QLIKE calibration session labels must match rows")
        sessions, inverse = np.unique(labels, return_inverse=True)
        if len(sessions) < 5:
            raise ValueError("QLIKE calibration requires at least five sessions")
        totals = np.zeros((len(sessions), forecast.shape[1]), dtype=np.float64)
        counts = np.zeros(len(sessions), dtype=np.int64)
        np.add.at(totals, inverse, ratios)
        np.add.at(counts, inverse, 1)
        scales = np.mean(totals / counts[:, None], axis=0)
    return np.clip(scales, minimum_scale, maximum_scale)


def _safe_ratio(candidate: float, baseline: float) -> float:
    return float(candidate / baseline) if np.isfinite(baseline) and baseline > 0 else float("nan")


def _multiclass_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    probs = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    one_hot = np.eye(3, dtype=np.float64)[truth]
    predicted = np.argmax(probs, axis=1)
    recalls: list[float] = []
    f1_scores: list[float] = []
    for class_index in range(3):
        true_positive = np.sum((predicted == class_index) & (truth == class_index))
        false_positive = np.sum((predicted == class_index) & (truth != class_index))
        false_negative = np.sum((predicted != class_index) & (truth == class_index))
        recall_denominator = true_positive + false_negative
        precision_denominator = true_positive + false_positive
        recall = true_positive / recall_denominator if recall_denominator else 0.0
        precision = true_positive / precision_denominator if precision_denominator else 0.0
        recalls.append(float(recall))
        f1_scores.append(
            float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        )
    clipped = np.clip(probs[np.arange(len(truth)), truth], 1e-12, 1.0)
    return {
        "direction_accuracy": float(np.mean(predicted == truth)),
        "direction_balanced_accuracy": float(np.mean(recalls)),
        "direction_macro_f1": float(np.mean(f1_scores)),
        "direction_brier": float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))),
        "direction_log_loss": float(-np.mean(np.log(clipped))),
    }


def horizon_distribution_metrics(
    *,
    predictions: DistributionPredictions,
    baseline_variance: np.ndarray,
    baseline_return_variance: np.ndarray | None = None,
    realized_variance: np.ndarray,
    cumulative_returns: np.ndarray,
    direction_classes: np.ndarray,
    horizons: tuple[int, ...],
) -> list[dict[str, float | int]]:
    """Calculate candidate and matched-baseline evidence per horizon."""
    baseline = np.asarray(baseline_variance, dtype=np.float64)
    baseline_distribution = np.asarray(
        baseline_return_variance if baseline_return_variance is not None else baseline_variance,
        dtype=np.float64,
    )
    realized = np.asarray(realized_variance, dtype=np.float64)
    returns = np.asarray(cumulative_returns, dtype=np.float64)
    classes = np.asarray(direction_classes, dtype=np.int64)
    expected_shape = predictions.variance.shape
    if any(
        values.shape != expected_shape
        for values in (baseline, baseline_distribution, realized, returns, classes)
    ):
        raise ValueError("target and baseline arrays must match prediction shape")
    if len(horizons) != expected_shape[1]:
        raise ValueError("horizon count does not match prediction columns")

    rows: list[dict[str, float | int]] = []
    for column, horizon in enumerate(horizons):
        model_qlike = qlike_losses(predictions.variance[:, column], realized[:, column])
        baseline_qlike = qlike_losses(baseline[:, column], realized[:, column])
        full_model_crps = gaussian_crps(
            returns[:, column],
            predictions.return_location[:, column],
            predictions.return_variance[:, column],
        )
        variance_only_crps = gaussian_crps(
            returns[:, column],
            np.zeros(len(returns), dtype=np.float64),
            predictions.return_variance[:, column],
        )
        baseline_crps = gaussian_crps(
            returns[:, column],
            np.zeros(len(returns), dtype=np.float64),
            baseline_distribution[:, column],
        )
        full_model_nll = gaussian_nll(
            returns[:, column],
            predictions.return_location[:, column],
            predictions.return_variance[:, column],
        )
        variance_only_nll = gaussian_nll(
            returns[:, column],
            np.zeros(len(returns), dtype=np.float64),
            predictions.return_variance[:, column],
        )
        baseline_nll = gaussian_nll(
            returns[:, column],
            np.zeros(len(returns), dtype=np.float64),
            baseline_distribution[:, column],
        )
        log_variance_error = np.log(predictions.variance[:, column]) - np.log(realized[:, column])
        mean_error = predictions.return_location[:, column] - returns[:, column]
        baseline_mean_error = -returns[:, column]
        evidence: dict[str, float | int] = {
            "horizon": int(horizon),
            "rows": int(len(returns)),
            "qlike": float(np.mean(model_qlike)),
            "baseline_qlike": float(np.mean(baseline_qlike)),
            "relative_qlike": _safe_ratio(
                float(np.mean(model_qlike)), float(np.mean(baseline_qlike))
            ),
            # The full-model score remains diagnostic. Volatility promotion
            # uses the candidate variance around the matched zero-return
            # location so a weak auxiliary mean cannot veto a strong variance
            # forecast (or hide behind it).
            "gaussian_crps": float(np.mean(full_model_crps)),
            "baseline_gaussian_crps": float(np.mean(baseline_crps)),
            "relative_gaussian_crps": _safe_ratio(
                float(np.mean(full_model_crps)), float(np.mean(baseline_crps))
            ),
            "variance_only_gaussian_crps": float(np.mean(variance_only_crps)),
            "relative_variance_only_gaussian_crps": _safe_ratio(
                float(np.mean(variance_only_crps)), float(np.mean(baseline_crps))
            ),
            "gaussian_nll": float(np.mean(full_model_nll)),
            "variance_only_gaussian_nll": float(np.mean(variance_only_nll)),
            "baseline_gaussian_nll": float(np.mean(baseline_nll)),
            "log_variance_mae": float(np.mean(np.abs(log_variance_error))),
            "log_variance_rmse": float(np.sqrt(np.mean(log_variance_error**2))),
            "return_location_mae": float(np.mean(np.abs(mean_error))),
            "baseline_return_mae": float(np.mean(np.abs(baseline_mean_error))),
            "relative_return_mae": _safe_ratio(
                float(np.mean(np.abs(mean_error))),
                float(np.mean(np.abs(baseline_mean_error))),
            ),
            "return_location_rmse": float(np.sqrt(np.mean(mean_error**2))),
            "baseline_return_rmse": float(np.sqrt(np.mean(baseline_mean_error**2))),
            "relative_return_rmse": _safe_ratio(
                float(np.sqrt(np.mean(mean_error**2))),
                float(np.sqrt(np.mean(baseline_mean_error**2))),
            ),
        }
        standard_deviation = np.sqrt(predictions.return_variance[:, column])
        for label, z_value in _NORMAL_INTERVAL_Z.items():
            lower = predictions.return_location[:, column] - z_value * standard_deviation
            upper = predictions.return_location[:, column] + z_value * standard_deviation
            evidence[f"coverage_{label}"] = float(
                np.mean((returns[:, column] >= lower) & (returns[:, column] <= upper))
            )
            evidence[f"mean_width_{label}"] = float(np.mean(upper - lower))
            variance_only_lower = -z_value * standard_deviation
            variance_only_upper = z_value * standard_deviation
            evidence[f"variance_only_coverage_{label}"] = float(
                np.mean(
                    (returns[:, column] >= variance_only_lower)
                    & (returns[:, column] <= variance_only_upper)
                )
            )
        evidence.update(
            _multiclass_metrics(
                predictions.direction_probabilities[:, column, :],
                classes[:, column],
            )
        )
        rows.append(evidence)
    return rows
