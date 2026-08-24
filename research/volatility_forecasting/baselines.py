"""Strong causal variance baselines selected inside each outer fold.

The production candidate is allowed to learn residuals around log-HAR, but
promotion must be measured against the strongest simple forecast available at
the same information cutoff.  Candidate selection and multiplicative
calibration therefore use only the fold's pre-evaluation calibration rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import VolatilityPanelExamples
from .metrics import fit_qlike_variance_scale, qlike_losses

_MIN_VARIANCE = 1e-12
_CALIBRATION_IMPROVEMENT = 0.995
_BLEND_GRID = tuple(float(value) for value in np.linspace(0.0, 1.0, 11))


@dataclass(frozen=True)
class AdaptiveBaselineHorizon:
    """One horizon's baseline family and parameters frozen before evaluation."""

    horizon: int
    family: str
    blend_alpha: float
    multiplicative_scale: float
    calibration_qlike: float
    har_calibration_qlike: float


@dataclass(frozen=True)
class AdaptiveBaselineSelection:
    """Fold-local baseline decision, one independently selected per horizon."""

    horizons: tuple[AdaptiveBaselineHorizon, ...]


def _feature_last_step(examples: VolatilityPanelExamples, name: str) -> np.ndarray | None:
    try:
        column = examples.feature_names.index(name)
    except ValueError:
        return None
    return np.asarray(examples.features[:, -1, column], dtype=np.float64)


def variance_baseline_candidates(
    examples: VolatilityPanelExamples,
) -> dict[str, np.ndarray]:
    """Construct finite causal candidate matrices from each origin's last row."""
    horizons = np.asarray(examples.horizons, dtype=np.float64)[None, :]
    candidates = {
        "causal_log_har": np.asarray(examples.baseline_variance, dtype=np.float64),
    }
    feature_families = {
        "riskmetrics_ewma_c2c": ("EWMA_Var", False),
        "rolling_c2c_5": ("Vol_C2C_5", True),
        "rolling_c2c_20": ("Vol_C2C_20", True),
        "rolling_c2c_60": ("Vol_C2C_60", True),
    }
    for family, (feature_name, square) in feature_families.items():
        values = _feature_last_step(examples, feature_name)
        if values is None:
            continue
        daily_variance = values**2 if square else values
        forecast = np.maximum(daily_variance[:, None] * horizons, _MIN_VARIANCE)
        if np.isfinite(forecast).all():
            candidates[family] = forecast

    rolling = [
        candidates[name]
        for name in ("rolling_c2c_5", "rolling_c2c_20", "rolling_c2c_60")
        if name in candidates
    ]
    if len(rolling) == 3:
        # A geometric mean gives each horizon a stable multi-scale variance
        # forecast without letting the largest raw scale dominate.
        stacked = np.stack(rolling, axis=0)
        candidates["rolling_c2c_multiscale"] = np.exp(
            np.mean(np.log(np.maximum(stacked, _MIN_VARIANCE)), axis=0)
        )
    return candidates


def _blend(har: np.ndarray, alternative: np.ndarray, alpha: float) -> np.ndarray:
    return np.exp(
        (1.0 - alpha) * np.log(np.maximum(har, _MIN_VARIANCE))
        + alpha * np.log(np.maximum(alternative, _MIN_VARIANCE))
    )


def _session_equal_loss(losses: np.ndarray, labels: np.ndarray) -> float:
    values = np.asarray(losses, dtype=np.float64)
    sessions, inverse = np.unique(labels, return_inverse=True)
    totals = np.zeros(len(sessions), dtype=np.float64)
    counts = np.zeros(len(sessions), dtype=np.int64)
    np.add.at(totals, inverse, values)
    np.add.at(counts, inverse, 1)
    return float(np.mean(totals / counts))


def fit_adaptive_variance_baseline(
    examples: VolatilityPanelExamples,
    calibration_indices: np.ndarray,
) -> AdaptiveBaselineSelection:
    """Select and calibrate a strong simple baseline without evaluation rows."""
    indices = np.asarray(calibration_indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) < 5 or len(np.unique(indices)) != len(indices):
        raise ValueError("baseline calibration indices must be unique and contain at least 5 rows")
    if indices.min() < 0 or indices.max() >= len(examples.features):
        raise ValueError("baseline calibration index is out of bounds")

    candidates = variance_baseline_candidates(examples)
    realized = np.asarray(examples.realized_variance[indices], dtype=np.float64)
    session_labels = examples.origin_dates[indices]
    har = candidates["causal_log_har"]
    selections: list[AdaptiveBaselineHorizon] = []

    for column, horizon in enumerate(examples.horizons):
        raw_har = har[indices, column : column + 1]
        har_scale = float(
            fit_qlike_variance_scale(
                raw_har,
                realized[:, column : column + 1],
                session_labels=session_labels,
            )[0]
        )
        calibrated_har = raw_har[:, 0] * har_scale
        har_loss = _session_equal_loss(
            qlike_losses(calibrated_har, realized[:, column]),
            session_labels,
        )
        best = AdaptiveBaselineHorizon(
            horizon=horizon,
            family="causal_log_har",
            blend_alpha=0.0,
            multiplicative_scale=har_scale,
            calibration_qlike=har_loss,
            har_calibration_qlike=har_loss,
        )

        for family, forecast in sorted(candidates.items()):
            if family == "causal_log_har":
                continue
            for alpha in _BLEND_GRID[1:]:
                raw = _blend(har[indices, column], forecast[indices, column], alpha)
                scale = float(
                    fit_qlike_variance_scale(
                        raw[:, None],
                        realized[:, column : column + 1],
                        session_labels=session_labels,
                    )[0]
                )
                loss = _session_equal_loss(
                    qlike_losses(raw * scale, realized[:, column]),
                    session_labels,
                )
                if loss < best.calibration_qlike:
                    best = AdaptiveBaselineHorizon(
                        horizon=horizon,
                        family=family,
                        blend_alpha=alpha,
                        multiplicative_scale=scale,
                        calibration_qlike=loss,
                        har_calibration_qlike=har_loss,
                    )

        # Avoid replacing HAR for a negligible in-sample fluctuation.
        if best.calibration_qlike >= har_loss * _CALIBRATION_IMPROVEMENT:
            best = AdaptiveBaselineHorizon(
                horizon=horizon,
                family="causal_log_har",
                blend_alpha=0.0,
                multiplicative_scale=har_scale,
                calibration_qlike=har_loss,
                har_calibration_qlike=har_loss,
            )
        selections.append(best)
    return AdaptiveBaselineSelection(horizons=tuple(selections))


def predict_adaptive_variance_baseline(
    examples: VolatilityPanelExamples,
    indices: np.ndarray,
    selection: AdaptiveBaselineSelection,
) -> np.ndarray:
    """Apply a frozen fold-local selection to arbitrary later origin rows."""
    rows = np.asarray(indices, dtype=np.int64)
    if rows.ndim != 1 or rows.size == 0:
        raise ValueError("baseline prediction indices must be a non-empty vector")
    if tuple(item.horizon for item in selection.horizons) != examples.horizons:
        raise ValueError("baseline selection horizons do not match examples")
    candidates = variance_baseline_candidates(examples)
    har = candidates["causal_log_har"]
    output = np.empty((len(rows), len(examples.horizons)), dtype=np.float64)
    for column, item in enumerate(selection.horizons):
        alternative = candidates.get(item.family)
        if alternative is None:
            raise ValueError(f"selected baseline family is unavailable: {item.family}")
        raw = _blend(har[rows, column], alternative[rows, column], item.blend_alpha)
        output[:, column] = np.maximum(raw * item.multiplicative_scale, _MIN_VARIANCE)
    if not np.isfinite(output).all():
        raise ValueError("adaptive baseline produced non-finite forecasts")
    return output
