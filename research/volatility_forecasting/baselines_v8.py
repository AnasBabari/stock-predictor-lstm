"""Leakage-safe v8 development baselines.

These are the preregistered baselines that any learned v8 candidate must
beat or complement on the sealed test set.  They run on the same
chronological split and use only train rows for fitting.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .baselines import (
    fit_adaptive_variance_baseline,
    predict_adaptive_variance_baseline,
)
from .data import VolatilityPanelExamples
from .metrics import qlike_losses


def _qlike(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Return the canonical QLIKE mean, with arguments ordered truth/forecast."""
    realized = np.maximum(np.asarray(y_true, dtype=np.float64), eps)
    forecast = np.maximum(np.asarray(y_pred, dtype=np.float64), eps)
    return float(np.mean(qlike_losses(forecast, realized)))


def _session_equal_qlike(
    realized: np.ndarray,
    forecast: np.ndarray,
    session_labels: np.ndarray,
) -> tuple[float, list[float]]:
    losses = qlike_losses(forecast, realized)
    sessions, inverse = np.unique(session_labels, return_inverse=True)
    per_session = np.zeros((len(sessions), losses.shape[1]), dtype=np.float64)
    counts = np.zeros(len(sessions), dtype=np.int64)
    np.add.at(per_session, inverse, losses)
    np.add.at(counts, inverse, 1)
    per_session /= counts[:, None]
    return float(np.mean(per_session)), np.mean(per_session, axis=0).tolist()


def _ridge_forecast(
    examples: VolatilityPanelExamples,
    fit_idx: np.ndarray,
    eval_idx: np.ndarray,
) -> np.ndarray:
    """Fit a positive log-variance Ridge using fit rows only."""
    train_x = np.asarray(examples.features[fit_idx].mean(axis=1), dtype=np.float64)
    eval_x = np.asarray(examples.features[eval_idx].mean(axis=1), dtype=np.float64)
    train_y = np.log(np.maximum(examples.realized_variance[fit_idx], 1e-12))
    columns: list[np.ndarray] = []
    for column in range(train_y.shape[1]):
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(train_x, train_y[:, column])
        columns.append(np.exp(model.predict(eval_x)))
    result = np.column_stack(columns)
    if not np.isfinite(result).all() or (result <= 0).any():
        raise RuntimeError("Ridge baseline produced invalid variance forecasts")
    return result


def evaluate_development_baselines(
    examples: VolatilityPanelExamples,
    *,
    fit_indices: np.ndarray,
    evaluation_indices: np.ndarray,
) -> dict[str, dict[str, object]]:
    """Evaluate baselines on an explicitly supplied development partition.

    The function deliberately has no access to ``V8SplitIndices``.  Callers
    cannot accidentally substitute the sealed test partition merely by
    passing the split object.
    """
    fit = np.asarray(fit_indices, dtype=np.int64)
    evaluation = np.asarray(evaluation_indices, dtype=np.int64)
    if not len(fit) or not len(evaluation) or np.intersect1d(fit, evaluation).size:
        raise ValueError("baseline fit/evaluation rows must be non-empty and disjoint")
    selection = fit_adaptive_variance_baseline(examples, fit)
    adaptive = predict_adaptive_variance_baseline(examples, evaluation, selection)
    ridge = _ridge_forecast(examples, fit, evaluation)
    realized = examples.realized_variance[evaluation]
    sessions = examples.origin_dates[evaluation]
    adaptive_mean, adaptive_horizons = _session_equal_qlike(realized, adaptive, sessions)
    ridge_mean, ridge_horizons = _session_equal_qlike(realized, ridge, sessions)
    return {
        "adaptive_calibrated_har_c2c_v1": {
            "qlike": adaptive_mean,
            "per_horizon_qlike": adaptive_horizons,
        },
        "ridge_log_variance": {
            "qlike": ridge_mean,
            "per_horizon_qlike": ridge_horizons,
        },
    }


def negative_controls() -> dict[str, str]:
    """Required negative controls for news (shuffled / timestamp-shifted).

    These must be evaluated on the same split; if they match the real
    news model's performance, the news signal is likely leakage.
    """
    return {
        "shuffled_news": "permute article order within each window — should not beat real news",
        "timestamp_shifted_news": "shift available_at +7 days — should not beat real news; if it does, leakage",
        "article_count_only": "use count only, no semantics — tests if volume alone explains gain",
        "news_only": "news branch alone without numeric — should underperform fusion",
    }
