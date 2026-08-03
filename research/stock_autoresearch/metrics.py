"""Baseline-relative price and direction forecasting metrics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def mse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.square(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return math.sqrt(mse(actual, predicted))


def regression_metrics(actual: np.ndarray, predicted: np.ndarray, baseline: np.ndarray) -> dict[str, float]:
    model_mae = mae(actual, predicted)
    model_rmse = rmse(actual, predicted)
    baseline_mae = mae(actual, baseline)
    baseline_rmse = rmse(actual, baseline)
    return {
        "mae": model_mae,
        "mse": mse(actual, predicted),
        "rmse": model_rmse,
        "baseline_mae": baseline_mae,
        "baseline_rmse": baseline_rmse,
        "relative_mae": model_mae / baseline_mae if baseline_mae else 0.0,
        "relative_rmse": model_rmse / baseline_rmse if baseline_rmse else 0.0,
        "bias": float(np.mean(predicted - actual)),
    }


def classification_metrics(actual_returns: np.ndarray, predicted_probs: np.ndarray) -> dict[str, Any]:
    """Metrics for direction classification: balanced accuracy, brier score, precision/recall/F1."""
    actual_labels = (np.asarray(actual_returns).ravel() > 0).astype(int)
    probs = np.clip(np.asarray(predicted_probs).ravel(), 0.0, 1.0)
    pred_labels = (probs >= 0.5).astype(int)

    tp = int(np.sum((actual_labels == 1) & (pred_labels == 1)))
    tn = int(np.sum((actual_labels == 0) & (pred_labels == 0)))
    fp = int(np.sum((actual_labels == 0) & (pred_labels == 1)))
    fn = int(np.sum((actual_labels == 1) & (pred_labels == 0)))

    n = len(actual_labels)
    accuracy = float((tp + tn) / n) if n else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    down_recall = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    balanced_acc = float((recall + down_recall) / 2.0)
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    brier = float(np.mean((probs - actual_labels) ** 2)) if n else 0.0
    positives = int(np.sum(actual_labels))
    naive_baseline = float(max(positives, n - positives) / n) if n else 0.5

    return {
        "accuracy": accuracy,
        "directional_accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "brier_score": brier,
        "naive_baseline": naive_baseline,
        "evaluation_rows": n,
    }
