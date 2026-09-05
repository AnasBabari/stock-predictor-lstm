"""Matched validation metrics and a fixed-alpha, latest-feature Ridge baseline."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def training_majority(targets):
    targets = np.asarray(targets)
    if targets.ndim != 2 or not len(targets) or not np.isfinite(targets).all():
        raise ValueError("Training targets must be a nonempty finite matrix")
    return np.where((targets > 0).sum(axis=0) >= (targets < 0).sum(axis=0), 1, -1)


def evaluate_predictions(actual, predicted, majority, stock_ids):
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    majority = np.asarray(majority)
    stock_ids = np.asarray(stock_ids)
    if (
        actual.ndim != 2
        or actual.shape != predicted.shape
        or not actual.size
        or majority.shape != (actual.shape[1],)
        or stock_ids.shape != (len(actual),)
        or not np.isfinite(actual).all()
        or not np.isfinite(predicted).all()
        or not np.isin(majority, [-1, 1]).all()
    ):
        raise ValueError("Invalid validation predictions, majority labels, or stock IDs")
    with np.errstate(over="ignore", invalid="ignore"):
        errors = 100 * (np.exp(predicted) - np.exp(actual))
        persistence = 100 * (1 - np.exp(actual))
    if not np.isfinite(errors).all() or not np.isfinite(persistence).all():
        raise ValueError("Nonfinite price-relative errors")
    correct = (np.sign(predicted) == np.sign(actual)) & (actual != 0)
    majority_correct = (np.sign(actual) == majority) & (actual != 0)

    def score(rows, columns):
        err, base = errors[rows][:, columns], persistence[rows][:, columns]
        mae, base_mae = float(np.abs(err).mean()), float(np.abs(base).mean())
        rmse = float(np.sqrt(np.square(err).mean()))
        base_rmse = float(np.sqrt(np.square(base).mean()))
        return {
            "mae_percent": mae,
            "rmse_percent": rmse,
            "persistence_mae_percent": base_mae,
            "persistence_rmse_percent": base_rmse,
            "relative_mae_vs_persistence": mae / base_mae if base_mae > 0 else None,
            "relative_rmse_vs_persistence": rmse / base_rmse if base_rmse > 0 else None,
            "direction_accuracy": float(correct[rows][:, columns].mean()),
            "majority_direction_accuracy": float(majority_correct[rows][:, columns].mean()),
            "zero_return_fraction": float((actual[rows][:, columns] == 0).mean()),
        }

    def breakdown(rows):
        result = score(rows, slice(None))
        result["per_horizon"] = [
            {"day": h + 1, **score(rows, slice(h, h + 1))} for h in range(actual.shape[1])
        ]
        return result

    return {
        "metric_version": "price-relative-validation-v1",
        "error_units": "percentage_points_of_origin_price",
        "direction_zero_policy": "actual_zero_always_incorrect",
        "majority_direction": majority.tolist(),
        "pooled": breakdown(slice(None)),
        "per_ticker": {str(stock): breakdown(stock_ids == stock) for stock in np.unique(stock_ids)},
    }


def fit_ridge_validation(dataset, alpha=100.0):
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    train, validation = dataset.split_train, dataset.split_validation
    # A static point-in-time baseline, not a flattened sequence regression.
    x_train = dataset.sequences[train, -1, :]
    x_validation = dataset.sequences[validation, -1, :]
    scaler = StandardScaler().fit(x_train)
    model = Ridge(alpha=alpha, solver="cholesky").fit(
        scaler.transform(x_train), dataset.targets[train]
    )
    predictions = model.predict(scaler.transform(x_validation))
    majority = training_majority(dataset.targets[train])
    stocks = np.asarray(dataset.ticker_names)[dataset.ticker_indices[validation]]
    report = evaluate_predictions(dataset.targets[validation], predictions, majority, stocks)
    report.update(
        {
            "model_type": "ridge_point_in_time",
            "alpha": alpha,
            "input_slice": "sequences[:, -1, :]",
            "num_features": x_train.shape[1],
            "test_evaluated": False,
            "train_rows": len(train),
            "validation_rows": len(validation),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "coefficients": model.coef_.tolist(),
            "intercept": model.intercept_.tolist(),
        }
    )
    return report
