"""Checkpoint inference and date-aggregated, exploratory Bartlett HAC inference."""

import numpy as np
import pandas as pd
from scipy.stats import norm

from .gpu_pipeline import PriceTrainingConfig, _build_model, _predict


def checkpoint_predictions(dataset, path):
    import torch
    from torch import nn

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("artifact_role") != "validation_selected_model":
        raise ValueError("Inference requires the evaluated selection checkpoint")
    if checkpoint["feature_names"] != list(dataset.feature_names) or checkpoint[
        "ticker_names"
    ] != list(dataset.ticker_names):
        raise ValueError("Checkpoint feature/ticker identity differs from dataset")
    settings = PriceTrainingConfig(**checkpoint["config"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _build_model(
        torch, nn, len(dataset.feature_names), len(dataset.ticker_names), settings
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    scalers = {k: np.asarray(v, dtype=np.float32) for k, v in checkpoint["scalers"].items()}
    result = _predict(torch, model, dataset, scalers, dataset.split_validation, device_name=device)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def ridge_predictions(dataset, report):
    x = dataset.sequences[dataset.split_validation, -1, :]
    scaled = (x - np.asarray(report["scaler_mean"])) / np.asarray(report["scaler_scale"])
    return scaled @ np.asarray(report["coefficients"]).T + np.asarray(report["intercept"])


def validation_table(dataset, ridge, lstm):
    indices = dataset.split_validation
    shape = dataset.targets[indices].shape
    if ridge.shape != shape or lstm.shape != shape:
        raise ValueError("Validation prediction shape mismatch")
    stocks = np.asarray(dataset.ticker_names)[dataset.ticker_indices[indices]]
    table = pd.DataFrame(
        {
            "sample_idx": np.repeat(indices, shape[1]),
            "date": pd.to_datetime(np.repeat(dataset.origin_dates[indices], shape[1])),
            "stock_id": np.repeat(stocks, shape[1]),
            "market": np.repeat(np.where(np.char.endswith(stocks, ".L"), "UK", "US"), shape[1]),
            "horizon": np.tile(np.arange(1, shape[1] + 1, dtype=np.int8), len(indices)),
            "y_true": dataset.targets[indices].ravel().astype(np.float32),
            "y_pred_ridge": ridge.ravel().astype(np.float32),
            "y_pred_lstm": lstm.ravel().astype(np.float32),
            "y_naive": np.zeros(len(indices) * shape[1], dtype=np.float32),
        }
    )
    table["market"] = table.market.astype("category")
    if (
        table.duplicated(["sample_idx", "horizon"]).any()
        or not np.isfinite(table.select_dtypes("number")).all().all()
    ):
        raise ValueError("Invalid validation export")
    return table


def hac_mean(values, bandwidth):
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 3 or bandwidth < 0 or bandwidth >= n or not np.isfinite(values).all():
        raise ValueError("Invalid HAC series/bandwidth")
    mean = float(values.mean())
    e = values - mean
    meat = e @ e
    for lag in range(1, bandwidth + 1):
        meat += 2 * (1 - lag / (bandwidth + 1)) * (e[lag:] @ e[:-lag])
    se = float(np.sqrt(max(0.0, meat / (n * (n - 1)))))
    return {
        "dates": n,
        "mean_loss_improvement": mean,
        "se": se,
        "ci95": [mean - 1.95996398454 * se, mean + 1.95996398454 * se],
        "p_two_sided": float(2 * norm.sf(abs(mean / se))) if se else None,
    }


def paired_tests(table, comparisons=None):
    comparisons = comparisons or [
        ("y_naive", "y_pred_ridge"),
        ("y_naive", "y_pred_lstm"),
        ("y_pred_ridge", "y_pred_lstm"),
    ]
    results = []
    for scope in ("ALL", "US", "UK"):
        part = table if scope == "ALL" else table[table.market == scope]
        for horizon, group in part.groupby("horizon", observed=True):
            actual = np.exp(group.y_true.to_numpy(dtype=float))
            for ref, cand in comparisons:
                difference = 100 * (
                    np.abs(np.exp(group[ref].to_numpy(dtype=float)) - actual)
                    - np.abs(np.exp(group[cand].to_numpy(dtype=float)) - actual)
                )
                daily = pd.Series(difference, index=group.date).groupby(level=0).mean().sort_index()
                widths = (
                    [0, 1, 5]
                    if horizon == 1
                    else [int(horizon) - 1, 2 * (int(horizon) - 1), 3 * (int(horizon) - 1)]
                )
                for width in widths:
                    results.append(
                        {
                            "scope": scope,
                            "horizon": int(horizon),
                            "reference": ref,
                            "candidate": cand,
                            "bandwidth": width,
                            **hac_mean(daily, width),
                        }
                    )
    # Conservative family adjustment over every reported comparison/sensitivity.
    valid = sorted(
        (r for r in results if r["p_two_sided"] is not None), key=lambda r: r["p_two_sided"]
    )
    bound = 0.0
    for rank, row in enumerate(valid):
        bound = max(bound, min(1.0, (len(valid) - rank) * row["p_two_sided"]))
        row["p_holm_all_reported"] = bound
    return {
        "method": "date_mean_Bartlett_HAC_intercept_normal_CI_n_over_n_minus_1",
        "interpretation": "exploratory validation; positive difference favors candidate; dates equally weighted",
        "results": results,
    }
