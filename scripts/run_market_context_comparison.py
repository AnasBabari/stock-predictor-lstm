"""Export preserved validation predictions, run paired tests, then context ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.price_forecasting.baselines import (  # noqa: E402
    evaluate_predictions,
    fit_ridge_validation,
    training_majority,
)
from research.price_forecasting.gpu_pipeline import (  # noqa: E402
    PriceTrainingConfig,
    build_global_price_dataset,
    train_cuda_price_model,
)
from research.price_forecasting.market_context import (  # noqa: E402
    append_context,
    build_market_context,
)
from research.price_forecasting.paired_validation import (  # noqa: E402
    checkpoint_predictions,
    paired_tests,
    ridge_predictions,
    validation_table,
)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((args.baseline_dir / "protocol.json").read_text())
    frames = {}
    for symbol, digest in protocol["file_hashes"].items():
        path = ROOT / "data/tri_exchange/cache" / f"{symbol}.parquet"
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Original cache changed: {symbol}")
        frames[symbol] = pd.read_parquet(path)
    settings = PriceTrainingConfig(**protocol["config"])
    print("Rebuilding original dataset; no training or test scoring", flush=True)
    dataset = build_global_price_dataset(frames, settings)
    for name in ("train", "validation"):
        indices = getattr(dataset, f"split_{name}")
        if hashlib.sha256(indices.tobytes()).hexdigest() != protocol[f"{name}_indices_sha256"]:
            raise ValueError("Partition identity mismatch")
    ridge = json.loads((args.baseline_dir / "ridge.json").read_text())
    selected = args.baseline_dir / "lstm_128x3/selection_model.pt"
    lstm_report = json.loads((args.baseline_dir / "lstm_128x3/validation_report.json").read_text())
    print("Inferring validation rows from preserved Ridge and LSTM", flush=True)
    predictions = validation_table(
        dataset, ridge_predictions(dataset, ridge), checkpoint_predictions(dataset, selected)
    )
    for column, expected in (
        ("y_pred_ridge", ridge["pooled"]),
        ("y_pred_lstm", lstm_report["selection"]["metrics"]),
    ):
        actual = predictions.y_true.to_numpy().reshape(-1, settings.horizon)
        predicted = predictions[column].to_numpy().reshape(-1, settings.horizon)
        evidence = evaluate_predictions(
            actual,
            predicted,
            training_majority(dataset.targets[dataset.split_train]),
            np.asarray(dataset.ticker_names)[dataset.ticker_indices[dataset.split_validation]],
        )
        for key in ("mae_percent", "rmse_percent", "direction_accuracy"):
            if not np.isclose(evidence["pooled"][key], expected[key], atol=1e-5, rtol=0):
                raise ValueError(f"Preserved inference fails report parity: {column}/{key}")
    predictions.to_parquet(args.output_dir / "validation_baseline.parquet", index=False)
    write_json(args.output_dir / "baseline_hac.json", paired_tests(predictions))
    write_json(
        args.output_dir / "protocol.json",
        {
            "baseline_protocol": protocol,
            "selection_checkpoint_sha256": hashlib.sha256(selected.read_bytes()).hexdigest(),
            "ridge_report_sha256": hashlib.sha256(
                (args.baseline_dir / "ridge.json").read_bytes()
            ).hexdigest(),
            "validation_origins": len(dataset.split_validation),
            "parquet_rows": len(predictions),
            "validation_start": str(predictions.date.min()),
            "validation_end": str(predictions.date.max()),
            "coverage": 0.8,
            "market_isolation": "US/UK suffix classification; no cross-market joins",
            "target": "unchanged cumulative log returns",
            "context": "leave-one-out mean log returns of fixed current basket, not an investable index",
            "missing": "rolling raw metrics forward-filled only from past; initial zero placeholder; missing/stale/breadth fields included",
            "test_scored": False,
            "deployment": False,
            "split_caveat": "Existing per-stock chronological splits preserved; not a globally calendar-aligned holdout",
            "inference_caveat": "Exploratory validation selected models; HAC does not undo model selection or survivorship bias",
            "hac_reference": "https://www.statsmodels.org/stable/generated/statsmodels.stats.sandwich_covariance.cov_hac.html",
        },
    )
    print("Baseline export and HAC complete", flush=True)
    if args.export_only:
        return
    context = build_market_context(frames)
    pd.concat(context, names=["stock_id", "date"]).reset_index().to_parquet(
        args.output_dir / "market_context.parquet", index=False
    )
    coverage = {
        s: {
            "rows": len(f),
            "flagged_rows": int(f.context_missing.sum()),
            "max_stale_sessions": int(f.context_stale_sessions.max()),
        }
        for s, f in context.items()
    }
    write_json(args.output_dir / "context_coverage.json", coverage)
    dataset = append_context(dataset, frames, context)
    print(
        f"Context appended: {len(dataset.feature_names)} features, identical row order", flush=True
    )
    augmented_ridge = fit_ridge_validation(dataset)
    write_json(args.output_dir / "ridge_context.json", augmented_ridge)
    print(f"Ridge context validation MAE: {augmented_ridge['pooled']['mae_percent']}", flush=True)
    train_cuda_price_model(
        dataset, args.output_dir / "lstm_context", settings, validation_only=True
    )
    augmented_predictions = validation_table(
        dataset,
        ridge_predictions(dataset, augmented_ridge),
        checkpoint_predictions(dataset, args.output_dir / "lstm_context/selection_model.pt"),
    )
    for column in ("sample_idx", "stock_id", "date", "horizon", "y_true"):
        if not predictions[column].equals(augmented_predictions[column]):
            raise ValueError("Augmentation changed sample/target identity")
    predictions["y_pred_ridge_context"] = augmented_predictions.y_pred_ridge
    predictions["y_pred_lstm_context"] = augmented_predictions.y_pred_lstm
    predictions.to_parquet(args.output_dir / "validation_context_comparison.parquet", index=False)
    write_json(
        args.output_dir / "context_hac.json",
        paired_tests(
            predictions,
            [
                ("y_pred_ridge", "y_pred_ridge_context"),
                ("y_pred_lstm", "y_pred_lstm_context"),
                ("y_naive", "y_pred_ridge_context"),
                ("y_naive", "y_pred_lstm_context"),
            ],
        ),
    )
    write_json(
        args.output_dir / "completion.json",
        {"status": "complete", "test_scored": False, "deployment": False},
    )
    print("Context comparison complete; no test scoring or deployment", flush=True)


if __name__ == "__main__":
    main()
