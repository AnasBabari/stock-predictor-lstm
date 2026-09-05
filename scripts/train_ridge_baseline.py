"""Run a fixed Ridge baseline, optionally followed by matched CUDA candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.price_forecasting.baselines import fit_ridge_validation  # noqa: E402
from research.price_forecasting.gpu_pipeline import (  # noqa: E402
    PriceTrainingConfig,
    build_global_price_dataset,
    train_cuda_price_model,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-comparison", action="store_true")
    args = parser.parse_args()
    # No implicit overwrite or automatic data download in this experiment.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    settings = PriceTrainingConfig(maximum_epochs=20, patience=3)
    manifest = ROOT / "data/tri_exchange/manifest_broad_300.json"
    tickers = sorted(json.loads(manifest.read_text(encoding="utf-8"))["valid_tickers"])
    frames, hashes = {}, {}
    for ticker in tickers:
        path = ROOT / "data/tri_exchange/cache" / f"{ticker}.parquet"
        hashes[ticker] = hashlib.sha256(path.read_bytes()).hexdigest()
        frames[ticker] = pd.read_parquet(path)
    print(f"Building matched dataset from {len(frames)} cached stocks", flush=True)
    dataset = build_global_price_dataset(frames, settings)
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    protocol = {
        "data_sha256": identity,
        "file_hashes": hashes,
        "train_indices_sha256": hashlib.sha256(dataset.split_train.tobytes()).hexdigest(),
        "validation_indices_sha256": hashlib.sha256(dataset.split_validation.tobytes()).hexdigest(),
        "config": asdict(settings),
        "ridge_alpha": 100.0,
        "test_evaluated": False,
        "selection_metric": "validation_mae_percent",
        "direction_baseline": "training_majority_per_horizon_ties_up_zeros_excluded",
        "candidates": ["ridge_latest_features", "lstm_32x1", "lstm_128x3"],
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    ridge = fit_ridge_validation(dataset)
    (args.output_dir / "ridge.json").write_text(
        json.dumps(ridge, indent=2, allow_nan=False), encoding="utf-8"
    )
    results = {"ridge_latest_features": ridge["pooled"]}
    print(json.dumps({"ridge_validation": ridge["pooled"]}), flush=True)
    if args.gpu_comparison:
        for name, hidden, layers in (("lstm_32x1", 32, 1), ("lstm_128x3", 128, 3)):
            print(f"Starting {name}: validation only, 20 epochs maximum, patience 3", flush=True)
            report = train_cuda_price_model(
                dataset,
                args.output_dir / name,
                replace(settings, hidden_size=hidden, layers=layers),
                validation_only=True,
            )
            results[name] = report["selection"]["metrics"]
            print(json.dumps({name: results[name]}), flush=True)
    summary = {
        "status": "validation_comparison_complete",
        "data_sha256": identity,
        "test_evaluated": False,
        "results": results,
        "lowest_validation_mae": min(results, key=lambda name: results[name]["mae_percent"]),
        "deployment_performed": False,
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
