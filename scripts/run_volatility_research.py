#!/usr/bin/env python3
"""Run the frozen global volatility TCN development benchmark on local GPU."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "research"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from volatility_forecasting.cache import (  # noqa: E402
    ExampleCacheError,
    example_cache_key,
    load_example_cache,
    panel_fingerprint,
    save_example_cache,
)
from volatility_forecasting.contracts import (  # noqa: E402
    VolatilityForecastProtocol,
    VolatilityPromotionGate,
)
from volatility_forecasting.data import build_volatility_panel_examples  # noqa: E402
from volatility_forecasting.evaluation import evaluate_tcn_development  # noqa: E402
from volatility_forecasting.folds import build_volatility_fold_plan  # noqa: E402
from volatility_forecasting.model import (  # noqa: E402
    BaselineResidualTCNConfig,
    TorchTrainingConfig,
)

from backend.panel.snapshots import load_panel_from_directory  # noqa: E402


def _json_value(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot encode {type(value).__name__}")


def _evaluation_record(evaluation) -> dict[str, object]:
    return {
        "protocol_version": evaluation.protocol_version,
        "seed": evaluation.seed,
        "folds": [asdict(fold) for fold in evaluation.folds],
        "pooled_metrics": list(evaluation.pooled_metrics),
        "promotion": [asdict(decision) for decision in evaluation.promotion],
        "oof_rows": int(len(evaluation.oof_indices)),
    }


def _seed_consensus(
    records: list[dict[str, object]], horizons: tuple[int, ...]
) -> dict[str, object]:
    summary: dict[str, object] = {}
    for column, horizon in enumerate(horizons):
        decisions = [record["promotion"][column] for record in records]
        metrics = [record["pooled_metrics"][column] for record in records]
        promoted_all_seeds = all(bool(decision["promoted"]) for decision in decisions)
        summary[str(horizon)] = {
            "promoted_all_seeds": promoted_all_seeds,
            "relative_qlike_median": float(
                np.median([float(metric["relative_qlike"]) for metric in metrics])
            ),
            "relative_qlike_worst_seed": float(
                np.max([float(metric["relative_qlike"]) for metric in metrics])
            ),
            "relative_crps_median": float(
                np.median(
                    [float(metric["relative_variance_only_gaussian_crps"]) for metric in metrics]
                )
            ),
            "coverage_80_range": [
                float(np.min([float(metric["coverage_80"]) for metric in metrics])),
                float(np.max([float(metric["coverage_80"]) for metric in metrics])),
            ],
            "reasons_by_seed": {
                str(record["seed"]): list(decision["reasons"])
                for record, decision in zip(records, decisions, strict=True)
            },
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the frozen baseline-residual volatility TCN"
    )
    parser.add_argument("--panel-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--maximum-epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--quick", action="store_true", help="Non-certifiable one-seed smoke run")
    parser.add_argument(
        "--example-cache-root",
        type=Path,
        help="Local derived-array cache (defaults beside the immutable panel)",
    )
    parser.add_argument("--no-example-cache", action="store_true")
    args = parser.parse_args()

    protocol = VolatilityForecastProtocol()
    gate = VolatilityPromotionGate()
    seeds = (protocol.seeds[1],) if args.quick else protocol.seeds
    maximum_epochs = min(args.maximum_epochs, 3) if args.quick else args.maximum_epochs
    training = TorchTrainingConfig(
        maximum_epochs=maximum_epochs,
        patience=2 if args.quick else 8,
        batch_size=args.batch_size,
        use_amp=args.device == "cuda",
    )
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    panel_checksum = panel_fingerprint(args.panel_dir)
    cache_root = (args.example_cache_root or (args.panel_dir.parent / "example-cache")).resolve()
    cache_dir = cache_root / example_cache_key(panel_checksum, protocol)
    examples = None
    if not args.no_example_cache and cache_dir.exists():
        try:
            print(f"Verifying derived example cache {cache_dir}...", flush=True)
            examples = load_example_cache(
                cache_dir,
                panel_checksum=panel_checksum,
                protocol=protocol,
            )
            print("Loaded verified causal examples from the local cache.", flush=True)
        except ExampleCacheError as error:
            print(f"Ignoring invalid example cache: {error}", flush=True)
    if examples is None:
        print(f"Loading immutable panel from {args.panel_dir}...", flush=True)
        panel = load_panel_from_directory(args.panel_dir)
        print(f"Building causal examples for {len(panel)} tickers...", flush=True)
        examples = build_volatility_panel_examples(panel, protocol)
        if not args.no_example_cache:
            print(f"Saving checksummed derived examples to {cache_dir}...", flush=True)
            save_example_cache(
                cache_dir,
                examples,
                panel_checksum=panel_checksum,
                protocol=protocol,
            )
    fold_plan = build_volatility_fold_plan(examples, protocol)
    print(
        f"Built {len(examples.features):,} examples; "
        f"training assets={len(fold_plan.train_tickers)}, "
        f"unseen assets={fold_plan.asset_holdout_tickers}",
        flush=True,
    )

    architecture = BaselineResidualTCNConfig(
        feature_count=examples.features.shape[-1],
        horizon_count=len(protocol.horizons),
    )
    seed_records: list[dict[str, object]] = []
    for seed in seeds:
        print(f"Evaluating seed {seed} on {len(fold_plan.folds)} folds...", flush=True)
        evaluation = evaluate_tcn_development(
            examples,
            fold_plan,
            protocol,
            model_config=architecture,
            training_config=training,
            promotion_gate=gate,
            seed=seed,
            device=args.device,
            resamples=200 if args.quick else 1000,
        )
        record = _evaluation_record(evaluation)
        seed_records.append(record)
        (run_dir / f"seed-{seed}.json").write_text(
            json.dumps(record, indent=2, sort_keys=True, default=_json_value),
            encoding="utf-8",
        )
        seven_day = next(decision for decision in evaluation.promotion if decision.horizon == 7)
        print(
            f"Seed {seed}: 7d relative QLIKE={seven_day.relative_qlike:.4f}, "
            f"promoted={seven_day.promoted}",
            flush=True,
        )

    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "certifiable": not args.quick and len(seed_records) == len(protocol.seeds),
        "mode": "quick_smoke" if args.quick else "full_development",
        "panel_dir": str(args.panel_dir.resolve()),
        "protocol": asdict(protocol),
        "promotion_gate": asdict(gate),
        "architecture": asdict(architecture),
        "training": asdict(training),
        "example_rows": len(examples.features),
        "feature_count": examples.features.shape[-1],
        "train_tickers": list(fold_plan.train_tickers),
        "asset_holdout_tickers": list(fold_plan.asset_holdout_tickers),
        "seeds": seed_records,
        "seed_consensus": _seed_consensus(seed_records, protocol.horizons),
    }
    report_path = run_dir / "development-report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=_json_value),
        encoding="utf-8",
    )
    print(f"Development report: {report_path}", flush=True)
    print(
        json.dumps(report["seed_consensus"]["7"], indent=2, sort_keys=True),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
