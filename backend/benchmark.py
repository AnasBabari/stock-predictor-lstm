"""Operator-only command for reproducible baseline and feature-ablation reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_pipeline import fetch_data
from experiments.ablation import feature_ablation_sets, run_feature_ablation
from experiments.runner import ExperimentConfig


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    parsed = tuple(_positive_int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("at least one value is required")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--lookback", type=_positive_int, default=60)
    parser.add_argument("--horizons", type=_parse_int_tuple, default=(1, 5, 20))
    parser.add_argument("--folds", type=_positive_int, default=5)
    parser.add_argument("--min-train-size", type=_positive_int, default=300)
    parser.add_argument("--validation-size", type=_positive_int, default=60)
    parser.add_argument(
        "--target-type",
        choices=("price_level", "simple_return", "log_return", "persistence_residual"),
        default="log_return",
    )
    parser.add_argument(
        "--feature-sets",
        default="price,ohlcv,ohlcv_market,ohlcv_technical_market",
        help=f"Comma-separated values from: {', '.join(feature_ablation_sets())}",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    feature_sets = tuple(value.strip() for value in args.feature_sets.split(",") if value.strip())
    feature_frame, _closing_prices, dates, metadata = fetch_data(args.ticker.upper())
    config = ExperimentConfig(
        lookback=args.lookback,
        horizons=args.horizons,
        target_type=args.target_type,
        folds=args.folds,
        min_train_size=args.min_train_size,
        validation_size=args.validation_size,
    )
    report = run_feature_ablation(
        feature_frame,
        feature_sets=feature_sets,
        config=config,
    )
    report["ticker"] = args.ticker.upper()
    report["data"] = {
        "start": str(dates[0].date()),
        "end": str(dates[-1].date()),
        "rows": int(len(feature_frame)),
        "snapshot_id": metadata["snapshot_id"],
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
