"""Operator-only command for reproducible baseline and feature-ablation reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from config import FEATURES
from data_pipeline import fetch_data
from experiments.ablation import feature_ablation_sets, run_feature_ablation
from experiments.candidates import NeuralCandidate
from experiments.runner import ExperimentConfig
from features.calendar import add_calendar_features
from features.market import add_market_context_from_frames
from features.pipeline import validate_features
from features.technical import add_technical_indicators
from snapshot import load_market_snapshot


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
    parser.add_argument(
        "--snapshot",
        type=Path,
        help="Verified market manifest; when omitted the command performs a live research fetch.",
    )
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
    parser.add_argument(
        "--models",
        default="baseline",
        help=("Comma-separated values from: baseline, lstm, gru, bilstm_attention_regression"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--neural-epochs", type=_positive_int, default=30)
    parser.add_argument("--output", type=Path)
    return parser


def _snapshot_features(ticker: str, manifest_path: Path):
    manifest, frames = load_market_snapshot(manifest_path)
    if ticker not in frames:
        raise ValueError(f"Verified snapshot does not contain {ticker}.")
    frame = add_technical_indicators(frames[ticker])
    frame, market_metadata = add_market_context_from_frames(frame, frames)
    frame = add_calendar_features(frame).dropna()
    frame = frame[FEATURES]
    validate_features(frame, FEATURES)
    snapshot_id = manifest["content_sha256"]
    metadata = {
        "snapshot_id": snapshot_id,
        "market_context": market_metadata,
        "manifest_sha256": manifest["manifest_sha256"],
    }
    return frame, frame["Close"].to_numpy(), frame.index, metadata


def _candidate_factories(args) -> tuple:
    requested = tuple(value.strip() for value in args.models.split(",") if value.strip())
    allowed = {"baseline", "lstm", "gru", "bilstm_attention_regression"}
    unknown = sorted(set(requested) - allowed)
    if unknown:
        raise ValueError(f"Unknown benchmark models: {unknown}")
    factories = []
    for architecture in requested:
        if architecture == "baseline":
            continue
        factories.append(
            lambda architecture=architecture: NeuralCandidate(
                architecture=architecture,
                epochs=args.neural_epochs,
                seed=args.seed,
            )
        )
    return tuple(factories)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    feature_sets = tuple(value.strip() for value in args.feature_sets.split(",") if value.strip())
    ticker = args.ticker.upper()
    if args.snapshot:
        feature_frame, _closing_prices, dates, metadata = _snapshot_features(ticker, args.snapshot)
    else:
        feature_frame, _closing_prices, dates, metadata = fetch_data(ticker)
    config = ExperimentConfig(
        lookback=args.lookback,
        horizons=args.horizons,
        target_type=args.target_type,
        folds=args.folds,
        min_train_size=args.min_train_size,
        validation_size=args.validation_size,
        seed=args.seed,
    )
    report = run_feature_ablation(
        feature_frame,
        feature_sets=feature_sets,
        config=config,
        dates=dates,
        snapshot_id=metadata["snapshot_id"],
        candidate_factories=_candidate_factories(args),
    )
    report["ticker"] = ticker
    report["data"] = {
        "start": str(dates[0].date()),
        "end": str(dates[-1].date()),
        "rows": int(len(feature_frame)),
        "snapshot_id": metadata["snapshot_id"],
        "snapshot_mode": "verified_manifest" if args.snapshot else "live_fetch",
    }
    report["report_id"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
