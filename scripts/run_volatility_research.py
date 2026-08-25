#!/usr/bin/env python3
"""Run the frozen global volatility TCN development benchmark on local GPU."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
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
    find_compatible_example_cache,
    load_example_cache,
    panel_fingerprint,
    save_example_cache,
)
from volatility_forecasting.contracts import (  # noqa: E402
    VolatilityForecastProtocol,
    VolatilityPromotionGate,
)
from volatility_forecasting.data import (  # noqa: E402
    VolatilityPanelExamples,
    build_volatility_panel_examples,
)
from volatility_forecasting.evaluation import evaluate_tcn_development  # noqa: E402
from volatility_forecasting.folds import build_volatility_fold_plan  # noqa: E402
from volatility_forecasting.metrics import qlike_losses  # noqa: E402
from volatility_forecasting.model import (  # noqa: E402
    BaselineResidualTCNConfig,
    TorchTrainingConfig,
)
from volatility_forecasting.news import (  # noqa: E402
    NEWS_FEATURE_NAMES_V2,
    NEWS_FEATURE_SCHEMA_VERSION,
)
from volatility_forecasting.news_ablation import (  # noqa: E402
    NewsAblationGate,
    assess_news_ablation,
)
from volatility_forecasting.news_alignment import (  # noqa: E402
    aggregate_news_for_market_rows,
    validate_news_coverage,
)
from volatility_forecasting.news_exposures import load_news_exposure_map  # noqa: E402
from volatility_forecasting.news_snapshot import load_news_snapshot  # noqa: E402

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
        distribution_promoted_all_seeds = all(
            bool(decision["return_distribution_promoted"]) for decision in decisions
        )
        summary[str(horizon)] = {
            "promoted_all_seeds": promoted_all_seeds,
            "return_distribution_promoted_all_seeds": distribution_promoted_all_seeds,
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
            "return_distribution_reasons_by_seed": {
                str(record["seed"]): list(decision["return_distribution_reasons"])
                for record, decision in zip(records, decisions, strict=True)
            },
        }
    return summary


def _fold_relative_qlike(evaluation) -> np.ndarray:
    return np.asarray(
        [[float(metric["relative_qlike"]) for metric in fold.metrics] for fold in evaluation.folds],
        dtype=np.float64,
    )


def _news_ablation_consensus(
    records: list[list[dict[str, object]]],
    seeds: tuple[int, ...],
    horizons: tuple[int, ...],
) -> dict[str, object]:
    if len(records) != len(seeds):
        raise ValueError("news ablation records do not match the evaluated seeds")
    summary: dict[str, object] = {}
    for column, horizon in enumerate(horizons):
        decisions = [record[column] for record in records]
        summary[str(horizon)] = {
            "promoted_all_seeds": all(bool(decision["promoted"]) for decision in decisions),
            "relative_qlike_to_market_median": float(
                np.median([float(decision["relative_qlike_to_market"]) for decision in decisions])
            ),
            "relative_qlike_to_market_worst_seed": float(
                np.max([float(decision["relative_qlike_to_market"]) for decision in decisions])
            ),
            "reasons_by_seed": {
                str(seed): list(decision["reasons"])
                for seed, decision in zip(seeds, decisions, strict=True)
            },
        }
    return summary


def _news_gap_exclusion_mask(
    origin_dates: np.ndarray,
    manifest: dict[str, object],
    *,
    lookback_days: int = 7,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Exclude origins whose causal news lookback intersects a provider gap."""
    if lookback_days < 1:
        raise ValueError("news gap lookback must be positive")
    values = manifest.get("provenance")
    provenance = values if isinstance(values, dict) else {}
    raw_dates = provenance.get("missing_archive_dates", [])
    if not isinstance(raw_dates, list) or any(not isinstance(value, str) for value in raw_dates):
        raise ValueError("news provider-gap metadata is malformed")
    gap_dates = tuple(sorted(set(raw_dates)))
    mask = np.zeros(len(origin_dates), dtype=bool)
    dates = np.asarray(origin_dates, dtype="datetime64[D]")
    for raw_date in gap_dates:
        try:
            gap = np.datetime64(raw_date, "D")
        except ValueError as error:
            raise ValueError("news provider-gap metadata contains an invalid date") from error
        mask |= (dates >= gap) & (dates <= gap + np.timedelta64(lookback_days, "D"))
    return mask, gap_dates


def _subset_examples(
    examples: VolatilityPanelExamples,
    keep: np.ndarray,
) -> VolatilityPanelExamples:
    """Keep row identities aligned when a news provider gap is excluded."""
    selected = np.asarray(keep, dtype=bool)
    if selected.ndim != 1 or len(selected) != len(examples.features) or not selected.any():
        raise ValueError("news gap filtering would leave no valid volatility examples")
    if selected.all():
        return examples
    return replace(
        examples,
        features=examples.features[selected],
        baseline_variance=examples.baseline_variance[selected],
        realized_variance=examples.realized_variance[selected],
        cumulative_returns=examples.cumulative_returns[selected],
        direction_classes=examples.direction_classes[selected],
        tickers=examples.tickers[selected],
        origin_dates=examples.origin_dates[selected],
        origin_closes=examples.origin_closes[selected],
    )


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
        "--seeds",
        type=int,
        nargs="+",
        help="Development-only seed subset; certification requires the frozen three seeds",
    )
    parser.add_argument(
        "--example-cache-root",
        type=Path,
        help="Local derived-array cache (defaults beside the immutable panel)",
    )
    parser.add_argument("--no-example-cache", action="store_true")
    parser.add_argument(
        "--news-snapshot-dir",
        type=Path,
        help="Verified immutable event snapshot; enables a paired market-plus-news ablation",
    )
    parser.add_argument(
        "--news-exposure-map",
        type=Path,
        help="Optional JSON ticker-to-topic exposure map used by the news ablation",
    )
    parser.add_argument("--news-channels", type=int, default=24)
    parser.add_argument(
        "--encoder",
        choices=("tcn", "patch_transformer"),
        default="tcn",
        help="Sequence encoder candidate evaluated under the identical protocol",
    )
    args = parser.parse_args()

    if args.news_exposure_map is not None and args.news_snapshot_dir is None:
        parser.error("--news-exposure-map requires --news-snapshot-dir")
    if args.news_channels < 1:
        parser.error("--news-channels must be positive")

    protocol = VolatilityForecastProtocol()
    gate = VolatilityPromotionGate()
    seeds = (
        tuple(args.seeds)
        if args.seeds
        else ((protocol.seeds[1],) if args.quick else protocol.seeds)
    )
    if len(set(seeds)) != len(seeds) or not set(seeds).issubset(protocol.seeds):
        parser.error(f"--seeds must be unique members of {protocol.seeds}")
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
    compatible_cache = find_compatible_example_cache(
        cache_root,
        panel_checksum=panel_checksum,
        protocol=protocol,
    )
    examples = None
    if not args.no_example_cache and compatible_cache is not None:
        try:
            print(f"Verifying derived example cache {compatible_cache}...", flush=True)
            examples = load_example_cache(
                compatible_cache,
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
    news_features: np.ndarray | None = None
    news_manifest: dict[str, object] | None = None
    news_gap_dates: tuple[str, ...] = ()
    news_excluded_rows = 0
    exposure_map: dict[str, dict[str, float]] = {}
    exposure_metadata: dict[str, object] | None = None
    if args.news_snapshot_dir is not None:
        print(f"Verifying immutable news snapshot {args.news_snapshot_dir}...", flush=True)
        events, news_manifest = load_news_snapshot(args.news_snapshot_dir)
        if args.news_exposure_map is not None:
            loaded_exposures = load_news_exposure_map(
                args.news_exposure_map,
                required_tickers={str(ticker) for ticker in examples.tickers},
            )
            exposure_map = loaded_exposures.exposures
            exposure_metadata = {
                "schema_version": loaded_exposures.schema_version,
                "methodology": loaded_exposures.methodology,
                "source_sha256": loaded_exposures.source_sha256,
                "source_path": str(args.news_exposure_map.resolve()),
            }
        news_matrix = aggregate_news_for_market_rows(
            events,
            examples.tickers,
            examples.origin_dates,
            exposure_map=exposure_map,
        )
        validate_news_coverage(news_manifest, news_matrix.cutoffs)
        news_features = news_matrix.values
        gap_mask, news_gap_dates = _news_gap_exclusion_mask(
            examples.origin_dates,
            news_manifest,
        )
        news_excluded_rows = int(gap_mask.sum())
        if news_excluded_rows:
            examples = _subset_examples(examples, ~gap_mask)
            news_features = news_features[~gap_mask]
            print(
                f"Excluded {news_excluded_rows:,} origins whose seven-day causal news "
                "lookback intersects an explicit provider gap.",
                flush=True,
            )
        print(
            f"Aligned {len(events):,} point-in-time events into "
            f"{news_features.shape[1]} news features for {len(news_features):,} valid rows.",
            flush=True,
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
        encoder_family=args.encoder,
        window_size=protocol.window_size,
    )
    news_architecture = (
        BaselineResidualTCNConfig(
            feature_count=examples.features.shape[-1],
            horizon_count=len(protocol.horizons),
            encoder_family=args.encoder,
            window_size=protocol.window_size,
            news_feature_count=len(NEWS_FEATURE_NAMES_V2),
            news_channels=args.news_channels,
        )
        if news_features is not None
        else None
    )
    seed_records: list[dict[str, object]] = []
    news_seed_records: list[dict[str, object]] = []
    news_ablation_records: list[list[dict[str, object]]] = []
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
        if news_features is not None and news_architecture is not None:
            print(f"Evaluating market-plus-news seed {seed} on matched folds...", flush=True)
            news_evaluation = evaluate_tcn_development(
                examples,
                fold_plan,
                protocol,
                model_config=news_architecture,
                training_config=training,
                promotion_gate=gate,
                seed=seed,
                device=args.device,
                resamples=200 if args.quick else 1000,
                news_features=news_features,
            )
            if not np.array_equal(evaluation.oof_indices, news_evaluation.oof_indices):
                raise RuntimeError("market and news evaluations did not use identical OOF rows")
            news_record = _evaluation_record(news_evaluation)
            news_seed_records.append(news_record)
            (run_dir / f"news-seed-{seed}.json").write_text(
                json.dumps(news_record, indent=2, sort_keys=True, default=_json_value),
                encoding="utf-8",
            )
            indices = evaluation.oof_indices
            decisions = assess_news_ablation(
                candidate_qlike_losses=qlike_losses(
                    news_evaluation.predictions.variance,
                    examples.realized_variance[indices],
                ),
                market_qlike_losses=qlike_losses(
                    evaluation.predictions.variance,
                    examples.realized_variance[indices],
                ),
                origin_dates=examples.origin_dates[indices],
                candidate_fold_relative_qlike=_fold_relative_qlike(news_evaluation),
                market_fold_relative_qlike=_fold_relative_qlike(evaluation),
                candidate_promoted_vs_har=tuple(
                    decision.volatility_promoted for decision in news_evaluation.promotion
                ),
                horizons=protocol.horizons,
                gate=NewsAblationGate(),
                resamples=200 if args.quick else 1000,
                seed=seed,
            )
            decision_rows = [asdict(decision) for decision in decisions]
            news_ablation_records.append(decision_rows)
            (run_dir / f"news-ablation-seed-{seed}.json").write_text(
                json.dumps(decision_rows, indent=2, sort_keys=True, default=_json_value),
                encoding="utf-8",
            )
            seven_day_news = next(decision for decision in decisions if decision.horizon == 7)
            print(
                f"News seed {seed}: 7d relative QLIKE vs market-only="
                f"{seven_day_news.relative_qlike_to_market:.4f}, "
                f"promoted={seven_day_news.promoted}",
                flush=True,
            )

    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "certifiable": (
            not args.quick
            and seeds == protocol.seeds
            and args.maximum_epochs == TorchTrainingConfig().maximum_epochs
            and args.batch_size == TorchTrainingConfig().batch_size
        ),
        "mode": (
            "quick_news_ablation"
            if args.quick and news_features is not None
            else "full_news_ablation"
            if news_features is not None
            else "quick_smoke"
            if args.quick
            else "full_development"
        ),
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
    if news_features is not None and news_manifest is not None and news_architecture is not None:
        report["news"] = {
            "snapshot": news_manifest,
            "missing_archive_dates": list(news_gap_dates),
            "excluded_provider_gap_rows": news_excluded_rows,
            "feature_schema_version": NEWS_FEATURE_SCHEMA_VERSION,
            "feature_names": list(NEWS_FEATURE_NAMES_V2),
            "exposure_map": exposure_map,
            "exposure_metadata": exposure_metadata,
            "architecture": asdict(news_architecture),
            "seeds": news_seed_records,
            "ablation_gate": asdict(NewsAblationGate()),
            "ablation_by_seed": news_ablation_records,
            "seed_consensus": _news_ablation_consensus(
                news_ablation_records,
                seeds,
                protocol.horizons,
            ),
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
    if news_features is not None:
        print(
            json.dumps(report["news"]["seed_consensus"]["7"], indent=2, sort_keys=True),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
