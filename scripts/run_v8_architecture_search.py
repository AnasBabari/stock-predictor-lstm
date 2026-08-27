#!/usr/bin/env python3
"""v8 architecture search — validation-only hyperparameter sweep.

This script reuses the sealed-test-safe v8 pipeline (train/val only, test never
opened) and sweeps architecture, learning-rate, and baseline-shrinkage knobs.
The sealed test remains closed; the best validation-only candidate is written
to ``--out`` as a rejected development-evidence artifact unless the frozen
universe itself is certifiable.

Usage:
  python scripts/run_v8_architecture_search.py \
    --panel-dir /path/to/v8-market-snapshot \
    --universe-manifest /path/to/universe-v8-manifest.json \
    --out /tmp/v8-search \
    --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "research"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from backend.panel.snapshots import load_panel_from_directory  # noqa: E402
from research.volatility_forecasting.baselines_v8 import (  # noqa: E402
    evaluate_development_baselines,
)
from research.volatility_forecasting.cache import (  # noqa: E402
    example_cache_key,
    find_compatible_example_cache,
    load_example_cache,
    panel_fingerprint,
    save_example_cache,
)
from research.volatility_forecasting.candidate_v8 import (  # noqa: E402
    V8MemberEvidence,
    save_v8_development_candidate,
    train_v8_numeric_ensemble,
)
from research.volatility_forecasting.data import build_volatility_panel_examples  # noqa: E402
from research.volatility_forecasting.model import (  # noqa: E402
    BaselineResidualTCNConfig,
    TorchTrainingConfig,
    VolatilityLossWeights,
)
from research.volatility_forecasting.split_v8 import build_v8_chronological_split  # noqa: E402
from research.volatility_forecasting.v8_protocol import v8_manifest, v8_protocol  # noqa: E402


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="v8 architecture search (train/val only)")
    ap.add_argument("--panel-dir", type=Path, required=True)
    ap.add_argument("--universe-manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="Empty output dir for search results")
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--max-epochs", type=int, default=20, help="Epoch budget for search candidates")
    ap.add_argument("--patience", type=int, default=5, help="Early-stopping patience for search")
    ap.add_argument(
        "--skip-example-cache",
        action="store_true",
        help="Build examples directly without probing legacy cache roots",
    )
    ap.add_argument("--example-cache-root", type=Path, default=None)
    ap.add_argument(
        "--holdouts",
        type=str,
        default="NMM,MSFT",
        help="Comma-separated holdouts, must match universe policy",
    )
    ap.add_argument("--max-configs", type=int, default=12, help="Maximum configs to evaluate")
    ap.add_argument(
        "--config-json",
        type=Path,
        default=None,
        help=(
            "Optional path to a single config JSON (same schema as search-report best_config). "
            "When supplied the random search space is skipped and only this config is run."
        ),
    )
    return ap.parse_args()


def _load_examples(
    panel_dir: Path,
    protocol,
    *,
    skip_cache: bool = False,
    cache_root: Path | None = None,
):
    roots = ([cache_root] if cache_root is not None else []) + ([] if skip_cache else [
        Path(r"C:\tmp\stocklstm-volatility-panel-v1\example-cache"),
        ROOT / "research" / ".cache" / "volatility-examples",
    ])
    for root in roots:
        if not root.is_dir():
            continue
        try:
            fp = panel_fingerprint(panel_dir)
            compat = find_compatible_example_cache(root, panel_checksum=fp, protocol=protocol)
            if compat:
                return load_example_cache(compat, panel_checksum=fp, protocol=protocol), fp
        except Exception:
            continue
    fp = panel_fingerprint(panel_dir) if (panel_dir / "manifest.json").exists() else "no-checksum"
    panel = load_panel_from_directory(panel_dir)
    examples = build_volatility_panel_examples(panel, protocol)
    if cache_root is not None and fp != "no-checksum":
        save_example_cache(
            cache_root / example_cache_key(fp, protocol),
            examples,
            panel_checksum=fp,
            protocol=protocol,
        )
    return examples, fp


def _config_label(cfg: dict) -> str:
    parts = [
        cfg["encoder_family"],
        f"ch{cfg.get('channels', cfg.get('transformer_d_model', '-'))}",
        f"drop{cfg['dropout']}",
        f"lr{cfg['learning_rate']:.0e}",
        f"wd{cfg['weight_decay']:.0e}",
        f"reg{cfg['baseline_regularization']}",
    ]
    return "-".join(str(p) for p in parts)


def _loss_weights_for_config(cfg: dict) -> VolatilityLossWeights:
    regularization = float(cfg["baseline_regularization"])
    return VolatilityLossWeights(
        qlike=0.65 - regularization,
        variance_crps=0.25,
        return_location=0.05,
        direction=0.05,
        baseline_regularization=regularization,
    )


def _build_search_space(max_configs: int) -> list[dict]:
    """Return a small, diverse grid of architecture/training configs."""
    tcn_channels = [32, 48]
    patch_d_model = [64]
    dropouts = [0.15, 0.25]
    lrs = [1e-3, 3e-4]
    wds = [1e-4, 1e-3]
    regs = [0.05, 0.10]

    configs: list[dict] = []
    for family, ch_or_dmodel, drop, lr, wd, reg in itertools.product(
        ["tcn", "patch_transformer"],
        tcn_channels + patch_d_model,
        dropouts,
        lrs,
        wds,
        regs,
    ):
        if family == "tcn" and ch_or_dmodel not in tcn_channels:
            continue
        if family == "patch_transformer" and ch_or_dmodel not in patch_d_model:
            continue
        if family == "tcn":
            channels = ch_or_dmodel
            tcn_cfg = {
                "encoder_family": family,
                "channels": channels,
                "dropout": drop,
                "learning_rate": lr,
                "weight_decay": wd,
                "baseline_regularization": reg,
            }
            configs.append(tcn_cfg)
        else:
            channels = 48  # patch encoder pools to this width
            patch_cfg = {
                "encoder_family": family,
                "channels": channels,
                "transformer_d_model": ch_or_dmodel,
                "dropout": drop,
                "learning_rate": lr,
                "weight_decay": wd,
                "baseline_regularization": reg,
            }
            configs.append(patch_cfg)

    # Deterministic deduplication + cap
    seen: set[str] = set()
    unique: list[dict] = []
    for cfg in configs:
        key = _config_label(cfg)
        if key in seen:
            continue
        seen.add(key)
        unique.append(cfg)
    rng = np.random.default_rng(20260827)
    rng.shuffle(unique)
    return unique[:max_configs]


def _rank_candidates(results: list[dict]) -> list[dict]:
    """Sort candidates: eligible first, then by worst required-horizon relative QLIKE."""

    def sort_key(row: dict):
        eligible = row.get("eligible", False)
        worst_qlike = row.get("worst_required_relative_qlike", float("inf"))
        mean_qlike = row.get("mean_required_relative_qlike", float("inf"))
        worst_upper = row.get("worst_required_ratio_upper_95", float("inf"))
        # Eligible first (True > False), then lower-is-better metrics
        return (not eligible, worst_qlike, mean_qlike, worst_upper, row["label"])

    return sorted(results, key=sort_key)


def _summarize_evidence(evidence: tuple[V8MemberEvidence, ...], required_horizons: tuple[int, ...]) -> dict:
    horizons = list(required_horizons)
    qlikes: list[float] = []
    uppers: list[float] = []
    for member in evidence:
        for h_idx, _h in enumerate(horizons):
            qlikes.append(float(member.metrics[h_idx]["relative_qlike"]))
            uppers.append(float(member.ratio_upper_95[h_idx]))
    return {
        "worst_required_relative_qlike": float(np.max(qlikes)) if qlikes else float("inf"),
        "mean_required_relative_qlike": float(np.mean(qlikes)) if qlikes else float("inf"),
        "worst_required_ratio_upper_95": float(np.max(uppers)) if uppers else float("inf"),
        "mean_required_ratio_upper_95": float(np.mean(uppers)) if uppers else float("inf"),
    }


def main() -> int:
    args = _parse_args()
    panel_dir = args.panel_dir.resolve()
    uni_path = args.universe_manifest.resolve()
    out = args.out.resolve()
    if out.exists():
        print(f"--out must not exist: {out}", file=sys.stderr)
        return 2
    if args.max_epochs < 1 or args.patience < 1 or args.batch_size < 1:
        print("epoch, patience, and batch size must be positive", file=sys.stderr)
        return 2
    if args.max_configs < 1:
        print("max-configs must be positive", file=sys.stderr)
        return 2

    protocol = v8_protocol(news_enabled=False)
    manifest = v8_manifest(news_enabled=False)
    print(f"v8 architecture search protocol {protocol.protocol_version}")

    uni = json.loads(uni_path.read_text(encoding="utf-8"))
    uni_sha = uni.get("sha256")
    if not uni_sha:
        print("universe manifest missing sha256", file=sys.stderr)
        return 2

    examples, panel_fp = _load_examples(
        panel_dir,
        protocol,
        skip_cache=args.skip_example_cache,
        cache_root=args.example_cache_root.resolve() if args.example_cache_root else None,
    )
    print(f"examples {len(examples.features)} rows, {len(np.unique(examples.origin_dates))} origins")

    holdouts = tuple(sorted({t.strip().upper() for t in args.holdouts.split(",") if t.strip()}))
    if not holdouts:
        print("holdouts required", file=sys.stderr)
        return 2

    split = build_v8_chronological_split(
        examples,
        protocol=protocol,
        required_asset_holdouts=holdouts,
        universe_manifest_sha256=uni_sha,
        universe_coverage_certifiable=bool(uni.get("coverage_certifiable")),
        panel_checksum=panel_fp,
        news_snapshot_checksum="sha256:"
        + hashlib.sha256(b"no_news").hexdigest(),
    )
    print(
        f"split train {split.manifest.train_rows} val {split.manifest.validation_rows} pooled_test {split.manifest.pooled_test_rows}"
    )

    baselines = evaluate_development_baselines(
        examples,
        fit_indices=split.train_indices,
        evaluation_indices=split.validation_indices,
    )
    adaptive_qlike = float(baselines["adaptive_calibrated_har_c2c_v1"]["qlike"])
    ridge_qlike = float(baselines["ridge_log_variance"]["qlike"])
    print(f"development validation QLIKE adaptive={adaptive_qlike:.6f} ridge={ridge_qlike:.6f}")

    required_horizons = tuple(int(value) for value in manifest["required_horizons"])
    seeds = tuple(int(v) for v in protocol.seeds)
    if args.config_json is not None:
        cfg_path = args.config_json.resolve()
        if not cfg_path.is_file():
            print(f"--config-json not found: {cfg_path}", file=sys.stderr)
            return 2
        search_space = [json.loads(cfg_path.read_text(encoding="utf-8"))]
        print(f"single-config mode from {cfg_path}")
    else:
        search_space = _build_search_space(args.max_configs)
    print(f"search space size {len(search_space)} configs x {len(seeds)} seeds")

    results: list[dict] = []
    for idx, cfg in enumerate(search_space, start=1):
        label = _config_label(cfg)
        print(f"[{idx}/{len(search_space)}] {label} ...", flush=True)
        try:
            architecture = BaselineResidualTCNConfig(
                feature_count=examples.features.shape[-1],
                horizon_count=len(examples.horizons),
                window_size=examples.features.shape[1],
                encoder_family=cfg["encoder_family"],
                channels=cfg["channels"],
                dropout=cfg["dropout"],
                transformer_d_model=cfg.get("transformer_d_model", 64),
                transformer_heads=4,
                transformer_layers=2,
                transformer_feedforward=128,
                patch_length=10,
                patch_stride=5,
            )
            loss_weights = _loss_weights_for_config(cfg)
            training_config = TorchTrainingConfig(
                maximum_epochs=args.max_epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                learning_rate=cfg["learning_rate"],
                weight_decay=cfg["weight_decay"],
                use_amp=args.device == "cuda",
            )
            ensemble, evidence, partitions = train_v8_numeric_ensemble(
                examples=examples,
                train_indices=split.train_indices,
                validation_indices=split.validation_indices,
                seeds=seeds,
                required_horizons=required_horizons,
                device=args.device,
                maximum_epochs=args.max_epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                architecture=architecture,
                loss_weights=loss_weights,
                training_config=training_config,
            )
            summary = _summarize_evidence(evidence, required_horizons)
            eligible = all(m.eligible for m in evidence)
            row = {
                "label": label,
                "config": cfg,
                "eligible": eligible,
                "members": [
                    {
                        "seed": m.seed,
                        "eligible": m.eligible,
                        "best_epoch": m.best_epoch,
                        "duration_seconds": m.duration_seconds,
                        "reasons": list(m.reasons),
                    }
                    for m in evidence
                ],
                "worst_required_relative_qlike": summary["worst_required_relative_qlike"],
                "mean_required_relative_qlike": summary["mean_required_relative_qlike"],
                "worst_required_ratio_upper_95": summary["worst_required_ratio_upper_95"],
                "mean_required_ratio_upper_95": summary["mean_required_ratio_upper_95"],
                "status": "ok",
            }
            results.append(row)
            print(
                f"  -> eligible={eligible} worst_rel_qlike={summary['worst_required_relative_qlike']:.6f} "
                f"worst_upper95={summary['worst_required_ratio_upper_95']:.6f}",
                flush=True,
            )
        except Exception as exc:  # pragma: no cover — robustness path
            row = {
                "label": label,
                "config": cfg,
                "eligible": False,
                "members": [],
                "worst_required_relative_qlike": float("inf"),
                "mean_required_relative_qlike": float("inf"),
                "worst_required_ratio_upper_95": float("inf"),
                "mean_required_ratio_upper_95": float("inf"),
                "status": f"error: {exc}",
            }
            results.append(row)
            print(f"  -> FAILED: {exc}", flush=True)

    ranked = _rank_candidates(results)
    out.mkdir(parents=True)

    # Save full search report
    report = {
        "protocol_version": protocol.protocol_version,
        "seeds": list(seeds),
        "required_horizons": list(required_horizons),
        "baseline_metrics": baselines,
        "search_budget": {
            "maximum_epochs": args.max_epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "device": args.device,
        },
        "candidates": ranked,
        "best_label": ranked[0]["label"] if ranked else None,
        "best_eligible": ranked[0]["eligible"] if ranked else False,
    }
    (out / "search-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    # Save best config
    best = ranked[0] if ranked else None
    if best:
        (out / "best-config.json").write_text(
            json.dumps(best["config"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    # If any candidate is eligible, save it as the best development candidate
    eligible = [r for r in ranked if r["eligible"]]
    if eligible:
        best_row = eligible[0]
        label = best_row["label"]
        print(f"saving best eligible candidate: {label}", flush=True)
        # Reconstruct architecture/loss_weights/training_config from stored config
        cfg = best_row["config"]
        architecture = BaselineResidualTCNConfig(
            feature_count=examples.features.shape[-1],
            horizon_count=len(examples.horizons),
            window_size=examples.features.shape[1],
            encoder_family=cfg["encoder_family"],
            channels=cfg["channels"],
            dropout=cfg["dropout"],
            transformer_d_model=cfg.get("transformer_d_model", 64),
            transformer_heads=4,
            transformer_layers=2,
            transformer_feedforward=128,
            patch_length=10,
            patch_stride=5,
        )
        loss_weights = _loss_weights_for_config(cfg)
        training_config = TorchTrainingConfig(
            maximum_epochs=args.max_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            learning_rate=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
            use_amp=args.device == "cuda",
        )
        # We need to re-train to get the ensemble object for saving.
        # This is the only place we duplicate training, and only for the winner.
        ensemble, evidence, partitions = train_v8_numeric_ensemble(
            examples=examples,
            train_indices=split.train_indices,
            validation_indices=split.validation_indices,
            seeds=seeds,
            required_horizons=required_horizons,
            device=args.device,
            maximum_epochs=args.max_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            architecture=architecture,
            loss_weights=loss_weights,
            training_config=training_config,
        )
        split_payload = split.manifest.__dict__
        split_digest = hashlib.sha256(
            json.dumps(split_payload, sort_keys=True, default=str).encode()
        ).hexdigest()
        news_checksum = str(split.manifest.news_snapshot_checksum)
        universe_certifiable = bool(uni.get("coverage_certifiable")) and bool(
            split.manifest.coverage_certifiable
        )
        best_out = out / "best-candidate"
        if best_out.exists():
            raise FileExistsError(f"best candidate output already exists: {best_out}")
        save_v8_development_candidate(
            best_out,
            ensemble=ensemble,
            evidence=evidence,
            protocol=manifest,
            split_manifest=split_payload,
            split_manifest_sha256=split_digest,
            panel_checksum=panel_fp,
            universe_manifest_sha256=str(uni_sha),
            news_snapshot_checksum=news_checksum,
            universe_certifiable=universe_certifiable,
        )
        (best_out / "split-v8-manifest.json").write_text(
            json.dumps(split_payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        (best_out / "universe-v8-manifest.json").write_text(
            json.dumps(uni, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        best_report = {
            "artifact_role": "rejected_v8_development_evidence",
            "model_identity": "search-winner-" + label,
            "baseline_metrics": baselines,
            "validation_calibration_end": str(partitions.calibration_end),
            "validation_selection_start": str(partitions.selection_start),
            "members": [
                {
                    "seed": row.seed,
                    "eligible": row.eligible,
                    "best_epoch": row.best_epoch,
                    "duration_seconds": row.duration_seconds,
                    "reasons": list(row.reasons),
                }
                for row in evidence
            ],
            "sealed_test_opened": False,
        }
        (best_out / "development-report.json").write_text(
            json.dumps(best_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"best eligible candidate saved to {best_out}")
    else:
        print("no eligible candidate found in search space")

    print(f"search report saved to {out / 'search-report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
