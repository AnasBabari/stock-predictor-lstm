#!/usr/bin/env python3
"""v8 research runner — chronological 70/15/15, purge/embargo-clean, asset-transfer.

This is the v8 analogue of ``scripts/run_prospective_volatility_research.py``
but for the sealed historical split.  It trains only on train, validates
only on validation, and never opens the test set.  The test set is sealed
until ``scripts/certify_v8_candidate.py``.

The numeric path trains real baseline-residual TCN members with CUDA when
requested. A candidate is certifiable only when both its validation gates and
the frozen universe coverage contract pass.

Usage:
  python scripts/run_v8_volatility_research.py \
    --panel-dir /path/to/v8-market-snapshot \
    --universe-manifest /path/to/universe-v8-manifest.json \
    --out /tmp/v8-candidate \
    --news-enabled false
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "research"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# PyTorch requires this before its first CUDA import for deterministic cuBLAS
# workspace selection on CUDA 10.2 and newer.
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
    save_v8_development_candidate,
    train_v8_ensemble,
    train_v8_numeric_ensemble,
)
from research.volatility_forecasting.data import build_volatility_panel_examples  # noqa: E402
from research.volatility_forecasting.market_snapshot_v8 import (  # noqa: E402
    verify_v8_market_snapshot,
)
from research.volatility_forecasting.model import (  # noqa: E402
    BaselineResidualTCNConfig,
    TorchTrainingConfig,
)
from research.volatility_forecasting.news_candidate_v8 import (  # noqa: E402
    evaluate_v8_news_ablation,
)
from research.volatility_forecasting.news_matrix_v8 import (  # noqa: E402
    build_v8_aligned_news_matrix,
)
from research.volatility_forecasting.split_v8 import (  # noqa: E402
    build_v8_chronological_split,
    build_v8_development_fold_plan,
)
from research.volatility_forecasting.universe_v8 import (  # noqa: E402
    universe_identity_maps,
    verify_universe_manifest,
)
from research.volatility_forecasting.v8_protocol import v8_manifest, v8_protocol  # noqa: E402


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="v8 research runner (train/val only, test sealed)")
    ap.add_argument("--panel-dir", type=Path, required=True)
    ap.add_argument("--universe-manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="Empty output dir for candidate")
    ap.add_argument("--news-enabled", type=lambda x: str(x).lower() == "true", default=False)
    ap.add_argument("--news-snapshot-dir", type=Path)
    ap.add_argument("--news-manifest", type=Path)
    ap.add_argument("--ticker-aliases", type=Path)
    ap.add_argument("--news-exposures", type=Path)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--maximum-epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=512)
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
    return ap.parse_args()


def _load_examples(
    panel_dir: Path,
    protocol,
    *,
    skip_cache: bool = False,
    cache_root: Path | None = None,
):
    roots = ([cache_root] if cache_root is not None else []) + (
        []
        if skip_cache
        else [
            Path(r"C:\tmp\stocklstm-volatility-panel-v1\example-cache"),
            ROOT / "research" / ".cache" / "volatility-examples",
        ]
    )
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


def main() -> int:
    args = _parse_args()
    panel_dir = args.panel_dir.resolve()
    uni_path = args.universe_manifest.resolve()
    out = args.out.resolve()
    if out.exists():
        print(f"--out must not exist (candidate directories are immutable): {out}", file=sys.stderr)
        return 2
    if args.maximum_epochs < 1 or args.patience < 1 or args.batch_size < 1:
        print("epoch, patience, and batch size must be positive", file=sys.stderr)
        return 2
    news_paths = (
        args.news_snapshot_dir,
        args.news_manifest,
        args.ticker_aliases,
        args.news_exposures,
    )
    if args.news_enabled and any(path is None for path in news_paths):
        print(
            "news-enabled v8 requires --news-snapshot-dir, --news-manifest, "
            "--ticker-aliases, and --news-exposures",
            file=sys.stderr,
        )
        return 2
    if not args.news_enabled and any(path is not None for path in news_paths):
        print("news inputs are only valid with --news-enabled true", file=sys.stderr)
        return 2

    holdouts = tuple(sorted({t.strip().upper() for t in args.holdouts.split(",") if t.strip()}))
    if not holdouts:
        print("holdouts required", file=sys.stderr)
        return 2

    protocol = v8_protocol(news_enabled=args.news_enabled)
    manifest = v8_manifest(news_enabled=args.news_enabled)
    print(f"v8 protocol {protocol.protocol_version} news_enabled={args.news_enabled}")
    print(f"holdouts {holdouts}")

    uni = verify_universe_manifest(json.loads(uni_path.read_text(encoding="utf-8")))
    uni_sha = uni.get("sha256")
    if not uni_sha:
        print("universe manifest missing sha256", file=sys.stderr)
        return 2
    exchange_map, security_id_map = universe_identity_maps(uni)
    try:
        market_manifest, _market_frames = verify_v8_market_snapshot(
            panel_dir,
            universe_manifest=uni,
            require_certifiable=False,
        )
    except ValueError as error:
        print(f"v8 market snapshot verification failed: {error}", file=sys.stderr)
        return 2

    examples, panel_fp = _load_examples(
        panel_dir,
        protocol,
        skip_cache=args.skip_example_cache,
        cache_root=args.example_cache_root.resolve() if args.example_cache_root else None,
    )
    print(
        f"examples {len(examples.features)} rows, {len(np.unique(examples.origin_dates))} origins"
    )

    aligned_news = None
    if args.news_enabled:
        assert all(path is not None for path in news_paths)
        news_manifest = json.loads(args.news_manifest.resolve().read_text(encoding="utf-8"))
        aligned_news = build_v8_aligned_news_matrix(
            examples,
            news_snapshot_dir=args.news_snapshot_dir.resolve(),
            news_manifest=news_manifest,
            universe_manifest=uni,
            market_manifest=market_manifest,
            ticker_aliases_path=args.ticker_aliases.resolve(),
            exposure_map_path=args.news_exposures.resolve(),
        )
        print(f"news matrix {aligned_news.values.shape} {aligned_news.matrix_sha256[:24]}")

    # Build chronological split — explicit holdouts, no test access beyond manifest
    split = build_v8_chronological_split(
        examples,
        protocol=protocol,
        required_asset_holdouts=holdouts,
        universe_manifest_sha256=uni_sha,
        universe_coverage_certifiable=bool(uni.get("coverage_certifiable")),
        panel_checksum=panel_fp,
        news_snapshot_checksum=(
            aligned_news.snapshot_sha256
            if aligned_news is not None
            else "sha256:" + hashlib.sha256(b"no_news").hexdigest()
        ),
        asset_exchange_map=exchange_map,
        asset_security_id_map=security_id_map,
    )
    print(
        f"split train {split.manifest.train_rows} val {split.manifest.validation_rows} pooled_test {split.manifest.pooled_test_rows}"
    )
    # Development baselines use train for fitting and validation for evaluation.
    # No sealed-test target or baseline is read here.
    baselines = evaluate_development_baselines(
        examples,
        fit_indices=split.train_indices,
        evaluation_indices=split.validation_indices,
    )
    adaptive_qlike = float(baselines["adaptive_calibrated_har_c2c_v1"]["qlike"])
    ridge_qlike = float(baselines["ridge_log_variance"]["qlike"])
    print(f"development validation QLIKE adaptive={adaptive_qlike:.6f} ridge={ridge_qlike:.6f}")

    print(
        f"training seeds {protocol.seeds} on {args.device} "
        f"epochs<={args.maximum_epochs} batch={args.batch_size}",
        flush=True,
    )
    training_config = TorchTrainingConfig(
        maximum_epochs=args.maximum_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        use_amp=args.device == "cuda",
    )
    architecture = BaselineResidualTCNConfig(
        feature_count=examples.features.shape[-1],
        horizon_count=len(examples.horizons),
        window_size=examples.features.shape[1],
    )
    ablation_payload = None
    if aligned_news is not None:
        fold_plan = build_v8_development_fold_plan(examples, split, protocol)
        ablation = evaluate_v8_news_ablation(
            examples=examples,
            fold_plan=fold_plan,
            protocol=protocol,
            news_features=aligned_news.values,
            seeds=protocol.seeds,
            market_architecture=architecture,
            training_config=training_config,
            device=args.device,
        )
        ablation_payload = tuple(asdict(row) for row in ablation)
        architecture = replace(
            architecture,
            news_feature_count=aligned_news.values.shape[1],
        )
        ensemble, evidence, partitions = train_v8_ensemble(
            examples=examples,
            train_indices=split.train_indices,
            validation_indices=split.validation_indices,
            seeds=protocol.seeds,
            required_horizons=tuple(int(value) for value in manifest["required_horizons"]),
            device=args.device,
            maximum_epochs=args.maximum_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            architecture=architecture,
            training_config=training_config,
            news_features=aligned_news.values,
        )
    else:
        ensemble, evidence, partitions = train_v8_numeric_ensemble(
            examples=examples,
            train_indices=split.train_indices,
            validation_indices=split.validation_indices,
            seeds=protocol.seeds,
            required_horizons=tuple(int(value) for value in manifest["required_horizons"]),
            device=args.device,
            maximum_epochs=args.maximum_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            architecture=architecture,
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
    candidate = save_v8_development_candidate(
        out,
        ensemble=ensemble,
        evidence=evidence,
        protocol=manifest,
        split_manifest=split_payload,
        split_manifest_sha256=split_digest,
        panel_checksum=panel_fp,
        universe_manifest_sha256=str(uni_sha),
        news_snapshot_checksum=news_checksum,
        universe_certifiable=universe_certifiable,
        training_config=asdict(training_config),
        news_matrix_sha256=(aligned_news.matrix_sha256 if aligned_news is not None else None),
        news_feature_names=(aligned_news.feature_names if aligned_news is not None else ()),
        news_ablation_evidence=ablation_payload,
    )
    (out / "split-v8-manifest.json").write_text(
        json.dumps(split_payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    (out / "universe-v8-manifest.json").write_text(
        json.dumps(uni, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "artifact_role": candidate["artifact_role"],
        "model_identity": candidate["model_identity"],
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
        "news_ablation": ablation_payload,
        "sealed_test_opened": False,
    }
    (out / "development-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if candidate["artifact_role"] == "prospective_v8_development_candidate" else 1


if __name__ == "__main__":
    raise SystemExit(main())
