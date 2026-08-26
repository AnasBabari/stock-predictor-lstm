#!/usr/bin/env python3
"""Run the pre-registered v7 prospective volatility objective comparison."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

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
from volatility_forecasting.contracts import VolatilityPromotionGate  # noqa: E402
from volatility_forecasting.data import build_volatility_panel_examples  # noqa: E402
from volatility_forecasting.evaluation import evaluate_tcn_development  # noqa: E402
from volatility_forecasting.folds import (  # noqa: E402
    build_prospective_development_fold_plan,
)
from volatility_forecasting.model import (  # noqa: E402
    BaselineResidualTCNConfig,
    TorchTrainingConfig,
)
from volatility_forecasting.prospective import (  # noqa: E402
    OBJECTIVE_PROFILES,
    ProspectiveCycleSettings,
    objective_manifest,
    prospective_protocol,
    select_prospective_profile,
    validate_prospective_panel_manifest,
)
from volatility_forecasting.resume import (  # noqa: E402
    ProspectiveResumeError,
    atomic_write_json,
    build_run_manifest,
    expected_fold_summaries,
    expected_oof_identity,
    record_filename,
    scan_run_directory,
    validate_seed_record,
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


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _evaluation_record(evaluation) -> dict[str, object]:
    return {
        "protocol_version": evaluation.protocol_version,
        "seed": evaluation.seed,
        "folds": [asdict(fold) for fold in evaluation.folds],
        "pooled_metrics": list(evaluation.pooled_metrics),
        "promotion": [asdict(decision) for decision in evaluation.promotion],
        "oof_rows": int(len(evaluation.oof_indices)),
        "oof_identity": {
            "first_index": int(evaluation.oof_indices.min()),
            "last_index": int(evaluation.oof_indices.max()),
            "sha256": _array_sha256(evaluation.oof_indices),
        },
    }


def _array_sha256(values: np.ndarray) -> str:
    import hashlib

    array = np.asarray(values, dtype=np.int64)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _seed_consensus(
    records: list[dict[str, object]], horizons: tuple[int, ...]
) -> dict[str, object]:
    summary: dict[str, object] = {}
    for column, horizon in enumerate(horizons):
        decisions = [record["promotion"][column] for record in records]
        metrics = [record["pooled_metrics"][column] for record in records]
        summary[str(horizon)] = {
            "promoted_all_seeds": all(bool(decision["promoted"]) for decision in decisions),
            "return_distribution_promoted_all_seeds": all(
                bool(decision["return_distribution_promoted"]) for decision in decisions
            ),
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
        description="Evaluate the frozen v7 prospective volatility objectives on CUDA",
    )
    parser.add_argument("--panel-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--example-cache-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--maximum-epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--quick", action="store_true", help="Three-epoch seed-42 screen only")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Fail-closed resume of an interrupted full run: every existing seed "
            "record must validate exactly against the requested configuration "
            "before it is skipped; anything unexpected aborts"
        ),
    )
    parser.add_argument(
        "--accept-legacy-records",
        action="store_true",
        help=(
            "Accept records written before embedded run manifests existed. Their "
            "panel and objective identity cannot be verified from the file alone, "
            "so this flag is an explicit operator attestation for an audited "
            "checkpoint; all recomputable checks are still enforced"
        ),
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=tuple(OBJECTIVE_PROFILES),
        default=list(OBJECTIVE_PROFILES),
    )
    args = parser.parse_args()
    if args.maximum_epochs < 1 or args.batch_size < 1:
        parser.error("epoch and batch settings must be positive")
    if len(set(args.profiles)) != len(args.profiles):
        parser.error("profiles must be unique")
    if args.accept_legacy_records and not args.resume:
        parser.error("--accept-legacy-records is only meaningful with --resume")
    defaults = TorchTrainingConfig()
    if args.resume and (
        args.quick
        or args.maximum_epochs != defaults.maximum_epochs
        or args.batch_size != defaults.batch_size
        or tuple(args.profiles) != ProspectiveCycleSettings().profile_names
    ):
        parser.error(
            "--resume supports only the exact preregistered full comparison "
            "(both profiles, default epochs and batch size, no --quick)"
        )
    output = args.run_dir.resolve()
    if args.resume:
        if not output.is_dir():
            parser.error("--resume requires an existing run directory")
    elif output.exists() and any(output.iterdir()):
        parser.error("--run-dir must not exist or must be empty (use --resume)")
    output.mkdir(parents=True, exist_ok=True)

    cycle = ProspectiveCycleSettings()
    protocol = prospective_protocol()
    panel_dir = args.panel_dir.resolve()
    panel_manifest = validate_prospective_panel_manifest(
        panel_dir,
        expected_cutoff=cycle.development_cutoff,
    )
    panel_checksum = panel_fingerprint(panel_dir)
    cache_root = args.example_cache_root.resolve()
    cache_dir = cache_root / example_cache_key(panel_checksum, protocol)
    compatible_cache = find_compatible_example_cache(
        cache_root,
        panel_checksum=panel_checksum,
        protocol=protocol,
    )
    examples = None
    if compatible_cache is not None:
        try:
            print(f"Verifying prospective example cache {compatible_cache}...", flush=True)
            examples = load_example_cache(
                compatible_cache,
                panel_checksum=panel_checksum,
                protocol=protocol,
            )
        except ExampleCacheError as error:
            print(f"Ignoring invalid prospective cache: {error}", flush=True)
    if examples is None:
        print(f"Loading immutable panel {panel_dir}...", flush=True)
        panel = load_panel_from_directory(panel_dir)
        print(f"Building prospective causal examples for {len(panel)} tickers...", flush=True)
        examples = build_volatility_panel_examples(panel, protocol)
        save_example_cache(
            cache_dir,
            examples,
            panel_checksum=panel_checksum,
            protocol=protocol,
        )

    fold_plan = build_prospective_development_fold_plan(
        examples,
        protocol,
        development_cutoff=np.datetime64(cycle.development_cutoff, "D"),
        prospective_certification_start=np.datetime64(
            cycle.prospective_certification_start,
            "D",
        ),
    )
    seeds = (42,) if args.quick else protocol.seeds
    training = TorchTrainingConfig(
        maximum_epochs=min(args.maximum_epochs, 3) if args.quick else args.maximum_epochs,
        patience=2 if args.quick else 8,
        batch_size=args.batch_size,
        use_amp=args.device == "cuda",
    )
    gate = VolatilityPromotionGate()
    architecture = BaselineResidualTCNConfig(
        feature_count=examples.features.shape[-1],
        horizon_count=len(protocol.horizons),
        encoder_family="tcn",
        window_size=protocol.window_size,
    )
    resamples = 200 if args.quick else 1000
    # JSON round-trip once so stored manifests and expected values compare
    # identically (tuples such as TCN dilations become lists on disk).
    architecture_manifest = json.loads(json.dumps(asdict(architecture)))
    planned_identity = expected_oof_identity(examples.tickers, examples.origin_dates, fold_plan)
    planned_folds = expected_fold_summaries(examples, fold_plan, protocol)

    existing_records: dict[tuple[str, int], Path] = {}
    if args.resume:
        try:
            existing_records = scan_run_directory(output, tuple(args.profiles), seeds)
        except ProspectiveResumeError as error:
            raise SystemExit(f"resume rejected: {error}") from error
        print(
            f"Resume inventory accepted: {len(existing_records)} existing record(s) "
            f"of {len(args.profiles) * len(seeds)} required.",
            flush=True,
        )

    def _validated(record: object, profile, seed: int, *, legacy_ok: bool) -> dict[str, object]:
        return validate_seed_record(
            record,
            profile_name=profile.name,
            loss_weights=profile.loss_weights,
            seed=seed,
            protocol=protocol,
            training=training,
            oof_identity=planned_identity,
            fold_summaries=planned_folds,
            panel_checksum=panel_checksum,
            device=args.device,
            resamples=resamples,
            architecture_manifest=architecture_manifest,
            accept_missing_manifest=legacy_ok,
        )

    profile_reports: dict[str, dict[str, object]] = {}
    resumed_names: list[str] = []
    legacy_names: list[str] = []
    for profile_name in args.profiles:
        profile = OBJECTIVE_PROFILES[profile_name]
        records: list[dict[str, object]] = []
        for seed in seeds:
            record_path = output / record_filename(profile_name, seed)
            existing = existing_records.get((profile_name, seed))
            if existing is not None:
                print(f"Validating existing record {existing.name}...", flush=True)
                try:
                    loaded = json.loads(existing.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as error:
                    raise SystemExit(
                        f"resume rejected: {existing.name} is unreadable or truncated: {error}"
                    ) from error
                try:
                    record = _validated(
                        loaded,
                        profile,
                        seed,
                        legacy_ok=args.accept_legacy_records,
                    )
                except ProspectiveResumeError as error:
                    raise SystemExit(f"resume rejected: {error}") from error
                resumed_names.append(existing.name)
                if "run_manifest" not in record:
                    legacy_names.append(existing.name)
            else:
                print(
                    f"Evaluating {profile_name} seed {seed} on {len(fold_plan.folds)} folds...",
                    flush=True,
                )
                evaluation = evaluate_tcn_development(
                    examples,
                    fold_plan,
                    protocol,
                    model_config=architecture,
                    training_config=training,
                    loss_weights=profile.loss_weights,
                    promotion_gate=gate,
                    seed=seed,
                    device=args.device,
                    resamples=resamples,
                )
                fresh = _evaluation_record(evaluation)
                fresh["run_manifest"] = build_run_manifest(
                    profile_name=profile_name,
                    loss_weights=profile.loss_weights,
                    seed=seed,
                    quick=bool(args.quick),
                    panel_checksum=panel_checksum,
                    panel_id=panel_manifest.get("panel_id"),
                    device=args.device,
                    resamples=resamples,
                    training=training,
                    architecture_manifest=architecture_manifest,
                    protocol=protocol,
                    context={
                        "git_commit": _git_head(),
                        "created_at_utc": datetime.now(UTC).isoformat(),
                        "torch": torch.__version__,
                        "gpu": (
                            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                        ),
                    },
                )
                # Round-trip through JSON so the validated object is exactly
                # what a future resume would read back from disk.
                record = json.loads(
                    json.dumps(fresh, indent=2, sort_keys=True, default=_json_value)
                )
                if not args.quick:
                    record = _validated(record, profile, seed, legacy_ok=False)
                atomic_write_json(record_path, record)
            identity = str(record["oof_identity"]["sha256"])
            if identity != planned_identity["sha256"]:
                raise RuntimeError("objective profiles did not use identical OOF rows")
            records.append(record)
            seven_day = record["promotion"][protocol.horizons.index(7)]
            print(
                f"{profile_name} seed {seed}: 7d relative QLIKE="
                f"{float(seven_day['relative_qlike']):.4f}, "
                f"promoted={bool(seven_day['promoted'])}",
                flush=True,
            )
        profile_reports[profile_name] = {
            "objective": objective_manifest(profile),
            "seeds": records,
            "seed_consensus": _seed_consensus(records, protocol.horizons),
        }

    full_evidence = (
        not args.quick
        and tuple(args.profiles) == cycle.profile_names
        and seeds == protocol.seeds
        and args.maximum_epochs == TorchTrainingConfig().maximum_epochs
        and args.batch_size == TorchTrainingConfig().batch_size
    )
    if full_evidence:
        selection = select_prospective_profile(
            {name: report["seed_consensus"] for name, report in profile_reports.items()},
            cycle,
        )
    else:
        selection = {
            "status": "screen_only",
            "selected_profile": None,
            "reasons": ["quick or non-default evidence cannot freeze a prospective candidate"],
        }

    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_head(),
        "mode": "prospective_full_development" if full_evidence else "prospective_screen",
        "freeze_eligible": full_evidence and selection["status"] == "selected",
        "consumed_holdout_reused": False,
        "panel_dir": str(panel_dir),
        "panel_id": panel_manifest.get("panel_id"),
        "panel_checksum": panel_checksum,
        "development_cutoff": cycle.development_cutoff,
        "prospective_certification_start": cycle.prospective_certification_start,
        "required_horizons": list(cycle.required_horizons),
        "protocol": asdict(protocol),
        "promotion_gate": asdict(gate),
        "architecture": asdict(architecture),
        "training": asdict(training),
        "runtime": {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": args.device,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "example_rows": len(examples.features),
        "train_tickers": list(fold_plan.train_tickers),
        "asset_holdout_tickers": list(fold_plan.asset_holdout_tickers),
        "folds": [
            {
                "fold": fold.fold,
                "train_end": str(fold.train_end),
                "validation_start": str(fold.validation_start),
                "validation_end": str(fold.validation_end),
                "train_rows": len(fold.train_indices),
                "validation_rows": len(fold.validation_indices),
            }
            for fold in fold_plan.folds
        ],
        "oof_identity_sha256": planned_identity["sha256"],
        "resume": bool(args.resume),
        "resumed_records": sorted(resumed_names),
        "legacy_records_accepted": sorted(legacy_names),
        "profiles": profile_reports,
        "selection": selection,
        "strict_release_policy": {
            "partial_release_allowed": False,
            "old_locked_holdout_reusable": False,
            "future_certification_required": True,
        },
    }
    expected_record_count = len(args.profiles) * len(seeds)
    actual_record_count = sum(len(report_row["seeds"]) for report_row in profile_reports.values())
    if actual_record_count != expected_record_count:
        raise RuntimeError(
            f"refusing to write a final report with {actual_record_count} of "
            f"{expected_record_count} validated records"
        )
    report_path = output / "prospective-development-report.json"
    atomic_write_json(report_path, report, default=_json_value)
    print(json.dumps(selection, indent=2, sort_keys=True), flush=True)
    print(f"Prospective report: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
