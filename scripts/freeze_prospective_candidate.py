#!/usr/bin/env python3
"""Fit the selected v7 development profile as an unsigned candidate.

This command is deliberately separate from locked certification and release
materialization.  It consumes only the preregistered development panel and a
full, freeze-eligible prospective report.  The output is research evidence
for a later genuinely-future certification; it is not a signed serving
bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "research"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from volatility_forecasting.cache import (  # noqa: E402
    ExampleCacheError,
    find_compatible_example_cache,
    load_example_cache,
    panel_fingerprint,
)
from volatility_forecasting.data import build_volatility_panel_examples  # noqa: E402
from volatility_forecasting.folds import build_prospective_development_fold_plan  # noqa: E402
from volatility_forecasting.model import BaselineResidualTCNConfig  # noqa: E402
from volatility_forecasting.prospective import (  # noqa: E402
    OBJECTIVE_PROFILES,
    ProspectiveCycleSettings,
    prospective_protocol,
    validate_prospective_panel_manifest,
)
from volatility_forecasting.refit import fit_frozen_ensemble  # noqa: E402

from backend.panel.snapshots import load_panel_from_directory  # noqa: E402


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_normalised(value: object) -> object:
    return json.loads(json.dumps(value))


def _array_sha256(values: np.ndarray) -> str:
    array = np.asarray(values, dtype=np.int64)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _expected_oof_indices(examples, fold_plan) -> np.ndarray:
    indices = np.concatenate([fold.validation_indices for fold in fold_plan.folds])
    order = np.lexsort((examples.tickers[indices], examples.origin_dates[indices]))
    return indices[order]


def _load_examples(panel_dir: Path, cache_root: Path, protocol):
    checksum = panel_fingerprint(panel_dir)
    cache = find_compatible_example_cache(
        cache_root,
        panel_checksum=checksum,
        protocol=protocol,
    )
    if cache is not None:
        try:
            return (
                load_example_cache(cache, panel_checksum=checksum, protocol=protocol),
                checksum,
                cache,
            )
        except ExampleCacheError as error:
            raise RuntimeError("compatible example cache failed verification") from error
    panel = load_panel_from_directory(panel_dir)
    return build_volatility_panel_examples(panel, protocol), checksum, None


def _validate_report(
    report: object,
    *,
    report_bytes: bytes,
    panel_dir: Path,
    protocol,
    cycle: ProspectiveCycleSettings,
) -> tuple[dict[int, dict[str, object]], BaselineResidualTCNConfig, str, dict[str, object]]:
    if not isinstance(report, dict):
        raise ValueError("prospective development report is not an object")
    if report.get("mode") != "prospective_full_development":
        raise ValueError("only the full prospective development report can be frozen")
    if report.get("freeze_eligible") is not True:
        raise ValueError("prospective development report is not freeze-eligible")
    if report.get("consumed_holdout_reused") is not False:
        raise ValueError("prospective report does not prove holdout separation")
    if _json_normalised(report.get("protocol")) != _json_normalised(asdict(protocol)):
        raise ValueError("prospective report protocol does not match the implementation")
    if Path(str(report.get("panel_dir", ""))).resolve() != panel_dir.resolve():
        raise ValueError("prospective report and candidate panel paths differ")
    if report.get("development_cutoff") != cycle.development_cutoff:
        raise ValueError("prospective report cutoff does not match preregistration")
    if report.get("prospective_certification_start") != cycle.prospective_certification_start:
        raise ValueError("prospective report certification boundary does not match preregistration")

    selection = report.get("selection")
    if not isinstance(selection, dict) or selection.get("status") != "selected":
        raise ValueError("prospective report has no selected profile")
    selected_profile = selection.get("selected_profile")
    if selected_profile not in cycle.profile_names:
        raise ValueError("selected profile is outside the preregistered profile set")
    profile = (
        report.get("profiles", {}).get(selected_profile)
        if isinstance(report.get("profiles"), dict)
        else None
    )
    if not isinstance(profile, dict):
        raise ValueError("selected profile evidence is missing")
    objective = profile.get("objective")
    expected_objective = {
        "name": selected_profile,
        "loss_weights": asdict(OBJECTIVE_PROFILES[selected_profile].loss_weights),
        "rationale": OBJECTIVE_PROFILES[selected_profile].rationale,
    }
    if _json_normalised(objective) != _json_normalised(expected_objective):
        raise ValueError("selected profile objective is not the frozen objective")
    seeds = profile.get("seeds")
    if not isinstance(seeds, list):
        raise ValueError("selected profile seed evidence is missing")
    records: dict[int, dict[str, object]] = {}
    for record in seeds:
        if not isinstance(record, dict) or not isinstance(record.get("seed"), int):
            raise ValueError("selected profile seed evidence is malformed")
        records[int(record["seed"])] = record
    if tuple(sorted(records)) != protocol.seeds:
        raise ValueError(f"selected profile evidence must cover frozen seeds {protocol.seeds}")
    architecture_payload = report.get("architecture")
    if not isinstance(architecture_payload, dict):
        raise ValueError("prospective architecture is missing")
    architecture_payload = dict(architecture_payload)
    if isinstance(architecture_payload.get("dilations"), list):
        architecture_payload["dilations"] = tuple(architecture_payload["dilations"])
    architecture = BaselineResidualTCNConfig(**architecture_payload)
    if architecture.feature_count != protocol.feature_count:
        raise ValueError("prospective architecture feature count is incompatible")
    if architecture.horizon_count != len(protocol.horizons):
        raise ValueError("prospective architecture horizon count is incompatible")
    return records, architecture, str(selected_profile), selection


def _save_candidate(
    output: Path, ensemble, report: dict[str, object], report_digest: str, profile_name: str
) -> None:
    output.mkdir(parents=True, exist_ok=False)
    members: list[dict[str, object]] = []
    for member in ensemble.members:
        filename = f"seed-{member.seed}.pt"
        path = output / filename
        torch.save(member.training.model.state_dict(), path)
        members.append(
            {
                "seed": member.seed,
                "model_identity": member.model_identity,
                "weights_file": filename,
                "weights_sha256": _sha256_file(path),
                "epoch_budget": member.epoch_budget,
                "best_epoch": member.training.best_epoch,
                "market_scaler": member.training.scaler.to_dict(),
                "news_scaler": (
                    member.training.news_scaler.to_dict()
                    if member.training.news_scaler is not None
                    else None
                ),
                "variance_scale": member.variance_scale.tolist(),
                "return_variance_scale": member.return_variance_scale.tolist(),
                "baseline_return_variance_scale": member.baseline_return_variance_scale.tolist(),
                "comparison_baseline": [
                    asdict(value) for value in member.comparison_baseline.horizons
                ],
                "loss_weights": asdict(member.loss_weights),
                "fit_end": str(member.fit_split.fit_end),
                "calibration_start": str(member.fit_split.early_stopping_start),
                "calibration_end": str(member.fit_split.early_stopping_end),
            }
        )
    manifest = {
        "artifact_role": "prospective_development_candidate",
        "release_eligible": False,
        "model_identity": ensemble.model_identity,
        "development_report_sha256": report_digest,
        "selected_profile": profile_name,
        "objective": report["profiles"][profile_name]["objective"],
        "protocol": report["protocol"],
        "architecture": report["architecture"],
        "development_cutoff": report["development_cutoff"],
        "prospective_certification_start": report["prospective_certification_start"],
        "members": members,
        "strict_release_policy": {
            "unsigned": True,
            "partial_release_allowed": False,
            "old_locked_holdout_reusable": False,
            "future_certification_required": True,
        },
    }
    (output / "candidate-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit the selected v7 profile as an unsigned prospective candidate",
    )
    parser.add_argument("--development-report", type=Path, required=True)
    parser.add_argument("--panel-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--example-cache-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    output = args.output_dir.resolve()
    if output.exists():
        parser.error("--output-dir must not already exist; choose a new immutable path")
    panel_dir = args.panel_dir.resolve()
    report_path = args.development_report.resolve()
    report_bytes = report_path.read_bytes()
    try:
        report = json.loads(report_bytes)
    except json.JSONDecodeError as error:
        raise ValueError("prospective development report is not valid JSON") from error

    cycle = ProspectiveCycleSettings()
    protocol = prospective_protocol()
    validate_prospective_panel_manifest(panel_dir, expected_cutoff=cycle.development_cutoff)
    records, architecture, profile_name, selection = _validate_report(
        report,
        report_bytes=report_bytes,
        panel_dir=panel_dir,
        protocol=protocol,
        cycle=cycle,
    )
    examples, panel_checksum, cache = _load_examples(
        panel_dir,
        args.example_cache_root.resolve(),
        protocol,
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
    expected_oof = _array_sha256(_expected_oof_indices(examples, fold_plan))
    if report.get("oof_identity_sha256") != expected_oof:
        raise ValueError("prospective report OOF identity does not match this panel")

    print(
        f"Fitting unsigned prospective {profile_name} ensemble for seeds {protocol.seeds}...",
        flush=True,
    )
    ensemble = fit_frozen_ensemble(
        examples=examples,
        fold_plan=fold_plan,
        protocol=protocol,
        development_records=records,
        architecture=architecture,
        device=args.device,
        batch_size=args.batch_size,
        loss_weights=OBJECTIVE_PROFILES[profile_name].loss_weights,
    )
    report_digest = hashlib.sha256(report_bytes).hexdigest()
    _save_candidate(output, ensemble, report, report_digest, profile_name)
    manifest = json.loads((output / "candidate-manifest.json").read_text(encoding="utf-8"))
    manifest["panel_checksum"] = panel_checksum
    manifest["example_cache"] = str(cache) if cache is not None else None
    manifest["selection"] = selection
    manifest["oof_identity_sha256"] = expected_oof
    (output / "candidate-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "prospective_candidate_fitted",
                "release_eligible": False,
                "selected_profile": profile_name,
                "model_identity": ensemble.model_identity,
                "output": str(output),
                "future_certification_required": True,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
