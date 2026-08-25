#!/usr/bin/env python3
"""Freeze, open, and record the locked global-volatility certification once."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

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
from volatility_forecasting.certification import (  # noqa: E402
    LockedPopulationInput,
    certify_locked_predictions,
)
from volatility_forecasting.contracts import VolatilityForecastProtocol  # noqa: E402
from volatility_forecasting.data import build_volatility_panel_examples  # noqa: E402
from volatility_forecasting.folds import build_volatility_fold_plan  # noqa: E402
from volatility_forecasting.model import BaselineResidualTCNConfig  # noqa: E402
from volatility_forecasting.refit import fit_frozen_ensemble  # noqa: E402

from backend.panel.snapshots import load_panel_from_directory  # noqa: E402


def _canonical_digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_normalised(value: object) -> object:
    """Round-trip through JSON so tuple-valued fields compare equal to lists."""
    return json.loads(json.dumps(value))


def validate_development_report(
    report: object,
    protocol: VolatilityForecastProtocol,
) -> tuple[dict[int, dict[str, object]], BaselineResidualTCNConfig, tuple[int, ...]]:
    """Fail closed unless the report can predeclare a certification candidate."""
    if not isinstance(report, dict) or report.get("certifiable") is not True:
        raise ValueError("development report is absent or non-certifiable")
    if _json_normalised(report.get("protocol")) != _json_normalised(asdict(protocol)):
        raise ValueError("development report protocol does not match the frozen implementation")
    if report.get("mode") != "full_development":
        raise ValueError("this certification command accepts the market-only control report")
    architecture_payload = report.get("architecture")
    if not isinstance(architecture_payload, dict):
        raise ValueError("development report architecture is missing")
    normalised = _json_normalised(architecture_payload)
    if isinstance(normalised.get("dilations"), list):
        normalised["dilations"] = tuple(normalised["dilations"])
    architecture = BaselineResidualTCNConfig(**normalised)
    if architecture.feature_count != protocol.feature_count:
        raise ValueError("development architecture feature count is incompatible")
    records = report.get("seeds")
    if not isinstance(records, list):
        raise ValueError("development report seed evidence is missing")
    by_seed: dict[int, dict[str, object]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("seed"), int):
            raise ValueError("development seed evidence is malformed")
        by_seed[int(record["seed"])] = record
    if tuple(sorted(by_seed)) != protocol.seeds:
        raise ValueError(f"development evidence must cover frozen seeds {protocol.seeds}")
    consensus = report.get("seed_consensus")
    if not isinstance(consensus, dict):
        raise ValueError("development seed consensus is missing")
    eligible = tuple(
        horizon
        for horizon in protocol.horizons
        if isinstance(consensus.get(str(horizon)), dict)
        and consensus[str(horizon)].get("promoted_all_seeds") is True
    )
    if not eligible:
        raise ValueError("no horizon passed development across every frozen seed")
    return by_seed, architecture, eligible


def _save_passed_candidate(output_dir: Path, ensemble, report: dict[str, object]) -> None:
    candidate_dir = output_dir / "candidate"
    candidate_dir.mkdir(parents=True, exist_ok=False)
    members: list[dict[str, object]] = []
    for member in ensemble.members:
        filename = f"seed-{member.seed}.pt"
        path = candidate_dir / filename
        torch.save(member.training.model.state_dict(), path)
        members.append(
            {
                "seed": member.seed,
                "model_identity": member.model_identity,
                "weights_file": filename,
                "weights_sha256": _canonical_digest(path.read_bytes()),
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
                "baseline_return_variance_scale": (member.baseline_return_variance_scale.tolist()),
                "comparison_baseline": [
                    asdict(value) for value in member.comparison_baseline.horizons
                ],
                "fit_end": str(member.fit_split.fit_end),
                "calibration_start": str(member.fit_split.early_stopping_start),
                "calibration_end": str(member.fit_split.early_stopping_end),
            }
        )
    manifest = {
        "artifact_role": "locked_certification_candidate",
        "model_identity": ensemble.model_identity,
        "protocol": report["protocol"],
        "architecture": report["architecture"],
        "members": members,
    }
    (candidate_dir / "candidate-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit the frozen seed ensemble and open both locked reserves exactly once"
    )
    parser.add_argument("--development-report", type=Path, required=True)
    parser.add_argument("--panel-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--example-cache-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--open-locked-holdout",
        action="store_true",
        help="Required acknowledgement: this consumes the one-shot certification evidence",
    )
    args = parser.parse_args()
    if not args.open_locked_holdout:
        parser.error("--open-locked-holdout is required")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("--output-dir must not exist or must be empty")
    output.mkdir(parents=True, exist_ok=True)

    report_bytes = args.development_report.read_bytes()
    try:
        report = json.loads(report_bytes)
    except json.JSONDecodeError as error:
        raise ValueError("development report is not valid JSON") from error
    protocol = VolatilityForecastProtocol()
    records, architecture, eligible = validate_development_report(report, protocol)
    panel_checksum = panel_fingerprint(args.panel_dir)
    report_panel = Path(str(report.get("panel_dir", ""))).resolve()
    if report_panel != args.panel_dir.resolve():
        raise ValueError("development report and certification panel paths differ")
    cache = find_compatible_example_cache(
        args.example_cache_root,
        panel_checksum=panel_checksum,
        protocol=protocol,
    )
    if cache is None:
        panel = load_panel_from_directory(args.panel_dir)
        examples = build_volatility_panel_examples(panel, protocol)
    else:
        try:
            examples = load_example_cache(
                cache,
                panel_checksum=panel_checksum,
                protocol=protocol,
            )
        except ExampleCacheError as error:
            raise RuntimeError("compatible example cache failed verification") from error
    fold_plan = build_volatility_fold_plan(examples, protocol)
    print(
        f"Fitting frozen {architecture.encoder_family} ensemble for seeds {protocol.seeds}...",
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
    )
    development_digest = _canonical_digest(report_bytes)
    # Durable one-shot marker is written before any holdout prediction. If the
    # process crashes after this point, the non-empty output directory blocks a
    # rerun that could otherwise turn the reserve into iterative evidence.
    (output / "holdout-opened.json").write_text(
        json.dumps(
            {
                "model_identity": ensemble.model_identity,
                "development_evidence_sha256": development_digest,
                "eligible_horizons": eligible,
                "certification_start": str(fold_plan.certification_start),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporal_indices = fold_plan.temporal_certification_indices
    transfer_indices = fold_plan.asset_transfer_certification_indices
    temporal_variance, temporal_return = ensemble.matched_baselines(examples, temporal_indices)
    transfer_variance, transfer_return = ensemble.matched_baselines(examples, transfer_indices)
    certification = certify_locked_predictions(
        examples=examples,
        fold_plan=fold_plan,
        temporal=LockedPopulationInput(
            population="temporal",
            indices=temporal_indices,
            predictions=ensemble.predict(examples, temporal_indices),
            baseline_variance=temporal_variance,
            baseline_return_variance=temporal_return,
        ),
        asset_transfer=LockedPopulationInput(
            population="asset_transfer",
            indices=transfer_indices,
            predictions=ensemble.predict(examples, transfer_indices),
            baseline_variance=transfer_variance,
            baseline_return_variance=transfer_return,
        ),
        model_identity=ensemble.model_identity,
        development_evidence_sha256=development_digest,
        eligible_horizons=eligible,
    )
    certification_path = output / "locked-certification.json"
    certification_path.write_text(
        json.dumps(certification.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if certification.status == "passed":
        _save_passed_candidate(output, ensemble, report)
    print(
        json.dumps(
            {
                "status": certification.status,
                "eligible_horizons": eligible,
                "certified_horizons": certification.certified_horizons,
                "model_identity": ensemble.model_identity,
                "report": str(certification_path),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0 if certification.status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
