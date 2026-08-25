#!/usr/bin/env python3
"""Persist the champion weights behind an already-consumed locked certification.

The one-shot reserve can never be reopened or re-evaluated. This command only
re-materialises the exact frozen ensemble that the certification run fitted on
development-eligible rows, verifies its content identity against the recorded
certification evidence, and stores it alongside the certified-horizon subset.
Any identity mismatch fails closed: serving weights must equal evaluated weights.
"""

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
)
from volatility_forecasting.contracts import VolatilityForecastProtocol  # noqa: E402
from volatility_forecasting.data import build_volatility_panel_examples  # noqa: E402
from volatility_forecasting.folds import build_volatility_fold_plan  # noqa: E402
from volatility_forecasting.refit import (  # noqa: E402
    fit_frozen_ensemble,
)

from backend.panel.snapshots import load_panel_from_directory  # noqa: E402
from scripts.certify_volatility_candidate import (  # noqa: E402
    validate_development_report,
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Materialise the certified champion from consumed locked evidence",
    )
    parser.add_argument("--development-report", type=Path, required=True)
    parser.add_argument("--panel-dir", type=Path, required=True)
    parser.add_argument("--certification-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--example-cache-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("--output-dir must not exist or must be empty")

    protocol = VolatilityForecastProtocol()
    report_path = args.development_report.resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    records, architecture, eligible = validate_development_report(report, protocol)

    certification_dir = args.certification_dir.resolve()
    opened_path = certification_dir / "holdout-opened.json"
    evidence_path = certification_dir / "locked-certification.json"
    if not opened_path.exists() or not evidence_path.exists():
        raise ValueError("consumed certification evidence is missing")
    opened = json.loads(opened_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))

    recorded_digest = evidence.get("development_evidence_sha256")
    if recorded_digest != _sha256_file(report_path):
        raise ValueError("certification evidence was produced from a different development report")
    recorded_identity = str(evidence.get("model_identity", ""))
    if not recorded_identity or recorded_identity != str(opened.get("model_identity", "")):
        raise ValueError("certification evidence identities disagree")
    certified_horizons = evidence.get("certified_horizons")
    if (
        not isinstance(certified_horizons, list)
        or not certified_horizons
        or any(h not in eligible for h in certified_horizons)
    ):
        raise ValueError("certified horizons are absent or outside the eligible set")

    def _decision_summary(row: dict) -> dict | None:
        if not isinstance(row, dict) or not isinstance(row.get("metrics"), dict):
            return None
        return {
            "decision": row.get("decision"),
            "relative_qlike": row.get("relative_qlike"),
            "ratio_upper_95": row.get("ratio_upper_95"),
            "dm_p_value": row.get("dm_p_value"),
            "holm_significant": row.get("holm_significant"),
            "coverage_80": row["metrics"].get("coverage_80"),
            "coverage_95": row["metrics"].get("coverage_95"),
            "variance_only_coverage_80": row["metrics"].get("variance_only_coverage_80"),
            "gaussian_crps": row["metrics"].get("gaussian_crps"),
            "required_ticker_relative_qlike": row.get("required_ticker_relative_qlike"),
            "reasons": list(row.get("reasons") or []),
        }

    decision_summaries: dict[int, dict] = {}
    for row in evidence.get("decisions") or []:
        if isinstance(row, dict) and isinstance(row.get("horizon"), int):
            summary = _decision_summary(row)
            if summary is not None:
                population = str(row.get("population", "unknown"))
                decision_summaries.setdefault(int(row["horizon"]), {})[population] = summary

    panel_checksum = _panel_checksum(args)
    cache = find_compatible_example_cache(
        args.example_cache_root.resolve(),
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
        f"Re-fitting frozen {architecture.encoder_family} ensemble "
        f"for seeds {protocol.seeds} on development-eligible rows...",
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
    if ensemble.model_identity != recorded_identity:
        raise ValueError(
            "re-fitted weights do not reproduce the evaluated champion "
            f"(expected {recorded_identity}, computed {ensemble.model_identity})"
        )

    output.mkdir(parents=True, exist_ok=False)
    members = []
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
            }
        )
    abstained = [h for h in eligible if h not in certified_horizons]
    manifest = {
        "artifact_role": "locked_certification_candidate",
        "model_identity": ensemble.model_identity,
        "protocol": report["protocol"],
        "architecture": report["architecture"],
        "members": members,
        "locked_certification": {
            "status": evidence.get("status"),
            "certified_horizons": certified_horizons,
            "abstained_eligible_horizons": abstained,
            "certification_start": evidence.get("certification_start"),
            "horizon_decisions": {
                str(horizon): summaries for horizon, summaries in sorted(decision_summaries.items())
            },
            "evidence_sha256": {
                "locked-certification.json": _sha256_file(evidence_path),
                "holdout-opened.json": _sha256_file(opened_path),
                "development-report.json": _sha256_file(report_path),
            },
        },
    }
    (output / "candidate-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "materialised",
                "model_identity": ensemble.model_identity,
                "certified_horizons": certified_horizons,
                "abstained_eligible_horizons": abstained,
                "output": str(output),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def _panel_checksum(args: argparse.Namespace):
    from volatility_forecasting.cache import panel_fingerprint

    return panel_fingerprint(args.panel_dir)


if __name__ == "__main__":
    raise SystemExit(main())
