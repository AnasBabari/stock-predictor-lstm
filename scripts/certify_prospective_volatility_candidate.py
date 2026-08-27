#!/usr/bin/env python3
"""Open the preregistered v7 future reserve exactly once.

This command never trains, tunes, calibrates, or selects a model.  It verifies
the frozen prospective candidate, proves that a later immutable panel preserves
the complete development prefix, selects the first target-complete future
reserve, and materializes a release-role candidate only when every locked gate
passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for candidate_path in (ROOT, ROOT / "research"):
    if str(candidate_path) not in sys.path:
        sys.path.insert(0, str(candidate_path))

from volatility_forecasting.cache import (  # noqa: E402
    ExampleCacheError,
    example_cache_key,
    find_compatible_example_cache,
    load_example_cache,
    panel_fingerprint,
    save_example_cache,
)
from volatility_forecasting.certification import (  # noqa: E402
    LockedPopulationInput,
    certify_locked_predictions,
)
from volatility_forecasting.contracts import VolatilityForecastProtocol  # noqa: E402
from volatility_forecasting.data import (  # noqa: E402
    VolatilityPanelExamples,
    build_volatility_panel_examples,
)
from volatility_forecasting.export import load_prospective_candidate_member  # noqa: E402
from volatility_forecasting.folds import (  # noqa: E402
    build_prospective_certification_fold_plan,
    build_prospective_development_fold_plan,
)
from volatility_forecasting.prospective import (  # noqa: E402
    OBJECTIVE_PROFILES,
    ProspectiveCycleSettings,
    objective_manifest,
    prospective_protocol,
    validate_prospective_panel_manifest,
)
from volatility_forecasting.refit import FrozenEnsemble, ensemble_identity  # noqa: E402
from volatility_forecasting.resume import expected_oof_identity  # noqa: E402

from backend.panel.snapshots import (  # noqa: E402
    canonical_csv,
    load_panel_from_directory,
)

CERTIFICATION_RESAMPLES = 2000


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _json_normalised(value: object) -> object:
    return json.loads(json.dumps(value))


def _read_json(path: Path, label: str) -> tuple[dict[str, object], bytes]:
    try:
        payload = path.read_bytes()
        parsed = json.loads(payload)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is missing or invalid") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return parsed, payload


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_candidate_contract(
    candidate_dir: Path,
    development_report_path: Path,
    development_panel_dir: Path,
) -> tuple[dict[str, object], dict[str, object], bytes]:
    """Fail closed unless all frozen candidate identities agree."""
    cycle = ProspectiveCycleSettings()
    protocol = prospective_protocol()
    manifest, _manifest_bytes = _read_json(
        candidate_dir / "candidate-manifest.json",
        "prospective candidate manifest",
    )
    report, report_bytes = _read_json(
        development_report_path,
        "prospective development report",
    )
    strict_policy = manifest.get("strict_release_policy")
    if manifest.get("artifact_role") != "prospective_development_candidate":
        raise ValueError("candidate is not a prospective development artifact")
    if manifest.get("release_eligible") is not False:
        raise ValueError("prospective candidate must remain non-release-eligible")
    if not isinstance(strict_policy, dict) or strict_policy != {
        "unsigned": True,
        "partial_release_allowed": False,
        "old_locked_holdout_reusable": False,
        "future_certification_required": True,
    }:
        raise ValueError("candidate strict release policy is incompatible")
    if report.get("mode") != "prospective_full_development":
        raise ValueError("development report is not the full prospective comparison")
    if report.get("freeze_eligible") is not True:
        raise ValueError("development report is not freeze-eligible")
    if _json_normalised(report.get("protocol")) != _json_normalised(asdict(protocol)):
        raise ValueError("development report protocol does not match v7")
    if _json_normalised(manifest.get("protocol")) != _json_normalised(report.get("protocol")):
        raise ValueError("candidate and development report protocols differ")
    if _json_normalised(manifest.get("architecture")) != _json_normalised(
        report.get("architecture")
    ):
        raise ValueError("candidate and development report architectures differ")
    report_digest = _sha256_bytes(report_bytes)
    if manifest.get("development_report_sha256") != report_digest:
        raise ValueError("candidate was frozen from a different development report")
    if (
        report.get("development_cutoff") != cycle.development_cutoff
        or manifest.get("development_cutoff") != cycle.development_cutoff
    ):
        raise ValueError("development cutoff does not match the preregistration")
    if report.get("prospective_certification_start") != cycle.prospective_certification_start or (
        manifest.get("prospective_certification_start") != cycle.prospective_certification_start
    ):
        raise ValueError("certification start does not match the preregistration")
    selection = report.get("selection")
    if not isinstance(selection, dict) or selection.get("status") != "selected":
        raise ValueError("development report did not select a candidate")
    selected_profile = selection.get("selected_profile")
    if selected_profile not in OBJECTIVE_PROFILES:
        raise ValueError("selected objective profile is outside v7")
    if manifest.get("selected_profile") != selected_profile:
        raise ValueError("candidate selected profile does not match the report")
    if _json_normalised(manifest.get("selection")) != _json_normalised(selection):
        raise ValueError("candidate selection evidence does not match the report")
    if _json_normalised(manifest.get("objective")) != _json_normalised(
        objective_manifest(OBJECTIVE_PROFILES[str(selected_profile)])
    ):
        raise ValueError("candidate objective does not match the frozen profile")
    validate_prospective_panel_manifest(
        development_panel_dir,
        expected_cutoff=cycle.development_cutoff,
    )
    development_checksum = panel_fingerprint(development_panel_dir)
    if manifest.get("panel_checksum") != development_checksum:
        raise ValueError("candidate development panel checksum does not match")
    model_identity = manifest.get("model_identity")
    if not isinstance(model_identity, str) or not model_identity.strip():
        raise ValueError("candidate model identity is missing")
    return manifest, report, report_bytes


def validate_panel_extension(
    development_panel: Mapping[str, pd.DataFrame],
    certification_panel: Mapping[str, pd.DataFrame],
    *,
    development_cutoff: str,
    certification_start: str,
) -> None:
    """Prove the enlarged panel preserves every immutable development row."""
    if set(development_panel) != set(certification_panel):
        raise ValueError("certification panel ticker universe differs from development")
    cutoff = np.datetime64(development_cutoff, "D")
    start = np.datetime64(certification_start, "D")
    missing_future: list[str] = []
    for ticker in sorted(development_panel):
        development = development_panel[ticker]
        certification = certification_panel[ticker]
        development_prefix = development.loc[
            np.asarray(development.index, dtype="datetime64[D]") <= cutoff
        ]
        certification_prefix = certification.loc[
            np.asarray(certification.index, dtype="datetime64[D]") <= cutoff
        ]
        if canonical_csv(development_prefix) != canonical_csv(certification_prefix):
            raise ValueError(f"certification panel changed the immutable {ticker} prefix")
        future = certification.loc[np.asarray(certification.index, dtype="datetime64[D]") >= start]
        if future.empty:
            missing_future.append(ticker)
    if missing_future:
        raise ValueError(
            "certification panel has no post-boundary rows for: " + ", ".join(missing_future)
        )


def _load_examples(
    panel_dir: Path,
    cache_root: Path,
    protocol: VolatilityForecastProtocol,
) -> tuple[VolatilityPanelExamples, str, Path | None]:
    panel_checksum = panel_fingerprint(panel_dir)
    compatible = find_compatible_example_cache(
        cache_root,
        panel_checksum=panel_checksum,
        protocol=protocol,
    )
    if compatible is not None:
        try:
            return (
                load_example_cache(
                    compatible,
                    panel_checksum=panel_checksum,
                    protocol=protocol,
                ),
                panel_checksum,
                compatible,
            )
        except ExampleCacheError as error:
            print(f"Ignoring invalid certification example cache: {error}", flush=True)
    panel = load_panel_from_directory(panel_dir)
    examples = build_volatility_panel_examples(panel, protocol)
    cache_dir = cache_root / example_cache_key(panel_checksum, protocol)
    save_example_cache(
        cache_dir,
        examples,
        panel_checksum=panel_checksum,
        protocol=protocol,
    )
    return examples, panel_checksum, cache_dir


def _load_verified_ensemble(
    candidate_dir: Path,
    manifest: dict[str, object],
) -> FrozenEnsemble:
    members_payload = manifest.get("members")
    if not isinstance(members_payload, list):
        raise ValueError("candidate member table is missing")
    protocol = prospective_protocol()
    seeds = tuple(
        sorted(
            int(row["seed"])
            for row in members_payload
            if isinstance(row, dict) and isinstance(row.get("seed"), int)
        )
    )
    if seeds != protocol.seeds or len(seeds) != len(members_payload):
        raise ValueError(f"candidate must contain exactly the frozen seeds {protocol.seeds}")
    members = tuple(load_prospective_candidate_member(candidate_dir, seed) for seed in seeds)
    identity = ensemble_identity(members)
    if identity != manifest.get("model_identity"):
        raise ValueError("prospective ensemble identity does not match its members")
    return FrozenEnsemble(members=members, model_identity=identity)


def _decision_summaries(report: dict[str, object]) -> dict[str, dict[str, object]]:
    summaries: dict[str, dict[str, object]] = {}
    decisions = report.get("decisions")
    if not isinstance(decisions, list):
        return summaries
    for row in decisions:
        if not isinstance(row, dict) or not isinstance(row.get("horizon"), int):
            continue
        population = str(row.get("population", "unknown"))
        horizon = str(row["horizon"])
        summaries.setdefault(horizon, {})[population] = {
            "decision": row.get("decision"),
            "relative_qlike": row.get("relative_qlike"),
            "ratio_upper_95": row.get("ratio_upper_95"),
            "dm_p_value": row.get("dm_p_value"),
            "holm_significant": row.get("holm_significant"),
            "required_ticker_relative_qlike": row.get("required_ticker_relative_qlike"),
            "metrics": row.get("metrics"),
            "reasons": row.get("reasons"),
        }
    return summaries


def materialize_passed_candidate(
    source_dir: Path,
    output_dir: Path,
    source_manifest: dict[str, object],
    certification_report: dict[str, object],
    marker_path: Path,
    report_path: Path,
) -> Path:
    """Copy the exact certified bytes behind a release-only artifact role."""
    if certification_report.get("status") != "passed":
        raise ValueError("only an overall passed certification may be materialized")
    certified_horizons = certification_report.get("certified_horizons")
    eligible_horizons = certification_report.get("eligible_horizons")
    if certified_horizons != eligible_horizons or not isinstance(certified_horizons, list):
        raise ValueError("partial prospective certification cannot be materialized")
    candidate_output = output_dir / "candidate"
    if candidate_output.exists():
        raise FileExistsError("materialized prospective candidate already exists")
    temporary_output = output_dir / ".candidate.tmp"
    if temporary_output.exists():
        shutil.rmtree(temporary_output)
    temporary_output.mkdir(parents=True, exist_ok=False)
    members = source_manifest.get("members")
    if not isinstance(members, list):
        raise ValueError("candidate member table is missing")
    for row in members:
        if not isinstance(row, dict):
            raise ValueError("candidate member row is malformed")
        filename = row.get("weights_file")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("candidate weights path is not allowed")
        source = source_dir / filename
        if _sha256_file(source) != row.get("weights_sha256"):
            raise ValueError("candidate weight checksum changed before materialization")
        shutil.copyfile(source, temporary_output / filename)

    locked_manifest = dict(source_manifest)
    locked_manifest["artifact_role"] = "locked_certification_candidate"
    locked_manifest["release_eligible"] = True
    locked_manifest["source_prospective_candidate_sha256"] = _sha256_file(
        source_dir / "candidate-manifest.json"
    )
    locked_manifest["strict_release_policy"] = {
        "unsigned": True,
        "partial_release_allowed": False,
        "old_locked_holdout_reusable": False,
        "future_certification_required": False,
        "locked_certification_passed": True,
    }
    locked_manifest["locked_certification"] = {
        "status": "passed",
        "certified_horizons": certified_horizons,
        "abstained_eligible_horizons": [],
        "certification_start": certification_report.get("certification_start"),
        "development_panel_checksum": certification_report.get("development_panel_checksum"),
        "certification_panel_checksum": certification_report.get("certification_panel_checksum"),
        "locked_origin_start": certification_report.get("locked_origin_start"),
        "locked_origin_end": certification_report.get("locked_origin_end"),
        "locked_origin_sessions": certification_report.get("locked_origin_sessions"),
        "horizon_decisions": _decision_summaries(certification_report),
        "evidence_sha256": {
            "holdout-opened.json": _sha256_file(marker_path),
            "locked-certification.json": _sha256_file(report_path),
        },
    }
    _write_json_atomic(temporary_output / "candidate-manifest.json", locked_manifest)
    temporary_output.replace(candidate_output)
    return candidate_output


def _validate_development_identity(
    examples: VolatilityPanelExamples,
    manifest: dict[str, object],
) -> None:
    cycle = ProspectiveCycleSettings()
    protocol = prospective_protocol()
    plan = build_prospective_development_fold_plan(
        examples,
        protocol,
        development_cutoff=np.datetime64(cycle.development_cutoff, "D"),
        prospective_certification_start=np.datetime64(
            cycle.prospective_certification_start,
            "D",
        ),
    )
    identity = expected_oof_identity(examples.tickers, examples.origin_dates, plan)
    if identity["sha256"] != manifest.get("oof_identity_sha256"):
        raise ValueError("candidate OOF identity does not match the development panel")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Open the frozen v7 future volatility reserve exactly once",
    )
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--development-report", type=Path, required=True)
    parser.add_argument("--development-panel-dir", type=Path, required=True)
    parser.add_argument("--certification-panel-dir", type=Path, required=True)
    parser.add_argument("--example-cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--open-locked-holdout",
        action="store_true",
        help="Required acknowledgement: this irreversibly consumes the v7 reserve",
    )
    args = parser.parse_args()
    if not args.open_locked_holdout:
        parser.error("--open-locked-holdout is required")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error("--output-dir must not exist or must be empty")

    candidate_dir = args.candidate_dir.resolve()
    report_path = args.development_report.resolve()
    development_panel_dir = args.development_panel_dir.resolve()
    certification_panel_dir = args.certification_panel_dir.resolve()
    cache_root = args.example_cache_root.resolve()
    cycle = ProspectiveCycleSettings()
    protocol = prospective_protocol()
    manifest, _report, report_bytes = validate_candidate_contract(
        candidate_dir,
        report_path,
        development_panel_dir,
    )

    development_manifest, _ = _read_json(
        development_panel_dir / "manifest.json",
        "development panel manifest",
    )
    certification_manifest, _ = _read_json(
        certification_panel_dir / "manifest.json",
        "certification panel manifest",
    )
    for field in ("provider", "timezone", "adjust_mode"):
        expected_value = development_manifest.get(field)
        if (
            not isinstance(expected_value, str)
            or not expected_value.strip()
            or certification_manifest.get(field) != expected_value
        ):
            raise ValueError(f"certification panel {field} differs from development")
    license_row = certification_manifest.get("license")
    if not isinstance(license_row, dict) or license_row.get("acknowledged") is not True:
        raise ValueError("certification panel provider license is not acknowledged")

    print("Verifying immutable development prefix in the certification panel...", flush=True)
    development_panel = load_panel_from_directory(development_panel_dir)
    certification_panel = load_panel_from_directory(certification_panel_dir)
    validate_panel_extension(
        development_panel,
        certification_panel,
        development_cutoff=cycle.development_cutoff,
        certification_start=cycle.prospective_certification_start,
    )
    development_examples, _development_checksum, _development_cache = _load_examples(
        development_panel_dir,
        cache_root,
        protocol,
    )
    _validate_development_identity(development_examples, manifest)

    certification_examples, certification_checksum, certification_cache = _load_examples(
        certification_panel_dir,
        cache_root,
        protocol,
    )
    fold_plan = build_prospective_certification_fold_plan(
        certification_examples,
        protocol,
        development_cutoff=np.datetime64(cycle.development_cutoff, "D"),
        prospective_certification_start=np.datetime64(
            cycle.prospective_certification_start,
            "D",
        ),
    )
    ensemble = _load_verified_ensemble(candidate_dir, manifest)

    output_dir.mkdir(parents=True, exist_ok=True)
    marker_path = output_dir / "holdout-opened.json"
    locked_dates = certification_examples.origin_dates[
        np.concatenate(
            (
                fold_plan.temporal_certification_indices,
                fold_plan.asset_transfer_certification_indices,
            )
        )
    ]
    marker = {
        "model_identity": ensemble.model_identity,
        "candidate_manifest_sha256": _sha256_file(candidate_dir / "candidate-manifest.json"),
        "development_evidence_sha256": _sha256_bytes(report_bytes),
        "development_panel_checksum": panel_fingerprint(development_panel_dir),
        "certification_panel_checksum": certification_checksum,
        "certification_example_cache": (
            str(certification_cache) if certification_cache is not None else None
        ),
        "certification_start": cycle.prospective_certification_start,
        "locked_origin_start": str(locked_dates.min()),
        "locked_origin_end": str(locked_dates.max()),
        "locked_origin_sessions": protocol.temporal_holdout_sessions,
        "eligible_horizons": list(cycle.required_horizons),
        "one_shot": True,
    }
    _write_json_atomic(marker_path, marker)

    temporal_indices = fold_plan.temporal_certification_indices
    transfer_indices = fold_plan.asset_transfer_certification_indices
    temporal_baseline, temporal_return_baseline = ensemble.matched_baselines(
        certification_examples,
        temporal_indices,
    )
    transfer_baseline, transfer_return_baseline = ensemble.matched_baselines(
        certification_examples,
        transfer_indices,
    )
    certification = certify_locked_predictions(
        examples=certification_examples,
        fold_plan=fold_plan,
        temporal=LockedPopulationInput(
            population="temporal",
            indices=temporal_indices,
            predictions=ensemble.predict(certification_examples, temporal_indices),
            baseline_variance=temporal_baseline,
            baseline_return_variance=temporal_return_baseline,
        ),
        asset_transfer=LockedPopulationInput(
            population="asset_transfer",
            indices=transfer_indices,
            predictions=ensemble.predict(certification_examples, transfer_indices),
            baseline_variance=transfer_baseline,
            baseline_return_variance=transfer_return_baseline,
        ),
        model_identity=ensemble.model_identity,
        development_evidence_sha256=_sha256_bytes(report_bytes),
        eligible_horizons=cycle.required_horizons,
        resamples=CERTIFICATION_RESAMPLES,
    )
    report_payload = _json_normalised(certification.to_dict())
    if not isinstance(report_payload, dict):  # pragma: no cover - dataclass contract
        raise RuntimeError("certification report did not serialize as an object")
    report_payload.update(
        {
            "protocol": asdict(protocol),
            "candidate_manifest_sha256": marker["candidate_manifest_sha256"],
            "development_panel_checksum": marker["development_panel_checksum"],
            "certification_panel_checksum": marker["certification_panel_checksum"],
            "locked_origin_start": marker["locked_origin_start"],
            "locked_origin_end": marker["locked_origin_end"],
            "locked_origin_sessions": marker["locked_origin_sessions"],
        }
    )
    report_path_out = output_dir / "locked-certification.json"
    _write_json_atomic(report_path_out, report_payload)
    materialized = None
    if certification.status == "passed":
        materialized = materialize_passed_candidate(
            candidate_dir,
            output_dir,
            manifest,
            report_payload,
            marker_path,
            report_path_out,
        )
    print(
        json.dumps(
            {
                "status": certification.status,
                "model_identity": certification.model_identity,
                "eligible_horizons": list(certification.eligible_horizons),
                "certified_horizons": list(certification.certified_horizons),
                "report": str(report_path_out),
                "materialized_candidate": str(materialized) if materialized else None,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if certification.status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
