#!/usr/bin/env python3
"""Recoverably materialize an already-passed v7 prospective certification."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for candidate_path in (ROOT, ROOT / "research"):
    if str(candidate_path) not in sys.path:
        sys.path.insert(0, str(candidate_path))

from volatility_forecasting.prospective import (  # noqa: E402
    ProspectiveCycleSettings,
    prospective_protocol,
)

from scripts.certify_prospective_volatility_candidate import (  # noqa: E402
    materialize_passed_candidate,
    validate_candidate_contract,
)


def _read_object(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is missing or invalid") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def validate_passed_evidence(
    candidate_dir: Path,
    development_report_path: Path,
    development_panel_dir: Path,
    certification_dir: Path,
) -> tuple[dict[str, object], dict[str, object], Path, Path]:
    """Bind passed evidence to the exact frozen candidate and preregistration."""
    manifest, _development_report, report_bytes = validate_candidate_contract(
        candidate_dir,
        development_report_path,
        development_panel_dir,
    )
    marker_path = certification_dir / "holdout-opened.json"
    report_path = certification_dir / "locked-certification.json"
    marker = _read_object(marker_path, "one-shot holdout marker")
    certification = _read_object(report_path, "locked certification report")
    cycle = ProspectiveCycleSettings()
    candidate_manifest_path = candidate_dir / "candidate-manifest.json"
    candidate_digest = hashlib.sha256(candidate_manifest_path.read_bytes()).hexdigest()
    development_digest = hashlib.sha256(report_bytes).hexdigest()
    model_identity = manifest.get("model_identity")
    if marker.get("one_shot") is not True:
        raise ValueError("certification evidence is not marked one-shot")
    if marker.get("candidate_manifest_sha256") != candidate_digest:
        raise ValueError("holdout marker refers to a different prospective candidate")
    if marker.get("development_evidence_sha256") != development_digest:
        raise ValueError("holdout marker refers to different development evidence")
    if marker.get("model_identity") != model_identity:
        raise ValueError("holdout marker model identity differs from the candidate")
    if certification.get("model_identity") != model_identity:
        raise ValueError("certification model identity differs from the candidate")
    if certification.get("development_evidence_sha256") != development_digest:
        raise ValueError("certification report refers to different development evidence")
    if certification.get("candidate_manifest_sha256") != candidate_digest:
        raise ValueError("certification report refers to a different prospective candidate")
    if certification.get("protocol") != json.loads(json.dumps(asdict(prospective_protocol()))):
        raise ValueError("certification report protocol differs from v7")
    for checksum_field in ("development_panel_checksum", "certification_panel_checksum"):
        checksum = certification.get(checksum_field)
        if (
            checksum != marker.get(checksum_field)
            or not isinstance(checksum, str)
            or not checksum.startswith("sha256:")
            or len(checksum) != len("sha256:") + 64
        ):
            raise ValueError(f"certification {checksum_field} is invalid or inconsistent")
    if certification.get("development_panel_checksum") != manifest.get("panel_checksum"):
        raise ValueError("certification development panel differs from the candidate")
    if certification.get("certification_start") != cycle.prospective_certification_start:
        raise ValueError("certification report start differs from the preregistration")
    required_horizons = list(cycle.required_horizons)
    if marker.get("eligible_horizons") != required_horizons:
        raise ValueError("holdout marker eligible horizons differ from v7")
    if certification.get("eligible_horizons") != required_horizons:
        raise ValueError("certification eligible horizons differ from v7")
    if certification.get("status") != "passed":
        raise ValueError("prospective certification did not pass overall")
    if certification.get("certified_horizons") != required_horizons:
        raise ValueError("partial prospective certification cannot be materialized")
    if certification.get("locked_origin_start") != marker.get("locked_origin_start") or (
        certification.get("locked_origin_end") != marker.get("locked_origin_end")
    ):
        raise ValueError("certification locked origin range differs from the one-shot marker")
    if certification.get("locked_origin_sessions") != marker.get("locked_origin_sessions"):
        raise ValueError("certification locked session count differs from the one-shot marker")
    return manifest, certification, marker_path, report_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Materialize exact v7 weights after an immutable passed certification",
    )
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--development-report", type=Path, required=True)
    parser.add_argument("--development-panel-dir", type=Path, required=True)
    parser.add_argument("--certification-dir", type=Path, required=True)
    args = parser.parse_args()

    candidate_dir = args.candidate_dir.resolve()
    certification_dir = args.certification_dir.resolve()
    if not certification_dir.is_dir():
        parser.error("--certification-dir must be an existing one-shot output directory")
    if (certification_dir / "candidate").exists():
        parser.error("the certification candidate is already materialized")
    manifest, certification, marker_path, report_path = validate_passed_evidence(
        candidate_dir,
        args.development_report.resolve(),
        args.development_panel_dir.resolve(),
        certification_dir,
    )
    output = materialize_passed_candidate(
        candidate_dir,
        certification_dir,
        manifest,
        certification,
        marker_path,
        report_path,
    )
    print(
        json.dumps(
            {
                "status": "materialized",
                "model_identity": certification["model_identity"],
                "candidate": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
