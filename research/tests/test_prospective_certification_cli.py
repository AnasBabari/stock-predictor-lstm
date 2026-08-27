from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from volatility_forecasting.prospective import (
    OBJECTIVE_PROFILES,
    ProspectiveCycleSettings,
    objective_manifest,
    prospective_protocol,
)

from backend.panel.snapshots import write_snapshot

SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "certify_prospective_volatility_candidate.py"
)
SPEC = importlib.util.spec_from_file_location("certify_prospective_volatility_candidate", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

MATERIALIZER_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "materialize_prospective_certification.py"
)
MATERIALIZER_SPEC = importlib.util.spec_from_file_location(
    "materialize_prospective_certification",
    MATERIALIZER_SCRIPT,
)
assert MATERIALIZER_SPEC and MATERIALIZER_SPEC.loader
MATERIALIZER = importlib.util.module_from_spec(MATERIALIZER_SPEC)
MATERIALIZER_SPEC.loader.exec_module(MATERIALIZER)


def _frame(dates: pd.DatetimeIndex, *, offset: float = 0.0) -> pd.DataFrame:
    close = np.arange(len(dates), dtype=float) + 100.0 + offset
    return pd.DataFrame(
        {
            "Open": close - 0.2,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Close": close,
            "Volume": np.arange(len(dates), dtype=float) + 1_000.0,
        },
        index=dates,
    )


def _candidate_contract(tmp_path: Path) -> tuple[Path, Path, Path]:
    cycle = ProspectiveCycleSettings()
    dates = pd.bdate_range(end=cycle.development_cutoff, periods=80)
    panel_dir = write_snapshot(
        tmp_path / "panels",
        {"MSFT": _frame(dates)},
        license_acknowledged=True,
    )
    protocol = prospective_protocol()
    architecture = {
        "feature_count": protocol.feature_count,
        "horizon_count": len(protocol.horizons),
        "window_size": protocol.window_size,
    }
    selection = {"status": "selected", "selected_profile": "multitask_v1"}
    report = {
        "mode": "prospective_full_development",
        "freeze_eligible": True,
        "protocol": asdict(protocol),
        "architecture": architecture,
        "development_cutoff": cycle.development_cutoff,
        "prospective_certification_start": cycle.prospective_certification_start,
        "selection": selection,
    }
    report_path = tmp_path / "prospective-development-report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    panel_manifest = json.loads((panel_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest = {
        "artifact_role": "prospective_development_candidate",
        "release_eligible": False,
        "model_identity": "global-volatility-ensemble:fixture",
        "development_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "selected_profile": "multitask_v1",
        "objective": objective_manifest(OBJECTIVE_PROFILES["multitask_v1"]),
        "protocol": asdict(protocol),
        "architecture": architecture,
        "development_cutoff": cycle.development_cutoff,
        "prospective_certification_start": cycle.prospective_certification_start,
        "panel_checksum": panel_manifest["pooled_checksum"],
        "selection": selection,
        "strict_release_policy": {
            "unsigned": True,
            "partial_release_allowed": False,
            "old_locked_holdout_reusable": False,
            "future_certification_required": True,
        },
    }
    (candidate_dir / "candidate-manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return candidate_dir, report_path, panel_dir


def test_candidate_contract_binds_report_panel_protocol_and_objective(tmp_path: Path) -> None:
    candidate_dir, report_path, panel_dir = _candidate_contract(tmp_path)
    manifest, report, report_bytes = MODULE.validate_candidate_contract(
        candidate_dir,
        report_path,
        panel_dir,
    )
    assert manifest["model_identity"] == "global-volatility-ensemble:fixture"
    assert report["freeze_eligible"] is True
    assert hashlib.sha256(report_bytes).hexdigest() == manifest["development_report_sha256"]

    changed = json.loads(report_path.read_text(encoding="utf-8"))
    changed["post_hoc_change"] = True
    report_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="different development report"):
        MODULE.validate_candidate_contract(candidate_dir, report_path, panel_dir)


def test_panel_extension_requires_exact_prefix_and_future_rows() -> None:
    development_dates = pd.bdate_range(end="2026-08-21", periods=8)
    future_dates = pd.bdate_range(start="2026-08-27", periods=3)
    development = {
        "MSFT": _frame(development_dates),
        "NMM": _frame(development_dates, offset=20.0),
    }
    certification = {
        ticker: pd.concat(
            (
                frame,
                _frame(future_dates, offset=float(frame["Close"].iloc[-1]) - 99.0),
            )
        )
        for ticker, frame in development.items()
    }
    MODULE.validate_panel_extension(
        development,
        certification,
        development_cutoff="2026-08-21",
        certification_start="2026-08-27",
    )

    changed = {ticker: frame.copy() for ticker, frame in certification.items()}
    changed["MSFT"].loc[development_dates[-1], "Close"] += 1.0
    with pytest.raises(ValueError, match="immutable MSFT prefix"):
        MODULE.validate_panel_extension(
            development,
            changed,
            development_cutoff="2026-08-21",
            certification_start="2026-08-27",
        )

    missing = {ticker: frame.copy() for ticker, frame in certification.items()}
    missing["NMM"] = missing["NMM"].loc[:"2026-08-21"]
    with pytest.raises(ValueError, match="no post-boundary rows for: NMM"):
        MODULE.validate_panel_extension(
            development,
            missing,
            development_cutoff="2026-08-21",
            certification_start="2026-08-27",
        )


def test_materialization_requires_whole_pass_and_preserves_weight_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    weights = source / "seed-41.pt"
    weights.write_bytes(b"verified-weights")
    source_manifest = {
        "artifact_role": "prospective_development_candidate",
        "release_eligible": False,
        "members": [
            {
                "seed": 41,
                "weights_file": weights.name,
                "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
            }
        ],
    }
    (source / "candidate-manifest.json").write_text(
        json.dumps(source_manifest),
        encoding="utf-8",
    )
    output = tmp_path / "certification"
    output.mkdir()
    marker = output / "holdout-opened.json"
    report = output / "locked-certification.json"
    marker.write_text("{}", encoding="utf-8")
    report.write_text("{}", encoding="utf-8")
    passed = {
        "status": "passed",
        "certification_start": "2026-08-27",
        "eligible_horizons": [1, 3, 5, 7],
        "certified_horizons": [1, 3, 5, 7],
        "decisions": [],
    }
    candidate_output = MODULE.materialize_passed_candidate(
        source,
        output,
        source_manifest,
        passed,
        marker,
        report,
    )
    assert (candidate_output / weights.name).read_bytes() == weights.read_bytes()
    materialized = json.loads(
        (candidate_output / "candidate-manifest.json").read_text(encoding="utf-8")
    )
    assert materialized["artifact_role"] == "locked_certification_candidate"
    assert materialized["release_eligible"] is True
    assert materialized["locked_certification"]["status"] == "passed"

    with pytest.raises(ValueError, match="Partial|partial"):
        MODULE.materialize_passed_candidate(
            source,
            tmp_path / "partial",
            source_manifest,
            {**passed, "certified_horizons": [1, 3, 5]},
            marker,
            report,
        )


def test_recovery_materializer_binds_passed_evidence_to_candidate(tmp_path: Path) -> None:
    candidate_dir, development_report, development_panel = _candidate_contract(tmp_path)
    candidate_manifest = json.loads(
        (candidate_dir / "candidate-manifest.json").read_text(encoding="utf-8")
    )
    report_digest = hashlib.sha256(development_report.read_bytes()).hexdigest()
    candidate_digest = hashlib.sha256(
        (candidate_dir / "candidate-manifest.json").read_bytes()
    ).hexdigest()
    cycle = ProspectiveCycleSettings()
    certification_dir = tmp_path / "locked"
    certification_dir.mkdir()
    marker = {
        "one_shot": True,
        "candidate_manifest_sha256": candidate_digest,
        "development_evidence_sha256": report_digest,
        "model_identity": candidate_manifest["model_identity"],
        "eligible_horizons": list(cycle.required_horizons),
        "development_panel_checksum": candidate_manifest["panel_checksum"],
        "certification_panel_checksum": "sha256:" + "a" * 64,
        "locked_origin_start": cycle.prospective_certification_start,
        "locked_origin_end": "2027-08-27",
        "locked_origin_sessions": prospective_protocol().temporal_holdout_sessions,
    }
    certification = {
        "status": "passed",
        "model_identity": candidate_manifest["model_identity"],
        "development_evidence_sha256": report_digest,
        "candidate_manifest_sha256": candidate_digest,
        "protocol": asdict(prospective_protocol()),
        "certification_start": cycle.prospective_certification_start,
        "eligible_horizons": list(cycle.required_horizons),
        "certified_horizons": list(cycle.required_horizons),
        "development_panel_checksum": candidate_manifest["panel_checksum"],
        "certification_panel_checksum": "sha256:" + "a" * 64,
        "locked_origin_start": cycle.prospective_certification_start,
        "locked_origin_end": "2027-08-27",
        "locked_origin_sessions": prospective_protocol().temporal_holdout_sessions,
    }
    (certification_dir / "holdout-opened.json").write_text(
        json.dumps(marker),
        encoding="utf-8",
    )
    report_path = certification_dir / "locked-certification.json"
    report_path.write_text(json.dumps(certification), encoding="utf-8")
    manifest, evidence, _marker_path, _report_path = MATERIALIZER.validate_passed_evidence(
        candidate_dir,
        development_report,
        development_panel,
        certification_dir,
    )
    assert manifest["model_identity"] == evidence["model_identity"]

    certification["status"] = "failed"
    report_path.write_text(json.dumps(certification), encoding="utf-8")
    with pytest.raises(ValueError, match="did not pass overall"):
        MATERIALIZER.validate_passed_evidence(
            candidate_dir,
            development_report,
            development_panel,
            certification_dir,
        )
