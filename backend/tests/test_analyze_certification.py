"""Unit tests for scripts/analyze_certification.py descriptive analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.analyze_certification import analyze_certification_artifact


def test_analyze_certification_artifact_parses_cleanly(tmp_path: Path) -> None:
    cert_file = tmp_path / "07_certification.json"
    out_file = tmp_path / "08_post_certification_analysis.json"

    dummy_data = {
        "certification_protocol_version": "global-cert-v2",
        "status": "holdout_opened",
        "decision": "pass",
        "certified_horizons": [5],
        "gate_config": {
            "require_temporal_relative_rmse": True,
            "max_temporal_relative_rmse": 1.0,
        },
        "decisions": {
            "5": {
                "candidate_name": "rolling_mean_shrunk",
                "decision": "pass",
                "temporal_relative_rmse": 0.999997,
                "temporal_relative_mae": 0.999996,
                "transfer_relative_rmse": 0.999998,
                "transfer_relative_mae": 0.999994,
                "temporal_direction_acc": 0.5304,
                "positive_prevalence": 0.5304,
                "majority_class_accuracy": 0.5304,
                "direction_accuracy_delta_vs_majority": 0.0,
                "balanced_accuracy": 0.5,
                "temporal_brier": None,
                "direction_probability_status": "not_available",
                "passed_gates": ["temporal_relative_rmse(1.0000 <= 1.0000)"],
                "failed_gates": [],
            }
        },
    }
    cert_file.write_text(json.dumps(dummy_data, indent=2), encoding="utf-8")

    result = analyze_certification_artifact(cert_file, out_file)
    assert result["analysis_type"] == "post_certification"
    assert result["affects_original_certification"] is False
    assert result["precommitted_gate"] is False
    assert "5" in result["horizon_breakdown"]
    assert out_file.exists()


def test_analyze_certification_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        analyze_certification_artifact(tmp_path / "non_existent.json")
