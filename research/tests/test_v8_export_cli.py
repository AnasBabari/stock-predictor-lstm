from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from export_v8_onnx import _load_verified_certification  # noqa: E402


def _write_certification(directory: Path, *, model_identity: str = "ensemble-v8") -> dict:
    report = {
        "status": "passed",
        "release_eligible": True,
        "model_identity": model_identity,
        "metric_source": "locked_historical_temporal_test_plus_asset_transfer",
    }
    path = directory / "v8-locked-certification.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return {
        "model_identity": model_identity,
        "certification_report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_export_requires_matching_passing_certification(tmp_path) -> None:
    manifest = _write_certification(tmp_path)
    report, digest = _load_verified_certification(tmp_path, manifest)
    assert report["status"] == "passed"
    assert digest == manifest["certification_report_sha256"]

    report_path = tmp_path / "v8-locked-certification.json"
    report_path.write_text(report_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        _load_verified_certification(tmp_path, manifest)


def test_export_rejects_report_for_another_ensemble(tmp_path) -> None:
    manifest = _write_certification(tmp_path, model_identity="ensemble-a")
    manifest["model_identity"] = "ensemble-b"
    with pytest.raises(ValueError, match="authorize"):
        _load_verified_certification(tmp_path, manifest)
