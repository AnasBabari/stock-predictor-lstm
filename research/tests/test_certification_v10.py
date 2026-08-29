"""Tests for V10 single-use sealed test certification."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.volatility_forecasting.certification_v10 import (
    CertificationReportV10,
    HorizonCertificationDecision,
    SealedTestOpeningRecordV10,
    SealedTestReopenError,
    verify_certification_prerequisites,
)
from research.volatility_forecasting.market_snapshot_v10 import DataIneligibilityError


def test_certification_fails_closed_on_ineligible_data(tmp_path: Path) -> None:
    protocol = {
        "protocol_status": "frozen",
        "data_eligibility": {
            "universe_certification_eligible": False,
            "market_panel_certification_eligible": False,
            "blocker": "Ineligible data source.",
        },
    }
    with pytest.raises(DataIneligibilityError, match="Ineligible data source"):
        verify_certification_prerequisites(protocol, {}, tmp_path)


def test_test_opening_record_refuses_duplicate_opening(tmp_path: Path) -> None:
    rec1 = SealedTestOpeningRecordV10(
        test_opened_at_utc="2026-08-29T21:00:00Z",
        candidate_package_sha256="0" * 64,
        protocol_sha256="1" * 64,
        split_manifest_sha256="2" * 64,
        operator="ci_runner",
        attempt=1,
    )
    rec1.save(tmp_path)

    # Second attempt to save must fail with SealedTestReopenError
    with pytest.raises(SealedTestReopenError, match="already exists"):
        rec1.save(tmp_path)


def test_certification_report_serialization(tmp_path: Path) -> None:
    d1 = HorizonCertificationDecision(
        horizon=1,
        family="tcn",
        outcome="certified_learned_model",
        relative_qlike=0.90,
        ratio_upper_95=0.98,
        dm_p_value=0.01,
        holm_adjusted_p_value=0.03,
        transfer_relative_qlike=0.95,
        passed_all_gates=True,
        reason="Cleared all certification gates",
    )
    d3 = HorizonCertificationDecision(
        horizon=3,
        family="har",
        outcome="certified_baseline",
        relative_qlike=1.00,
        ratio_upper_95=1.00,
        dm_p_value=0.50,
        holm_adjusted_p_value=1.00,
        transfer_relative_qlike=1.00,
        passed_all_gates=True,
        reason="Certified baseline fallback",
    )
    report = CertificationReportV10(
        report_id="cert-v10-001",
        protocol_id="volatility-v10",
        protocol_sha256="0" * 64,
        candidate_package_sha256="1" * 64,
        data_snapshot_sha256="2" * 64,
        evaluated_at_utc="2026-08-29T21:00:00Z",
        decisions=(d1, d3),
        certified_horizons=(1, 3),
    )
    report_file = report.save(tmp_path)
    assert report_file.exists()
    reloaded = json.loads(report_file.read_text(encoding="utf-8"))
    assert reloaded["report_id"] == "cert-v10-001"
    assert reloaded["certified_horizons"] == [1, 3]
