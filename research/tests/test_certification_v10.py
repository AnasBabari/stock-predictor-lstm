"""Tests for V10 atomic single-use sealed test certification and real evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from research.volatility_forecasting.certification_v10 import (
    SealedTestOpeningRecordV10,
    SealedTestReopenError,
    evaluate_horizon_certification,
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
    rec1.save_atomic(tmp_path)

    # Second atomic attempt must fail with SealedTestReopenError
    with pytest.raises(SealedTestReopenError, match="already exists"):
        rec1.save_atomic(tmp_path)


def test_evaluate_horizon_certification_passes_superior_candidate() -> None:
    actuals = np.array([0.0004] * 50)
    cand_preds = np.array([0.000401] * 50)  # Near perfect
    base_preds = np.array([0.000600] * 50)  # Worse

    dec, _ = evaluate_horizon_certification(
        horizon=1,
        candidate_family="tcn",
        cand_preds=cand_preds,
        base_preds=base_preds,
        actuals=actuals,
    )
    assert dec.outcome == "certified_learned_model"
    assert dec.relative_qlike < 1.0
    assert dec.passed_all_gates is True


def test_evaluate_horizon_certification_abstains_on_inferior_candidate() -> None:
    actuals = np.array([0.0004] * 50)
    cand_preds = np.array([0.000800] * 50)  # Worse
    base_preds = np.array([0.000401] * 50)  # Near perfect

    dec, _ = evaluate_horizon_certification(
        horizon=3,
        candidate_family="tcn",
        cand_preds=cand_preds,
        base_preds=base_preds,
        actuals=actuals,
    )
    assert dec.outcome == "abstention"
    assert dec.passed_all_gates is False
