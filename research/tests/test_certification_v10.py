"""Tests for selection statistics and sealed single-use certification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from research.volatility_forecasting.certification_v10 import (
    evaluate_sealed_certification,
    record_test_opening_atomic,
)
from research.volatility_forecasting.horizon_selection_v10 import (
    circular_block_bootstrap_ratio_upper_95,
    diebold_mariano_hac_p_value,
    holm_bonferroni_adjustment,
    select_horizon_champions,
)


def test_circular_block_bootstrap_and_dm_hac() -> None:
    rng = np.random.default_rng(42)
    # Candidate clearly beats baseline
    base = rng.uniform(0.0004, 0.0006, size=100)
    cand = base * 0.85

    upper_95 = circular_block_bootstrap_ratio_upper_95(cand, base, n_resamples=500)
    assert upper_95 < 1.0

    p_val = diebold_mariano_hac_p_value(cand, base, horizon=5)
    assert p_val < 0.05


def test_holm_bonferroni_monotonicity() -> None:
    raw_p = [0.01, 0.04, 0.03, 0.20]
    adj = holm_bonferroni_adjustment(raw_p)
    assert len(adj) == 4
    assert adj[0] <= adj[2] <= adj[1] <= adj[3]


def test_select_horizon_champions_marks_baseline_fallback() -> None:
    # No records -> fallback
    sels = select_horizon_champions([], horizons=[1, 5])
    assert sels[1].selected_role == "development_baseline_candidate"
    assert sels[5].selected_role == "development_baseline_candidate"
    assert sels[1].passed_all_gates is False


def test_atomic_single_use_test_opening_prevents_duplicate_opening(tmp_path: Path) -> None:
    receipt_file = tmp_path / "test_opening.json"
    p1 = record_test_opening_atomic(receipt_file, "run-1", "pkg_sha", "split_sha")
    assert p1.exists()

    with pytest.raises(PermissionError, match="Duplicate test opening is strictly forbidden"):
        record_test_opening_atomic(receipt_file, "run-1", "pkg_sha", "split_sha")


def test_certification_fails_closed_when_data_ineligible(tmp_path: Path) -> None:
    pkg = {"candidate_name": "tcn_h1", "weights_sha256": "0" * 64}
    preds = {1: np.array([0.0004])}
    acts = {1: np.array([0.0004])}

    report = evaluate_sealed_certification(
        run_id="run-test",
        protocol_version="volatility-v10",
        candidate_package=pkg,
        candidate_predictions_by_horizon=preds,
        baseline_predictions_by_horizon=preds,
        test_actuals_by_horizon=acts,
        transfer_candidate_preds_by_horizon=preds,
        transfer_baseline_preds_by_horizon=preds,
        transfer_actuals_by_horizon=acts,
        universe_eligible=False,
        market_panel_eligible=False,
    )
    assert report.data_eligibility_verified is False
    assert len(report.certified_horizons) == 0
    assert report.decisions_by_horizon["1"]["certified_status"] == "abstained_data_ineligible"
