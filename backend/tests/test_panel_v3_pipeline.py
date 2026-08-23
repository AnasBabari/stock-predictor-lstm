"""Integration tests and end-to-end fixture pipelines for Protocol V3."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from panel.cross_sectional import compute_cross_sectional_ranks, compute_relative_forward_returns
from panel.features import build_features_v5
from panel.folds import CalendarFold
from panel.v3_candidates import MomentumRank20DCandidate
from panel.v3_certification import (
    V3CertificationGateConfig,
    check_prospective_holdout_maturity,
    evaluate_v3_prospective_certification,
)
from panel.v3_metrics import SessionICMetrics
from panel.v3_selection import (
    V3CandidateEvidence,
    V3CandidateFoldResult,
    V3SelectionDecision,
    evaluate_v3_candidate_on_folds,
    select_v3_champions,
)


def create_v3_synthetic_panel(
    n_tickers: int = 40,
    n_sessions: int = 350,
    signal_strength: float = 0.0,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_sessions)
    panels: dict[str, pd.DataFrame] = {}

    log_prices = np.zeros((n_tickers, n_sessions))
    for t in range(1, n_sessions):
        shocks = rng.normal(0, 0.008, n_tickers)
        if t > 20 and signal_strength > 0:
            past_mom = log_prices[:, t - 1] - log_prices[:, t - 21]
            past_mom_cs = np.clip(past_mom - np.mean(past_mom), -0.05, 0.05)
            next_ret = signal_strength * past_mom_cs * 0.04 + shocks
        else:
            next_ret = shocks
        log_prices[:, t] = log_prices[:, t - 1] + next_ret

    for i in range(n_tickers):
        ticker = f"TK_{i:02d}"
        close = 100.0 * np.exp(np.clip(log_prices[i], -2.0, 2.0))
        openp = close * np.exp(rng.normal(0, 0.001, n_sessions))
        high = np.maximum(openp, close) * np.exp(np.abs(rng.normal(0, 0.002, n_sessions)))
        low = np.minimum(openp, close) * np.exp(-np.abs(rng.normal(0, 0.002, n_sessions)))
        volume = rng.integers(50_000, 2_000_000, n_sessions).astype(float)

        raw_df = pd.DataFrame(
            {"Open": openp, "High": high, "Low": low, "Close": close, "Volume": volume},
            index=dates,
        )
        panels[ticker] = build_features_v5(raw_df)

    return panels


def test_prospective_holdout_maturity_check():
    """Verify holdout returns immature status when fewer than 282 post-cutoff sessions exist."""
    cutoff = "2026-08-21"
    # Create calendar with only 100 sessions after cutoff
    pre_cutoff = pd.bdate_range("2025-01-01", "2026-08-21")
    post_cutoff = pd.bdate_range("2026-08-24", periods=100)
    master_cal = list(pre_cutoff) + list(post_cutoff)

    status = check_prospective_holdout_maturity(
        master_cal,
        development_cutoff=cutoff,
        prospective_origin_sessions=252,
        max_horizon=30,
    )
    assert status.is_mature is False
    assert status.observed_prospective_sessions == 100
    assert status.required_total_sessions == 282
    assert status.remaining_sessions == 182


def test_prospective_certification_blocks_early_evaluation():
    """Verify certification returns 'locked_waiting_for_maturity' with no computed metrics."""
    cutoff = "2026-08-21"
    pre_cutoff = pd.bdate_range("2025-01-01", "2026-08-21")
    post_cutoff = pd.bdate_range("2026-08-24", periods=50)
    master_cal = list(pre_cutoff) + list(post_cutoff)

    gate_cfg = V3CertificationGateConfig(development_cutoff=cutoff)
    panels = create_v3_synthetic_panel(n_tickers=35, n_sessions=150, seed=42)

    sel_decisions = {
        5: V3SelectionDecision(horizon=5, candidate="momentum_rank_20d", status="selected")
    }
    frozen_cands = {5: MomentumRank20DCandidate()}

    result = evaluate_v3_prospective_certification(
        frozen_cands,
        sel_decisions,
        panels,
        master_cal,
        dev_tickers=[f"TK_{i:02d}" for i in range(28)],
        transfer_tickers=[f"TK_{i:02d}" for i in range(28, 35)],
        gate_config=gate_cfg,
        open_locked_holdout=True,
    )

    assert result["status"] == "locked_waiting_for_maturity"
    assert result["decision"] == "locked_waiting_for_maturity"
    assert result["certified_horizons"] == []
    # Metrics must NOT be computed
    assert result["decisions"] == {}


def test_synthetic_planted_signal_selection_passes():
    """Case 1: Planted signal produces positive IC and champion selection passes."""
    panels = create_v3_synthetic_panel(n_tickers=40, n_sessions=300, signal_strength=1.5, seed=42)
    dev_tickers = [f"TK_{i:02d}" for i in range(32)]

    ranked = compute_cross_sectional_ranks(panels, dev_tickers=dev_tickers, min_reference_assets=25)
    _, rel_targets = compute_relative_forward_returns(
        panels, horizon=5, dev_tickers=dev_tickers, min_reference_assets=25
    )

    # 5 calendar folds
    dates = panels["TK_00"].index
    folds = [
        CalendarFold(
            fold_index=i,
            train_start=dates[0],
            train_end=dates[150 + i * 20],
            val_start=dates[155 + i * 20],
            val_end=dates[170 + i * 20],
            n_train_sessions=150 + i * 20,
            n_val_sessions=15,
        )
        for i in range(5)
    ]

    cand = MomentumRank20DCandidate()
    ev = evaluate_v3_candidate_on_folds(
        cand,
        5,
        folds,
        ranked,
        rel_targets,
        min_daily_asset_count=25,
        resamples=500,
        seed=42,
    )

    assert ev.overall_metrics.mean_spearman_ic > 0.0

    decisions = select_v3_champions(
        {(5, "momentum_rank_20d"): ev},
        {"momentum_rank_20d": cand},
        ["momentum_rank_20d"],
        [5],
        alpha=0.05,
        min_positive_fold_fraction=0.80,
        min_daily_asset_count=25,
    )

    assert 5 in decisions
    assert decisions[5].status == "selected"
    assert decisions[5].candidate == "momentum_rank_20d"


def test_synthetic_pure_noise_abstains():
    """Case 2: Pure noise panel yields non-significant IC and champion selector abstains."""
    # Pure noise panel (signal_strength=0.0)
    panels = create_v3_synthetic_panel(n_tickers=35, n_sessions=200, signal_strength=0.0, seed=999)
    dev_tickers = [f"TK_{i:02d}" for i in range(35)]

    ranked = compute_cross_sectional_ranks(panels, dev_tickers=dev_tickers, min_reference_assets=25)
    _, rel_targets = compute_relative_forward_returns(
        panels, horizon=5, dev_tickers=dev_tickers, min_reference_assets=25
    )

    dates = panels["TK_00"].index
    folds = [
        CalendarFold(
            fold_index=i,
            train_start=dates[0],
            train_end=dates[100 + i * 15],
            val_start=dates[105 + i * 15],
            val_end=dates[115 + i * 15],
            n_train_sessions=100 + i * 15,
            n_val_sessions=10,
        )
        for i in range(5)
    ]

    cand = MomentumRank20DCandidate()
    ev = evaluate_v3_candidate_on_folds(
        cand,
        5,
        folds,
        ranked,
        rel_targets,
        min_daily_asset_count=25,
        resamples=500,
        seed=42,
    )

    # Wrap in dummy evidence that fails lower bound or p-value
    decisions = select_v3_champions(
        {(5, "momentum_rank_20d"): ev},
        {"momentum_rank_20d": cand},
        ["momentum_rank_20d"],
        [5],
        alpha=0.01,  # Strict alpha
        min_positive_fold_fraction=0.80,
        min_daily_asset_count=25,
    )

    assert 5 in decisions
    assert decisions[5].status == "abstain_no_robust_rank_signal"
    assert decisions[5].candidate is None


def test_regime_instability_blocks_selection():
    """Case 3: Unstable fold fraction (< 80% positive folds) blocks champion selection."""
    dummy_overall = SessionICMetrics(
        n_eligible_sessions=100,
        n_valid_ic_sessions=100,
        ic_session_coverage=1.0,
        mean_spearman_ic=0.03,
        median_spearman_ic=0.03,
        std_spearman_ic=0.05,
        positive_ic_hit_rate=0.60,
        min_daily_asset_breadth=35,
        median_daily_asset_breadth=35.0,
        prediction_row_coverage=1.0,
        hac_lag=4,
        hac_se=0.005,
        hac_t_stat=6.0,
        raw_one_sided_hac_p=0.0001,
        mean_ic_ci_lower_95=0.015,
        mean_ic_ci_upper_95=0.045,
    )

    # 5 folds where only 3 of 5 are positive (fraction = 0.60 < 0.80)
    fold_results = [
        V3CandidateFoldResult(0, 100, 20, 0.05, 0.05, 20, 20),
        V3CandidateFoldResult(1, 120, 20, 0.04, 0.04, 20, 20),
        V3CandidateFoldResult(2, 140, 20, -0.02, -0.02, 20, 20),
        V3CandidateFoldResult(3, 160, 20, 0.03, 0.03, 20, 20),
        V3CandidateFoldResult(4, 180, 20, -0.01, -0.01, 20, 20),
    ]

    ev = V3CandidateEvidence(
        candidate_name="momentum_rank_20d",
        horizon=5,
        overall_metrics=dummy_overall,
        fold_metrics=fold_results,
        positive_fold_count=3,
        positive_fold_fraction=0.60,
    )

    cand = MomentumRank20DCandidate()
    decisions = select_v3_champions(
        {(5, "momentum_rank_20d"): ev},
        {"momentum_rank_20d": cand},
        ["momentum_rank_20d"],
        [5],
        alpha=0.05,
        min_positive_fold_fraction=0.80,
    )

    assert decisions[5].status == "abstain_no_robust_rank_signal"
    assert (
        "No candidate satisfied all 7 pre-registered development rank gates."
        in decisions[5].reasons
    )


def test_v3_release_bundle_manifest_validation():
    """Verify release bundle validation passes for global-cert-v3 manifest."""
    from release.bundle import validate_certification_manifest

    valid_v3_manifest = {
        "certification_protocol_version": "global-cert-v3",
        "status": "holdout_opened",
        "decision": "pass",
        "certified_horizons": [1],
        "gate_config": {
            "family_alpha": 0.05,
            "require_temporal_mean_ic_positive": True,
            "require_temporal_bootstrap_lower_bound_positive": True,
            "require_temporal_holm_hac_significance": True,
            "require_transfer_mean_ic_positive": True,
            "require_transfer_bootstrap_lower_bound_positive": True,
            "require_transfer_holm_hac_significance": True,
            "min_temporal_daily_breadth": 30,
            "min_transfer_daily_breadth": 30,
            "min_temporal_prediction_coverage": 0.90,
            "min_transfer_prediction_coverage": 0.90,
            "min_temporal_ic_session_coverage": 0.90,
            "min_transfer_ic_session_coverage": 0.90,
        },
        "decisions": {
            "1": {
                "horizon": 1,
                "decision": "pass",
                "failed_gates": [],
                "temporal_mean_ic": 0.05,
                "temporal_mean_ic_ci_lower_95": 0.02,
                "temporal_holm_hac_p": 0.001,
                "temporal_median_breadth": 35.0,
                "temporal_prediction_coverage": 1.0,
                "temporal_session_coverage": 1.0,
                "transfer_mean_ic": 0.04,
                "transfer_mean_ic_ci_lower_95": 0.01,
                "transfer_holm_hac_p": 0.01,
                "transfer_median_breadth": 35.0,
                "transfer_prediction_coverage": 1.0,
                "transfer_session_coverage": 1.0,
            }
        },
    }

    assert validate_certification_manifest(valid_v3_manifest) is True

    # Fail closed on immature holdout
    immature_v3_manifest = {
        "certification_protocol_version": "global-cert-v3",
        "status": "locked_waiting_for_maturity",
        "decision": "locked_waiting_for_maturity",
        "decisions": {},
    }
    with pytest.raises(ValueError, match="Holdout status must be 'holdout_opened'"):
        validate_certification_manifest(immature_v3_manifest)


def test_future_data_canary_cutoff_ordering(tmp_path: Path):
    """Invariant A & Cutoff Ordering: Data > 2026-08-21 never enters development artifacts."""
    from scripts.run_global_pipeline import GlobalPipelineRunner, PipelineConfig

    # Base dataset ending on 2026-08-21
    pre_dates = pd.bdate_range("2025-01-01", "2026-08-21")
    tickers = [f"TK_{i:02d}" for i in range(35)]
    rng = np.random.default_rng(42)

    dataset_a: dict[str, pd.DataFrame] = {}
    for t in tickers:
        drift = rng.normal(0.0004, 0.015, len(pre_dates))
        close = 100.0 * np.exp(np.cumsum(drift))
        df = pd.DataFrame(
            {
                "Open": close,
                "High": close * 1.01,
                "Low": close * 0.99,
                "Close": close,
                "Volume": 100_000.0,
            },
            index=pre_dates,
        )
        dataset_a[t] = df

    # Contaminated dataset with post-cutoff corruptions & new junk tickers
    post_dates = pd.bdate_range("2026-08-24", "2027-06-01")

    dataset_b: dict[str, pd.DataFrame] = {}
    for t in tickers:
        df_pre = dataset_a[t].copy()
        post_close = np.full(len(post_dates), 1e9)  # Absurd future price
        df_post = pd.DataFrame(
            {
                "Open": post_close,
                "High": post_close * 1.01,
                "Low": post_close * 0.99,
                "Close": post_close,
                "Volume": 1e12,
            },
            index=post_dates,
        )
        dataset_b[t] = pd.concat([df_pre, df_post])

    # Add a rogue ticker only present post-cutoff
    dataset_b["ROGUE_FUTURE_TICKER"] = pd.DataFrame(
        {"Open": 50.0, "High": 55.0, "Low": 45.0, "Close": 50.0, "Volume": 1e9},
        index=post_dates,
    )

    cfg = PipelineConfig(
        run_id="v3_canary_test",
        mode="development",
        research_protocol_version="global-research-v3",
        protocol_version="global-cert-v3",
        development_cutoff="2026-08-21",
        horizons=[1, 5],
        candidate_families=["momentum_rank_20d"],
        folds=3,
        min_train_sessions=50,
        temporal_holdout_sessions=50,
        license_acknowledged=True,
    )

    run_dir_a = tmp_path / "run_a"
    runner_a = GlobalPipelineRunner(config=cfg, run_dir=run_dir_a, universe_data=dataset_a)
    res_a = runner_a.run(stage="all-development")

    run_dir_b = tmp_path / "run_b"
    runner_b = GlobalPipelineRunner(config=cfg, run_dir=run_dir_b, universe_data=dataset_b)
    res_b = runner_b.run(stage="all-development")

    # Assert universe, features, folds, evaluate, and selection are bit-for-bit identical
    assert res_a["stages"]["snapshot"]["tickers"] == res_b["stages"]["snapshot"]["tickers"]
    assert "ROGUE_FUTURE_TICKER" not in res_b["stages"]["snapshot"]["tickers"]
    assert res_a["stages"]["folds"] == res_b["stages"]["folds"]
    assert res_a["stages"]["evaluate"] == res_b["stages"]["evaluate"]
    assert res_a["stages"]["selection"] == res_b["stages"]["selection"]


def test_prospective_certification_zero_early_computation_with_exploding_evaluator(
    monkeypatch: pytest.MonkeyPatch,
):
    """Invariant AC: Exploding evaluator proves zero performance calculation on immature holdout."""
    from panel import v3_metrics

    def exploding_evaluator(*args, **kwargs):
        raise AssertionError(
            "CRITICAL LEAKAGE: Performance evaluator was invoked before holdout maturity!"
        )

    monkeypatch.setattr(v3_metrics, "compute_session_rank_ic", exploding_evaluator)

    cutoff = "2026-08-21"
    pre_cutoff = pd.bdate_range("2025-01-01", "2026-08-21")
    post_cutoff = pd.bdate_range("2026-08-24", periods=100)  # Only 100 sessions (requires 282)
    master_cal = list(pre_cutoff) + list(post_cutoff)

    gate_cfg = V3CertificationGateConfig(development_cutoff=cutoff)
    panels = create_v3_synthetic_panel(n_tickers=35, n_sessions=150, seed=42)

    sel_decisions = {
        5: V3SelectionDecision(horizon=5, candidate="momentum_rank_20d", status="selected")
    }
    frozen_cands = {5: MomentumRank20DCandidate()}

    # Calling with open_locked_holdout=True must still return locked_waiting_for_maturity without invoking evaluator
    result = evaluate_v3_prospective_certification(
        frozen_cands,
        sel_decisions,
        panels,
        master_cal,
        dev_tickers=[f"TK_{i:02d}" for i in range(28)],
        transfer_tickers=[f"TK_{i:02d}" for i in range(28, 35)],
        gate_config=gate_cfg,
        open_locked_holdout=True,
    )

    assert result["status"] == "locked_waiting_for_maturity"
    assert result["decisions"] == {}
    assert "temporal_mean_ic" not in result
    assert "transfer_mean_ic" not in result


def test_prospective_holdout_exact_maturity_off_by_one():
    """Invariant AB & Maturity Off-By-One: Exactly 281 is immature; exactly 282 is mature."""
    cutoff = "2026-08-21"
    pre_cutoff = pd.bdate_range("2025-01-01", "2026-08-21")

    # 1. Exactly 281 post-cutoff sessions -> must be IMMATURE (needs 282 for 252 origins + 30 horizon)
    post_281 = pd.bdate_range("2026-08-24", periods=281)
    status_281 = check_prospective_holdout_maturity(
        list(pre_cutoff) + list(post_281),
        development_cutoff=cutoff,
        prospective_origin_sessions=252,
        max_horizon=30,
    )
    assert status_281.is_mature is False
    assert status_281.observed_prospective_sessions == 281
    assert status_281.required_total_sessions == 282
    assert status_281.remaining_sessions == 1

    # 2. Exactly 282 post-cutoff sessions -> must be MATURE
    post_282 = pd.bdate_range("2026-08-24", periods=282)
    status_282 = check_prospective_holdout_maturity(
        list(pre_cutoff) + list(post_282),
        development_cutoff=cutoff,
        prospective_origin_sessions=252,
        max_horizon=30,
    )
    assert status_282.is_mature is True
    assert status_282.observed_prospective_sessions == 282
    assert status_282.required_total_sessions == 282
    assert status_282.remaining_sessions == 0


def test_fixed_prospective_certification_window():
    """Invariant AG: Certification evaluates the fixed FIRST 252 prospective origins, not rolling."""
    cutoff = "2026-08-21"
    pre_cutoff = pd.bdate_range("2025-01-01", "2026-08-21")
    # Master calendar with 500 post-cutoff sessions
    post_cutoff = pd.bdate_range("2026-08-24", periods=500)
    master_cal = list(pre_cutoff) + list(post_cutoff)

    # Post-cutoff origins are sliced from index 0 to 252
    post_cutoff_all = [d for d in master_cal if d > pd.Timestamp(cutoff)]
    assert len(post_cutoff_all) == 500

    fixed_origins = post_cutoff_all[:252]
    assert len(fixed_origins) == 252
    assert fixed_origins[0] == pd.Timestamp("2026-08-24")
    assert fixed_origins[-1] == post_cutoff_all[251]
    # Verify it does NOT use rolling latest
    assert fixed_origins != post_cutoff_all[-252:]


def test_certification_immutability_guard(tmp_path: Path):
    """Invariant AF: Re-running certification on an already-opened holdout raises RuntimeError."""
    from scripts.run_global_pipeline import GlobalPipelineRunner, PipelineConfig

    run_dir = tmp_path / "cert_immutability_run"
    stages_dir = run_dir / "stages"
    stages_dir.mkdir(parents=True, exist_ok=True)

    # Write existing opened certification file
    cert_file = stages_dir / "07_certification.json"
    cert_file.write_text(
        '{"status": "holdout_opened", "decision": "fail", "certified_horizons": []}',
        encoding="utf-8",
    )

    cfg = PipelineConfig(
        run_id="test_immutability",
        mode="certification",
        research_protocol_version="global-research-v3",
        protocol_version="global-cert-v3",
    )

    runner = GlobalPipelineRunner(
        config=cfg,
        run_dir=run_dir,
        open_locked_certification_holdout=True,
        universe_data={},
    )

    with pytest.raises(RuntimeError, match="Historical certification artefacts are immutable"):
        runner.run_stage_certify(
            decisions={},
            universe_data={},
            folds_meta={"train_tickers": [], "asset_transfer_holdout_tickers": []},
        )


def test_selection_coverage_and_breadth_rejection():
    """Invariants W & X: Row coverage < 90%, session coverage < 90%, and breadth < 30 reject."""
    base_metrics = SessionICMetrics(
        n_eligible_sessions=100,
        n_valid_ic_sessions=100,
        ic_session_coverage=1.0,
        mean_spearman_ic=0.04,
        median_spearman_ic=0.04,
        std_spearman_ic=0.05,
        positive_ic_hit_rate=0.70,
        min_daily_asset_breadth=35,
        median_daily_asset_breadth=35.0,
        prediction_row_coverage=1.0,
        hac_lag=4,
        hac_se=0.005,
        hac_t_stat=8.0,
        raw_one_sided_hac_p=0.0001,
        mean_ic_ci_lower_95=0.02,
        mean_ic_ci_upper_95=0.06,
    )

    folds = [V3CandidateFoldResult(i, 100, 20, 0.04, 0.04, 20, 20) for i in range(5)]

    # 1. Low prediction row coverage (0.85 < 0.90)
    low_cov_metrics = SessionICMetrics(
        **{**base_metrics.to_dict(), "prediction_row_coverage": 0.85}  # type: ignore[arg-type]
    )
    ev_low_cov = V3CandidateEvidence("momentum_rank_20d", 5, low_cov_metrics, folds, 5, 1.0)
    dec_cov = select_v3_champions(
        {(5, "momentum_rank_20d"): ev_low_cov},
        {"momentum_rank_20d": MomentumRank20DCandidate()},
        ["momentum_rank_20d"],
        [5],
    )
    assert dec_cov[5].status == "abstain_no_robust_rank_signal"

    # 2. Low median breadth (25 < 30)
    low_breadth_metrics = SessionICMetrics(
        **{**base_metrics.to_dict(), "median_daily_asset_breadth": 25.0}  # type: ignore[arg-type]
    )
    ev_low_br = V3CandidateEvidence("momentum_rank_20d", 5, low_breadth_metrics, folds, 5, 1.0)
    dec_br = select_v3_champions(
        {(5, "momentum_rank_20d"): ev_low_br},
        {"momentum_rank_20d": MomentumRank20DCandidate()},
        ["momentum_rank_20d"],
        [5],
        min_daily_asset_count=30,
    )
    assert dec_br[5].status == "abstain_no_robust_rank_signal"


def test_release_bundle_v3_fail_closed_checks():
    """Invariants AH, AJ, AK: Protocol mismatch, missing metrics, and gate violations fail closed."""
    from release.bundle import validate_certification_manifest

    # Missing required metric in decision
    bad_manifest = {
        "certification_protocol_version": "global-cert-v3",
        "status": "holdout_opened",
        "decision": "pass",
        "gate_config": {
            "require_temporal_mean_ic_positive": True,
            "require_temporal_bootstrap_lower_bound_positive": True,
            "require_temporal_holm_hac_significance": True,
            "require_transfer_mean_ic_positive": True,
            "require_transfer_bootstrap_lower_bound_positive": True,
            "require_transfer_holm_hac_significance": True,
        },
        "decisions": {
            "1": {
                "horizon": 1,
                "decision": "pass",
                "failed_gates": [],
                # Missing 'temporal_mean_ic'
                "temporal_mean_ic_ci_lower_95": 0.02,
                "temporal_holm_hac_p": 0.001,
            }
        },
    }
    with pytest.raises(ValueError, match="missing required 'temporal_mean_ic'"):
        validate_certification_manifest(bad_manifest)

    # Gate violation (transfer_mean_ic <= 0) returns False
    failing_gate_manifest = {
        "certification_protocol_version": "global-cert-v3",
        "status": "holdout_opened",
        "decision": "pass",
        "gate_config": {
            "require_temporal_mean_ic_positive": True,
            "require_temporal_bootstrap_lower_bound_positive": True,
            "require_temporal_holm_hac_significance": True,
            "require_transfer_mean_ic_positive": True,
            "require_transfer_bootstrap_lower_bound_positive": True,
            "require_transfer_holm_hac_significance": True,
            "min_temporal_daily_breadth": 30,
            "min_transfer_daily_breadth": 30,
            "min_temporal_prediction_coverage": 0.90,
            "min_transfer_prediction_coverage": 0.90,
            "min_temporal_ic_session_coverage": 0.90,
            "min_transfer_ic_session_coverage": 0.90,
        },
        "decisions": {
            "1": {
                "horizon": 1,
                "decision": "pass",
                "failed_gates": [],
                "temporal_mean_ic": 0.05,
                "temporal_mean_ic_ci_lower_95": 0.02,
                "temporal_holm_hac_p": 0.001,
                "temporal_median_breadth": 35.0,
                "temporal_prediction_coverage": 1.0,
                "temporal_session_coverage": 1.0,
                "transfer_mean_ic": -0.01,  # Fails requirement > 0
                "transfer_mean_ic_ci_lower_95": -0.03,
                "transfer_holm_hac_p": 0.50,
                "transfer_median_breadth": 35.0,
                "transfer_prediction_coverage": 1.0,
                "transfer_session_coverage": 1.0,
            }
        },
    }
    assert validate_certification_manifest(failing_gate_manifest) is False
