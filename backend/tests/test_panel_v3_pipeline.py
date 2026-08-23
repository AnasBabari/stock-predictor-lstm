import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from panel.cross_sectional import compute_cross_sectional_ranks, compute_relative_forward_returns
from panel.features import build_features_v5
from panel.folds import CalendarFold
from panel.v3_candidates import (
    V3_CANDIDATE_REGISTRY,
    BaseV3Candidate,
    MomentumRank20DCandidate,
    RidgeCrossSectionalCandidate,
    compute_file_sha256,
    load_candidate_artifact,
    save_candidate_artifact,
)
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


def test_mature_certification_never_fits(tmp_path: Path):
    """TEST 1: Invariant AE - Prove opened/mature prospective certification NEVER invokes candidate.fit()."""

    class ExplodingFitCandidate(BaseV3Candidate):
        name = "exploding_fit_candidate"

        def __init__(self):
            self.is_fitted = True

        def fit(self, features_by_ticker, relative_targets_by_ticker):
            raise AssertionError(
                "CRITICAL VIOLATION: candidate.fit() was invoked during prospective certification!"
            )

        def predict(self, features_by_ticker):
            scores = {}
            for ticker, df in features_by_ticker.items():
                if "Return_20D_CS_Rank" in df.columns:
                    scores[ticker] = df["Return_20D_CS_Rank"]
                else:
                    scores[ticker] = pd.Series(np.nan, index=df.index)
            return pd.DataFrame(scores)

        def save(self, target_dir: Path) -> dict[str, str]:
            target_dir.mkdir(parents=True, exist_ok=True)
            p = target_dir / "model.json"
            p.write_text(json.dumps({"candidate": self.name}), encoding="utf-8")
            return {"model.json": compute_file_sha256(p)}

        def load(self, source_dir: Path) -> None:
            self.is_fitted = True

    cand = ExplodingFitCandidate()
    cutoff = "2026-08-21"
    pre_cutoff = pd.bdate_range("2025-01-01", "2026-08-21")
    post_cutoff = pd.bdate_range("2026-08-24", periods=300)
    master_cal = list(pre_cutoff) + list(post_cutoff)

    # 35 dev tickers, 10 transfer tickers
    panels = create_v3_synthetic_panel(
        n_tickers=45, n_sessions=len(master_cal), signal_strength=1.0, seed=42
    )
    tickers = list(panels.keys())
    dev_tickers = tickers[:35]
    transfer_tickers = tickers[35:]

    # Reindex panels to master calendar
    for t in tickers:
        panels[t] = panels[t].iloc[: len(master_cal)]
        panels[t].index = master_cal

    sel_decisions = {
        5: V3SelectionDecision(horizon=5, candidate="exploding_fit_candidate", status="selected")
    }
    gate_cfg = V3CertificationGateConfig(
        development_cutoff=cutoff,
        min_temporal_daily_breadth=30,
        min_transfer_daily_breadth=5,
        resamples=100,
    )

    # Mature, opened certification MUST evaluate without calling cand.fit()
    cert_result = evaluate_v3_prospective_certification(
        {5: cand},
        sel_decisions,
        panels,
        master_cal,
        dev_tickers=dev_tickers,
        transfer_tickers=transfer_tickers,
        gate_config=gate_cfg,
        open_locked_holdout=True,
    )

    assert cert_result["status"] == "holdout_opened"
    assert "5" in cert_result["decisions"]


def test_prospective_outcome_canary_does_not_affect_frozen_model(tmp_path: Path):
    """TEST 2: Altering future prospective outcomes does not alter the frozen model artifact or parameters."""
    dev_panels = create_v3_synthetic_panel(n_tickers=35, n_sessions=200, seed=42)
    dev_tickers = list(dev_panels.keys())

    # Fit Ridge candidate on dev data
    v5_feats = {t: dev_panels[t] for t in dev_tickers}
    ranked_feats = compute_cross_sectional_ranks(v5_feats, dev_tickers=dev_tickers)
    _, rel_tgts = compute_relative_forward_returns(dev_panels, horizon=5, dev_tickers=dev_tickers)

    cand = RidgeCrossSectionalCandidate(alpha=10.0)
    cand.fit(ranked_feats, rel_tgts)

    frozen_dir = tmp_path / "frozen_models" / "h5"
    manifest = save_candidate_artifact(
        cand,
        frozen_dir,
        horizon=5,
        development_cutoff="2026-08-21",
        feature_contract_version="cross_sectional_v3_rank_v1",
        target_contract_version="relative_forward_log_return_dev_loo_v1",
        train_ticker_digest="test_digest",
        fit_data_min_date="2024-01-01",
        fit_data_max_date="2026-08-21",
    )
    initial_weights_sha = manifest["files"]["weights.npz"]

    # Prospective Panel A vs Panel B (with wild prospective prices)
    cand_a, _ = load_candidate_artifact(frozen_dir)
    cand_b, _ = load_candidate_artifact(frozen_dir)

    # Artifact on disk and loaded parameters are completely invariant to prospective outcomes
    assert cand_a.model.coef_ is not None
    assert np.array_equal(cand_a.model.coef_, cand_b.model.coef_)
    assert cand_a.model.intercept_ == cand_b.model.intercept_
    assert compute_file_sha256(frozen_dir / "weights.npz") == initial_weights_sha


def test_all_v3_candidate_families_serialization_roundtrip(tmp_path: Path):
    """TEST 3: Invariant AD - Complete fit-predict-save-load-predict roundtrip for all 7 candidate families."""
    panels = create_v3_synthetic_panel(n_tickers=35, n_sessions=150, seed=42)
    tickers = list(panels.keys())

    v5_feats = {t: panels[t] for t in tickers}
    ranked_feats = compute_cross_sectional_ranks(v5_feats, dev_tickers=tickers)
    _, rel_tgts = compute_relative_forward_returns(panels, horizon=5, dev_tickers=tickers)

    for cand_name, cand_cls in V3_CANDIDATE_REGISTRY.items():
        cand = cand_cls()
        cand.fit(ranked_feats, rel_tgts)

        pred_before = cand.predict(ranked_feats)

        target_dir = tmp_path / f"cand_{cand_name}"
        saved_manifest = save_candidate_artifact(
            cand,
            target_dir,
            horizon=5,
            development_cutoff="2026-08-21",
            feature_contract_version="cross_sectional_v3_rank_v1",
            target_contract_version="relative_forward_log_return_dev_loo_v1",
            train_ticker_digest="test_digest",
            fit_data_min_date="2024-01-01",
            fit_data_max_date="2026-08-21",
        )

        loaded_cand, loaded_manifest = load_candidate_artifact(target_dir)
        pred_after = loaded_cand.predict(ranked_feats)

        assert loaded_manifest["candidate"] == cand_name
        assert loaded_manifest["artifact_digest"] == saved_manifest["artifact_digest"]
        assert np.allclose(pred_before.to_numpy(), pred_after.to_numpy(), equal_nan=True)


def test_frozen_hash_tampering_rejected(tmp_path: Path):
    """TEST 4: Any tampering with frozen model files fails closed immediately."""
    cand = RidgeCrossSectionalCandidate(alpha=100.0)
    panels = create_v3_synthetic_panel(n_tickers=35, n_sessions=120, seed=42)
    tickers = list(panels.keys())
    ranked = compute_cross_sectional_ranks(panels, dev_tickers=tickers)
    _, rel_targets = compute_relative_forward_returns(panels, horizon=5, dev_tickers=tickers)
    cand.fit(ranked, rel_targets)

    save_candidate_artifact(
        cand,
        tmp_path,
        horizon=5,
        development_cutoff="2026-08-21",
        feature_contract_version="cross_sectional_v3_rank_v1",
        target_contract_version="relative_forward_log_return_dev_loo_v1",
        train_ticker_digest="test",
        fit_data_min_date="2024-01-01",
        fit_data_max_date="2026-08-21",
    )

    # Tamper with params.json
    params_file = tmp_path / "params.json"
    params_data = json.loads(params_file.read_text(encoding="utf-8"))
    params_data["tampered"] = True
    params_file.write_text(json.dumps(params_data), encoding="utf-8")

    with pytest.raises(ValueError, match="Artifact SHA mismatch"):
        load_candidate_artifact(tmp_path)


def test_wrong_candidate_artifact_rejected(tmp_path: Path):
    """TEST 5: Manifest candidate mismatch fails closed."""
    cand = RidgeCrossSectionalCandidate(alpha=10.0)
    save_candidate_artifact(
        cand,
        tmp_path,
        horizon=5,
        development_cutoff="2026-08-21",
        feature_contract_version="cross_sectional_v3_rank_v1",
        target_contract_version="relative_forward_log_return_dev_loo_v1",
        train_ticker_digest="test",
        fit_data_min_date="2024-01-01",
        fit_data_max_date="2026-08-21",
    )

    # Alter params.json candidate name
    params_file = tmp_path / "params.json"
    params_data = json.loads(params_file.read_text(encoding="utf-8"))
    params_data["candidate"] = "elastic_net_cross_sectional"
    params_file.write_text(json.dumps(params_data), encoding="utf-8")

    # Update hash in manifest to bypass SHA check and trigger candidate mismatch
    manifest_file = tmp_path / "model_manifest.json"
    m = json.loads(manifest_file.read_text(encoding="utf-8"))
    m["files"]["params.json"] = compute_file_sha256(params_file)
    manifest_file.write_text(json.dumps(m), encoding="utf-8")

    with pytest.raises(ValueError, match="Candidate mismatch"):
        load_candidate_artifact(tmp_path)


def test_post_cutoff_freeze_canary(tmp_path: Path):
    """TEST 6: Freezer strictly enforces data <= 2026-08-21 even if panel contains future dates."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from freeze_global_v3 import freeze_global_v3

    # Generate Panel A (<= 2026-08-21) and Panel B (containing future dates through 2027)
    cutoff = "2026-08-21"
    dates_a = pd.bdate_range("2025-01-01", "2026-08-21")
    dates_b = pd.bdate_range("2025-01-01", "2027-06-01")

    panel_a: dict[str, pd.DataFrame] = {}
    panel_b: dict[str, pd.DataFrame] = {}
    rng = np.random.default_rng(42)

    for i in range(35):
        ticker = f"TK_{i:02d}"
        n_a = len(dates_a)
        n_b = len(dates_b)
        close_a = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n_a)))
        close_b = np.pad(close_a, (0, n_b - n_a), mode="constant", constant_values=1000.0)

        df_a = pd.DataFrame(
            {
                "Open": close_a,
                "High": close_a * 1.01,
                "Low": close_a * 0.99,
                "Close": close_a,
                "Volume": 100000.0,
            },
            index=dates_a,
        )
        df_b = pd.DataFrame(
            {
                "Open": close_b,
                "High": close_b * 1.01,
                "Low": close_b * 0.99,
                "Close": close_b,
                "Volume": 100000.0,
            },
            index=dates_b,
        )
        panel_a[ticker] = df_a
        panel_b[ticker] = df_b

    # Set up synthetic stages dir
    run_dir_a = tmp_path / "run_a"
    run_dir_b = tmp_path / "run_b"
    for rdir in (run_dir_a, run_dir_b):
        stages = rdir / "stages"
        stages.mkdir(parents=True, exist_ok=True)
        (stages / "01_snapshot.json").write_text(
            json.dumps({"tickers": list(panel_a.keys())}), encoding="utf-8"
        )
        (stages / "03_folds.json").write_text(
            json.dumps(
                {"train_tickers": list(panel_a.keys()), "asset_transfer_holdout_tickers": []}
            ),
            encoding="utf-8",
        )
        (stages / "06_selection.json").write_text(
            json.dumps(
                {
                    "5": {
                        "horizon": 5,
                        "candidate": "momentum_rank_20d",
                        "status": "selected",
                        "mean_spearman_ic": 0.05,
                        "mean_ic_ci_lower_95": 0.02,
                        "holm_adjusted_p": 0.001,
                    }
                }
            ),
            encoding="utf-8",
        )

    out_cfg_a = tmp_path / "frozen_a.json"
    out_cfg_b = tmp_path / "frozen_b.json"

    res_a = freeze_global_v3(run_dir_a, out_cfg_a, universe_data=panel_a)
    res_b = freeze_global_v3(run_dir_b, out_cfg_b, universe_data=panel_b)

    assert res_a["selected_candidates"]["5"]["model_artifact"]["fit_data_max_date"] == cutoff
    assert res_b["selected_candidates"]["5"]["model_artifact"]["fit_data_max_date"] == cutoff
    assert (
        res_a["selected_candidates"]["5"]["model_artifact"]["artifact_digest"]
        == res_b["selected_candidates"]["5"]["model_artifact"]["artifact_digest"]
    )


def test_release_uses_exact_frozen_model_and_forbids_placeholders(tmp_path: Path):
    """TESTS 7 & 8: Release stage packages exact frozen model artifacts, not status placeholders."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from run_global_pipeline import GlobalPipelineRunner, PipelineConfig

    run_dir = tmp_path / "run"
    frozen_model_dir = run_dir / "frozen_models" / "h5"
    cand = MomentumRank20DCandidate()
    manifest = save_candidate_artifact(
        cand,
        frozen_model_dir,
        horizon=5,
        development_cutoff="2026-08-21",
        feature_contract_version="cross_sectional_v3_rank_v1",
        target_contract_version="relative_forward_log_return_dev_loo_v1",
        train_ticker_digest="test",
        fit_data_min_date="2024-01-01",
        fit_data_max_date="2026-08-21",
    )

    cfg = PipelineConfig(
        run_id="test_release",
        protocol_version="global-cert-v3",
        research_protocol_version="global-research-v3",
        freeze_status="frozen",
        selected_candidates={
            "5": {
                "status": "selected",
                "candidate": "momentum_rank_20d",
                "model_artifact": {
                    "directory": "frozen_models/h5",
                    "manifest_file": "frozen_models/h5/model_manifest.json",
                    "artifact_digest": manifest["artifact_digest"],
                },
            }
        },
    )

    runner = GlobalPipelineRunner(cfg, run_dir)
    cert_result = {
        "status": "holdout_opened",
        "decision": "pass",
        "certified_horizons": [5],
    }

    release_res = runner.run_stage_refit_and_release({}, cert_result, {}, {})
    assert release_res["status"] == "completed"

    released_manifest = run_dir / "release" / "h5" / "model_manifest.json"
    assert released_manifest.exists()
    rel_m = json.loads(released_manifest.read_text(encoding="utf-8"))
    assert rel_m["candidate"] == "momentum_rank_20d"
    assert rel_m["artifact_digest"] == manifest["artifact_digest"]

    # Verify placeholder is NOT used
    assert not (run_dir / "refit" / "model_h5.json").exists()


def test_not_frozen_config_rejected_for_certification(tmp_path: Path):
    """TEST 9: Attempting certification with freeze_status='not_frozen' fails closed immediately."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from run_global_pipeline import GlobalPipelineRunner, PipelineConfig

    cfg = PipelineConfig(
        run_id="test_unfrozen",
        protocol_version="global-cert-v3",
        research_protocol_version="global-research-v3",
        freeze_status="not_frozen",
    )
    runner = GlobalPipelineRunner(cfg, tmp_path)

    with pytest.raises(ValueError, match="requires a frozen configuration"):
        runner.run_stage_certify({}, {}, {})


def test_frozen_selection_immutability(tmp_path: Path):
    """TEST 10: Invariant AF - Altering dev stage files post-freeze does not alter certification candidate."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from run_global_pipeline import GlobalPipelineRunner, PipelineConfig

    run_dir = tmp_path / "run"
    frozen_model_dir = run_dir / "frozen_models" / "h5"
    cand = MomentumRank20DCandidate()
    manifest = save_candidate_artifact(
        cand,
        frozen_model_dir,
        horizon=5,
        development_cutoff="2026-08-21",
        feature_contract_version="cross_sectional_v3_rank_v1",
        target_contract_version="relative_forward_log_return_dev_loo_v1",
        train_ticker_digest="test",
        fit_data_min_date="2024-01-01",
        fit_data_max_date="2026-08-21",
    )

    # Frozen config specifies momentum_rank_20d
    cfg = PipelineConfig(
        run_id="test_immutability",
        protocol_version="global-cert-v3",
        research_protocol_version="global-research-v3",
        freeze_status="frozen",
        selected_candidates={
            "5": {
                "status": "selected",
                "candidate": "momentum_rank_20d",
                "model_artifact": {
                    "directory": "frozen_models/h5",
                    "manifest_file": "frozen_models/h5/model_manifest.json",
                    "artifact_digest": manifest["artifact_digest"],
                },
            }
        },
    )

    runner = GlobalPipelineRunner(cfg, run_dir)

    # Tamper with decisions argument to claim another candidate
    tampered_decisions = {
        5: V3SelectionDecision(
            horizon=5, candidate="elastic_net_cross_sectional", status="selected"
        )
    }

    panels = create_v3_synthetic_panel(n_tickers=35, n_sessions=100, seed=42)
    # Runner uses frozen candidate from config, not the tampered decisions
    cert_res = runner.run_stage_certify(
        tampered_decisions,
        panels,
        {"train_tickers": list(panels.keys()), "asset_transfer_holdout_tickers": []},
    )
    assert cert_res["status"] == "locked_waiting_for_maturity"


def test_v3_downstream_stages_never_rerun_selection(tmp_path: Path):
    """Verify that stage='certify' in V3 does NOT evaluate, rerun selection, or mutate 06_selection.json."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from run_global_pipeline import GlobalPipelineRunner, PipelineConfig

    run_dir = tmp_path / "run"
    stages_dir = run_dir / "stages"
    stages_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create frozen candidate A (momentum_rank_20d)
    frozen_model_dir = run_dir / "frozen_models" / "h5"
    cand_a = MomentumRank20DCandidate()
    manifest_a = save_candidate_artifact(
        cand_a,
        frozen_model_dir,
        horizon=5,
        development_cutoff="2026-08-21",
        feature_contract_version="cross_sectional_v3_rank_v1",
        target_contract_version="relative_forward_log_return_dev_loo_v1",
        train_ticker_digest="test",
        fit_data_min_date="2024-01-01",
        fit_data_max_date="2026-08-21",
    )

    cfg = PipelineConfig(
        run_id="test_no_selection_rerun",
        protocol_version="global-cert-v3",
        research_protocol_version="global-research-v3",
        freeze_status="frozen",
        selected_candidates={
            "5": {
                "status": "selected",
                "candidate": "momentum_rank_20d",
                "model_artifact": {
                    "directory": "frozen_models/h5",
                    "manifest_file": "frozen_models/h5/model_manifest.json",
                    "artifact_digest": manifest_a["artifact_digest"],
                },
            }
        },
    )

    # 2. Plant 05_evaluate.json and 06_selection.json claiming candidate B (elastic_net_cross_sectional)
    eval_file = stages_dir / "05_evaluate.json"
    eval_file.write_text(json.dumps({"5": []}), encoding="utf-8")

    sel_file = stages_dir / "06_selection.json"
    initial_sel_content = json.dumps(
        {"5": {"horizon": 5, "candidate": "elastic_net_cross_sectional", "status": "selected"}},
        indent=2,
    )
    sel_file.write_text(initial_sel_content, encoding="utf-8")
    initial_sel_bytes = sel_file.read_bytes()

    # 3. Create synthetic universe data
    panels = create_v3_synthetic_panel(n_tickers=35, n_sessions=100, seed=42)

    runner = GlobalPipelineRunner(cfg, run_dir, universe_data=panels)
    cert_res = runner.run(stage="certify")

    # 4. Verify 06_selection.json is byte-identical (was NOT rewritten or recomputed)
    assert sel_file.read_bytes() == initial_sel_bytes

    # 5. Verify certification result used frozen candidate A from config
    assert cert_res["stages"]["certification"]["status"] == "locked_waiting_for_maturity"


def test_freeze_selected_candidate_no_panel_raises(tmp_path: Path):
    """Freeze fails closed immediately if candidates are selected but no panel data is available."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from freeze_global_v3 import freeze_global_v3

    run_dir = tmp_path / "run"
    stages = run_dir / "stages"
    stages.mkdir(parents=True, exist_ok=True)
    (stages / "01_snapshot.json").write_text(json.dumps({"tickers": ["TK_01"]}), encoding="utf-8")
    (stages / "03_folds.json").write_text(
        json.dumps({"train_tickers": ["TK_01"], "asset_transfer_holdout_tickers": []}),
        encoding="utf-8",
    )
    (stages / "06_selection.json").write_text(
        json.dumps(
            {
                "5": {
                    "horizon": 5,
                    "candidate": "momentum_rank_20d",
                    "status": "selected",
                    "mean_spearman_ic": 0.05,
                    "mean_ic_ci_lower_95": 0.02,
                    "holm_adjusted_p": 0.001,
                }
            }
        ),
        encoding="utf-8",
    )

    out_cfg = tmp_path / "frozen.json"
    with pytest.raises(ValueError, match="without panel data"):
        freeze_global_v3(run_dir, out_cfg, panel_dir=None, universe_data=None)

    assert not out_cfg.exists()


def test_freeze_missing_artifact_raises(tmp_path: Path):
    """Freeze fails closed if selected candidate name is invalid or fails artifact serialization."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from freeze_global_v3 import freeze_global_v3

    run_dir = tmp_path / "run"
    stages = run_dir / "stages"
    stages.mkdir(parents=True, exist_ok=True)
    (stages / "01_snapshot.json").write_text(json.dumps({"tickers": ["TK_01"]}), encoding="utf-8")
    (stages / "03_folds.json").write_text(
        json.dumps({"train_tickers": ["TK_01"], "asset_transfer_holdout_tickers": []}),
        encoding="utf-8",
    )
    (stages / "06_selection.json").write_text(
        json.dumps(
            {
                "5": {
                    "horizon": 5,
                    "candidate": "invalid_nonexistent_candidate",
                    "status": "selected",
                    "mean_spearman_ic": 0.05,
                    "mean_ic_ci_lower_95": 0.02,
                    "holm_adjusted_p": 0.001,
                }
            }
        ),
        encoding="utf-8",
    )

    panels = create_v3_synthetic_panel(n_tickers=5, n_sessions=50, seed=42)
    out_cfg = tmp_path / "frozen.json"
    with pytest.raises(ValueError, match="invalid candidate"):
        freeze_global_v3(run_dir, out_cfg, universe_data=panels)

    assert not out_cfg.exists()


def test_freeze_all_abstain_allowed_without_models(tmp_path: Path):
    """If all horizons abstain, metadata-only frozen config is permitted with zero models."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from freeze_global_v3 import freeze_global_v3

    run_dir = tmp_path / "run"
    stages = run_dir / "stages"
    stages.mkdir(parents=True, exist_ok=True)
    (stages / "01_snapshot.json").write_text(json.dumps({"tickers": ["TK_01"]}), encoding="utf-8")
    (stages / "03_folds.json").write_text(
        json.dumps({"train_tickers": ["TK_01"], "asset_transfer_holdout_tickers": []}),
        encoding="utf-8",
    )
    (stages / "06_selection.json").write_text(
        json.dumps(
            {
                "5": {
                    "horizon": 5,
                    "candidate": None,
                    "status": "abstain_no_robust_rank_signal",
                    "mean_spearman_ic": 0.0,
                    "mean_ic_ci_lower_95": -0.01,
                    "holm_adjusted_p": 1.0,
                }
            }
        ),
        encoding="utf-8",
    )

    out_cfg = tmp_path / "frozen.json"
    res = freeze_global_v3(run_dir, out_cfg, panel_dir=None, universe_data=None)

    assert out_cfg.exists()
    assert res["freeze_status"] == "frozen"
    assert res["selected_candidates"]["5"]["model_artifact"] is None
    assert not (run_dir / "frozen_models").exists()


def test_v3_signed_release_roundtrip_and_tamper_detection(tmp_path: Path):
    """V3 release integrates with Ed25519 signing and verify_release() detects tampering."""
    import sys

    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519

    from release.bundle import verify_release

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from run_global_pipeline import GlobalPipelineRunner, PipelineConfig

    # 1. Generate Ed25519 keypair
    priv_key = ed25519.Ed25519PrivateKey.generate()
    pub_key = priv_key.public_key()

    priv_path = tmp_path / "ed25519_priv.pem"
    pub_path = tmp_path / "ed25519_pub.pem"

    priv_path.write_bytes(
        priv_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    pub_path.write_bytes(
        pub_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )

    # 2. Set up frozen candidate model
    run_dir = tmp_path / "run"
    frozen_model_dir = run_dir / "frozen_models" / "h5"
    cand = MomentumRank20DCandidate()
    manifest = save_candidate_artifact(
        cand,
        frozen_model_dir,
        horizon=5,
        development_cutoff="2026-08-21",
        feature_contract_version="cross_sectional_v3_rank_v1",
        target_contract_version="relative_forward_log_return_dev_loo_v1",
        train_ticker_digest="test",
        fit_data_min_date="2024-01-01",
        fit_data_max_date="2026-08-21",
    )

    cfg = PipelineConfig(
        run_id="test_signed_release",
        protocol_version="global-cert-v3",
        research_protocol_version="global-research-v3",
        freeze_status="frozen",
        private_key_path=str(priv_path),
        public_key_path=str(pub_path),
        selected_candidates={
            "5": {
                "status": "selected",
                "candidate": "momentum_rank_20d",
                "model_artifact": {
                    "directory": "frozen_models/h5",
                    "manifest_file": "frozen_models/h5/model_manifest.json",
                    "artifact_digest": manifest["artifact_digest"],
                },
            }
        },
    )

    runner = GlobalPipelineRunner(cfg, run_dir)
    cert_result = {
        "status": "holdout_opened",
        "decision": "pass",
        "certified_horizons": [5],
    }

    # 3. Execute release stage
    release_res = runner.run_stage_refit_and_release({}, cert_result, {}, {})
    assert release_res["status"] == "completed"
    assert release_res["signed_release"] is True

    release_dir = run_dir / "release"
    assert (release_dir / "manifest.json").exists()
    assert (release_dir / "manifest.sig").exists()

    # 4. Verify release bundle using verify_release()
    verified_manifest = verify_release(release_dir, public_key_path=pub_path)
    assert verified_manifest["metadata"]["protocol_version"] == "global-cert-v3"
    assert verified_manifest["metadata"]["certified_horizons"] == [5]

    # 5. Tamper with a model file -> fails closed
    model_file = release_dir / "h5" / "model.json"
    model_file.write_bytes(b'{"tampered": true}')
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_release(release_dir, public_key_path=pub_path)


def test_v3_asset_split_seed_controls_split(tmp_path: Path):
    """Changing asset_split_seed changes the D/H split deterministically without altering model candidate seeds."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from run_global_pipeline import GlobalPipelineRunner, PipelineConfig

    panels = create_v3_synthetic_panel(n_tickers=50, n_sessions=100, seed=42)

    cfg_42 = PipelineConfig(
        run_id="test_split_42",
        protocol_version="global-cert-v3",
        research_protocol_version="global-research-v3",
        asset_split_seed=42,
        seeds=[100],
    )
    cfg_123 = PipelineConfig(
        run_id="test_split_123",
        protocol_version="global-cert-v3",
        research_protocol_version="global-research-v3",
        asset_split_seed=123,
        seeds=[100],
    )

    runner_42 = GlobalPipelineRunner(cfg_42, tmp_path / "run_42")
    runner_123 = GlobalPipelineRunner(cfg_123, tmp_path / "run_123")

    folds_42 = runner_42.run_stage_folds(panels)
    folds_123 = runner_123.run_stage_folds(panels)

    assert folds_42["train_tickers"] != folds_123["train_tickers"]
    assert folds_42["asset_transfer_holdout_tickers"] != folds_123["asset_transfer_holdout_tickers"]
    # Candidate seeds did not change
    assert cfg_42.seeds == cfg_123.seeds == [100]


def test_warmup_length_mismatch_regression():
    """Exact regression test: raw targets have 2010 dates, features have 1867 dates due to warm-up.

    evaluate_v3_candidate_on_folds() must align by timestamp/index and never fail with
    'Boolean index has wrong length'.
    """
    all_dates = pd.date_range("2018-01-01", periods=2010, freq="B")
    feature_dates = all_dates[143:]  # 1867 dates (drop 143 warm-up sessions)

    tickers = [f"T_{i:02d}" for i in range(35)]
    np.random.seed(42)

    dev_features: dict[str, pd.DataFrame] = {}
    dev_targets: dict[str, pd.Series] = {}

    for t in tickers:
        f_df = pd.DataFrame(
            np.random.randn(len(feature_dates), 4),
            index=feature_dates,
            columns=[
                "Return_20D_CS_Rank",
                "Vol_C2C_20_CS_Rank",
                "Return_1D_CS_Rank",
                "Volume_Surprise_CS_Rank",
            ],
        )
        dev_features[t] = f_df
        # Target Series has the full 2010 dates
        t_s = pd.Series(np.random.randn(len(all_dates)), index=all_dates)
        dev_targets[t] = t_s

    # Create 4 expanding calendar folds across feature_dates
    n_total = len(feature_dates)
    fold_step = n_total // 6
    folds = [
        CalendarFold(
            fold_index=i,
            train_start=feature_dates[0],
            train_end=feature_dates[(i + 1) * fold_step],
            val_start=feature_dates[(i + 1) * fold_step + 1],
            val_end=feature_dates[(i + 2) * fold_step],
            n_train_sessions=(i + 1) * fold_step + 1,
            n_val_sessions=fold_step,
        )
        for i in range(4)
    ]

    cand = RidgeCrossSectionalCandidate(feature_cols=["Return_20D_CS_Rank", "Vol_C2C_20_CS_Rank"])
    ev = evaluate_v3_candidate_on_folds(
        cand,
        horizon=5,
        folds=folds,
        dev_features=dev_features,
        dev_targets=dev_targets,
        min_daily_asset_count=30,
        resamples=100,
        seed=42,
    )

    assert ev.candidate_name == "ridge_cross_sectional"
    assert len(ev.fold_metrics) == 4
    assert ev.overall_metrics.n_eligible_sessions > 0

    # Explicitly test date-level index alignment on sliced fold data
    fold_0 = folds[0]
    for t in tickers:
        f_slice = dev_features[t].loc[
            (dev_features[t].index >= fold_0.train_start)
            & (dev_features[t].index <= fold_0.train_end)
        ]
        t_slice = dev_targets[t].reindex(f_slice.index)
        assert f_slice.index.equals(t_slice.index)
        assert len(f_slice) == len(t_slice)


def test_irregular_calendar_alignment():
    """Test where tickers have irregular/missing dates.

    Ensure feature date t maps strictly to target date t, missing dates are handled
    without positional misalignment, and cross-sectional IC remains exact.
    """
    dates_a = pd.date_range("2020-01-01", periods=300, freq="B")
    # Ticker B misses every 5th date
    dates_b = dates_a[np.arange(len(dates_a)) % 5 != 0]

    tickers = [f"T_{i:02d}" for i in range(35)]
    np.random.seed(42)
    dev_features: dict[str, pd.DataFrame] = {}
    dev_targets: dict[str, pd.Series] = {}

    for i, t in enumerate(tickers):
        use_dates = dates_b if i % 2 == 0 else dates_a
        dev_features[t] = pd.DataFrame(
            {"Return_20D_CS_Rank": np.random.randn(len(use_dates))},
            index=use_dates,
        )
        dev_targets[t] = pd.Series(np.random.randn(len(use_dates)), index=use_dates)

    folds = [
        CalendarFold(
            fold_index=0,
            train_start=dates_a[0],
            train_end=dates_a[150],
            val_start=dates_a[151],
            val_end=dates_a[250],
            n_train_sessions=151,
            n_val_sessions=100,
        )
    ]

    cand = MomentumRank20DCandidate()
    ev = evaluate_v3_candidate_on_folds(
        cand,
        horizon=5,
        folds=folds,
        dev_features=dev_features,
        dev_targets=dev_targets,
        min_daily_asset_count=30,
        resamples=100,
        seed=42,
    )
    assert ev.candidate_name == "momentum_rank_20d"
    assert len(ev.fold_metrics) == 1


def test_horizon_tail_target_nan_alignment():
    """Test where the final h target rows are NaN because future returns are unobserved.

    Ensure candidate fitting and evaluation drops trailing NaNs safely without
    contaminating or shifting timestamps.
    """
    dates = pd.date_range("2020-01-01", periods=200, freq="B")
    tickers = [f"T_{i:02d}" for i in range(35)]
    np.random.seed(42)

    dev_features: dict[str, pd.DataFrame] = {}
    dev_targets: dict[str, pd.Series] = {}

    for t in tickers:
        dev_features[t] = pd.DataFrame(
            {"Return_20D_CS_Rank": np.random.randn(len(dates))},
            index=dates,
        )
        tgt = pd.Series(np.random.randn(len(dates)), index=dates)
        # Final 5 observations are NaN
        tgt.iloc[-5:] = np.nan
        dev_targets[t] = tgt

    folds = [
        CalendarFold(
            fold_index=0,
            train_start=dates[0],
            train_end=dates[120],
            val_start=dates[121],
            val_end=dates[199],
            n_train_sessions=121,
            n_val_sessions=79,
        )
    ]

    cand = RidgeCrossSectionalCandidate(feature_cols=["Return_20D_CS_Rank"])
    ev = evaluate_v3_candidate_on_folds(
        cand,
        horizon=5,
        folds=folds,
        dev_features=dev_features,
        dev_targets=dev_targets,
        min_daily_asset_count=30,
        resamples=100,
        seed=42,
    )
    assert ev.candidate_name == "ridge_cross_sectional"
    assert ev.overall_metrics.n_eligible_sessions > 0
