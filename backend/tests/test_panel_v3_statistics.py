"""Unit tests for Protocol V3 session IC, Newey-West HAC, bootstrap, and Holm correction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from panel.v3_metrics import (
    compute_ic_hac,
    compute_ic_moving_block_bootstrap,
    compute_session_rank_ic,
    holm_bonferroni_family,
)


def test_session_rank_ic_perfect_and_reversed():
    dates = pd.bdate_range("2024-01-01", periods=10)
    tickers = [f"T{i:02d}" for i in range(35)]

    # Scores: 0..34 across tickers
    scores_mat = np.tile(np.arange(35), (10, 1))
    scores_df = pd.DataFrame(scores_mat, index=dates, columns=tickers)

    # Targets: identical to scores -> IC should be +1.0
    targets_df = scores_df.copy()
    daily_ic, breadth = compute_session_rank_ic(scores_df, targets_df, min_daily_asset_count=30)
    assert len(daily_ic.dropna()) == 10
    np.testing.assert_allclose(daily_ic.values, 1.0)
    np.testing.assert_array_equal(breadth.values, 35)

    # Targets: reversed -> IC should be -1.0
    rev_targets_df = pd.DataFrame(34 - scores_mat, index=dates, columns=tickers)
    daily_ic_rev, _ = compute_session_rank_ic(scores_df, rev_targets_df, min_daily_asset_count=30)
    np.testing.assert_allclose(daily_ic_rev.values, -1.0)


def test_session_rank_ic_constant_and_low_breadth():
    dates = pd.bdate_range("2024-01-01", periods=5)
    tickers = [f"T{i:02d}" for i in range(35)]

    # Constant prediction score (zeros) -> must be NaN, NOT 0.0
    const_scores = pd.DataFrame(0.0, index=dates, columns=tickers)
    targets_df = pd.DataFrame(np.random.randn(5, 35), index=dates, columns=tickers)

    daily_ic, _ = compute_session_rank_ic(const_scores, targets_df, min_daily_asset_count=30)
    assert daily_ic.isna().all(), "Constant prediction must yield NaN daily IC, not 0.0"

    # Breadth < 30 (only 10 tickers) -> must be NaN
    sub_tickers = tickers[:10]
    sub_scores = pd.DataFrame(np.random.randn(5, 10), index=dates, columns=sub_tickers)
    sub_targets = pd.DataFrame(np.random.randn(5, 10), index=dates, columns=sub_tickers)
    daily_ic_low_breadth, _ = compute_session_rank_ic(
        sub_scores, sub_targets, min_daily_asset_count=30
    )
    assert daily_ic_low_breadth.isna().all()


def test_hac_lag_policy_and_significance():
    rng = np.random.default_rng(42)
    # Simulated strong positive IC series
    positive_ic = rng.normal(0.05, 0.02, 200)

    se_1d, t_1d, p_1d = compute_ic_hac(positive_ic, horizon=1)
    # 1d has lag 0
    assert t_1d > 10.0
    assert p_1d < 1e-6

    # 30d has lag 29
    se_30d, t_30d, p_30d = compute_ic_hac(positive_ic, horizon=30)
    assert se_30d > 0.0
    assert t_30d > 0.0
    assert p_30d < 0.01

    # Zero/negative mean IC should give large one-sided p-value
    noise_ic = rng.normal(0.0, 0.05, 200)
    _, _, p_noise = compute_ic_hac(noise_ic, horizon=5)
    assert p_noise > 0.10


def test_moving_block_bootstrap_confidence_interval():
    rng = np.random.default_rng(42)
    positive_ic = rng.normal(0.04, 0.02, 200)

    lo, hi = compute_ic_moving_block_bootstrap(positive_ic, horizon=5, resamples=1000, seed=42)
    assert lo > 0.0, f"Lower bound of strong positive IC must be > 0, got {lo}"
    assert hi > lo
    assert lo < 0.04 < hi

    # Noise series crossing zero
    noise_ic = rng.normal(0.001, 0.04, 200)
    lo_noise, hi_noise = compute_ic_moving_block_bootstrap(
        noise_ic, horizon=5, resamples=1000, seed=42
    )
    assert lo_noise < 0.0, f"Noise series lower bound must cross zero, got {lo_noise}"


def test_holm_bonferroni_family_indexing():
    """Verify Holm correction operates across full (horizon, candidate) family."""
    p_values: dict[tuple[int, str], float] = {
        (1, "cand_A"): 0.001,
        (1, "cand_B"): 0.020,
        (5, "cand_A"): 0.005,
        (5, "cand_B"): 0.150,
        (30, "cand_A"): 0.040,
        (30, "cand_B"): 0.800,
    }

    results = holm_bonferroni_family(p_values, alpha=0.05)
    assert len(results) == 6

    # (1, "cand_A") has lowest raw p (0.001), adjusted by m=6 -> 0.006 <= 0.05 -> reject True
    reject_top, adj_p_top = results[(1, "cand_A")]
    assert reject_top is True
    assert np.isclose(adj_p_top, 0.006)

    # (30, "cand_B") should fail rejection
    reject_weak, _ = results[(30, "cand_B")]
    assert reject_weak is False


def test_session_rank_ic_constant_target():
    """Invariant I: Constant target values must yield NaN daily IC, not 0.0."""
    dates = pd.bdate_range("2024-01-01", periods=5)
    tickers = [f"T{i:02d}" for i in range(35)]
    scores = pd.DataFrame(np.random.randn(5, 35), index=dates, columns=tickers)
    const_targets = pd.DataFrame(0.05, index=dates, columns=tickers)

    daily_ic, _ = compute_session_rank_ic(scores, const_targets, min_daily_asset_count=30)
    assert daily_ic.isna().all(), "Constant target must yield NaN daily IC, not 0.0"


def test_daily_vs_pooled_ic():
    """Invariant L: Session rank IC evaluates cross-sectionally per origin, never pooled across time."""
    dates = pd.bdate_range("2024-01-01", periods=10)
    tickers = [f"T{i:02d}" for i in range(35)]

    # Day 0: perfect positive rank correlation (+1.0)
    # Day 1: perfect negative rank correlation (-1.0)
    # Alternating daily IC (+1, -1, +1, -1) -> mean daily IC = 0.0
    scores_mat = np.zeros((10, 35))
    targets_mat = np.zeros((10, 35))
    for d in range(10):
        scores_mat[d] = np.arange(35)
        targets_mat[d] = np.arange(35) if d % 2 == 0 else (34 - np.arange(35))

    scores_df = pd.DataFrame(scores_mat, index=dates, columns=tickers)
    targets_df = pd.DataFrame(targets_mat, index=dates, columns=tickers)

    daily_ic, _ = compute_session_rank_ic(scores_df, targets_df, min_daily_asset_count=30)
    # Mean of daily ICs must be 0.0
    assert np.isclose(np.nanmean(daily_ic), 0.0)
    # Daily ICs must alternate between +1 and -1
    for d in range(10):
        expected = 1.0 if d % 2 == 0 else -1.0
        assert np.isclose(daily_ic.iloc[d], expected)


def test_moving_block_bootstrap_determinism():
    """Invariant R: Moving-block bootstrap must be 100% bit-for-bit deterministic under fixed seed."""
    rng = np.random.default_rng(999)
    ic_series = rng.normal(0.03, 0.05, 150)

    ci1 = compute_ic_moving_block_bootstrap(ic_series, horizon=5, resamples=500, seed=42)
    ci2 = compute_ic_moving_block_bootstrap(ic_series, horizon=5, resamples=500, seed=42)
    assert ci1 == ci2, "Moving-block bootstrap must be deterministic given seed"
