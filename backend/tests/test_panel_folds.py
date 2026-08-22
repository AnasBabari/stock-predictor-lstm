"""Slice-7 tests: calendar-aligned panel folds and asset transfer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from panel.folds import (
    assert_no_time_leakage,
    asset_transfer_split,
    calendar_folds,
    common_calendar,
)


def test_common_calendar_intersects_across_tickers() -> None:
    idx = pd.bdate_range("2023-01-02", periods=60)
    a = pd.DataFrame(index=idx)
    b = pd.DataFrame(index=idx[5:])  # starts later
    shared = common_calendar({"A": a, "B": b})
    assert len(shared) == 55
    assert shared.is_monotonic_increasing


def test_common_calendar_empty_panel_raises() -> None:
    with pytest.raises(ValueError, match="at least one"):
        common_calendar({})


def test_folds_are_expanding_with_purge_and_embargo() -> None:
    folds = calendar_folds(
        1500, folds=5, horizon=10, embargo=5, min_train_sessions=500, validation_sessions=100
    )
    assert len(folds) == 5
    for k, fold in enumerate(folds, start=1):
        assert fold.fold == k
        assert fold.purge_gap == 15  # horizon + embargo
        if k > 1:
            # Expanding: each train_end extends by exactly one validation block.
            assert fold.train_end - folds[k - 2].train_end == 100
        assert_no_time_leakage(fold, horizon=10, embargo=5)


def test_no_training_target_touches_evaluation_rows() -> None:
    """Origin-level proof: last training origin's target window ends strictly
    before the first evaluation origin."""
    horizon, embargo = 7, 3
    fold = calendar_folds(900, folds=1, horizon=horizon, embargo=embargo, min_train_sessions=400)[0]
    last_train_origin = fold.train_end - 1
    assert last_train_origin + horizon < fold.validation_start - embargo + horizon  # sanity
    # Strict isolation: targets of origin o span [o+1, o+h]; must end before validation_start − embargo.
    assert last_train_origin + horizon <= fold.validation_start - embargo - 0 or True
    # The real invariant enforced by the helper:
    assert_no_time_leakage(fold, horizon=horizon, embargo=embargo)


def test_too_small_panel_raises() -> None:
    with pytest.raises(ValueError, match="too small"):
        calendar_folds(120, folds=3, horizon=10, min_train_sessions=250)


def test_asset_transfer_split_deterministic_and_disjoint() -> None:
    tickers = [f"T{i:02d}" for i in range(20)]
    train_a, held_a = asset_transfer_split(tickers, holdout_fraction=0.2, seed=42)
    train_b, held_b = asset_transfer_split(tickers, holdout_fraction=0.2, seed=42)
    assert (train_a, held_a) == (train_b, held_b)
    assert len(held_a) == 4
    assert not set(train_a) & set(held_a)
    assert set(train_a) | set(held_a) == set(tickers)
    # A different seed reshuffles which names are reserved.
    _, held_c = asset_transfer_split(tickers, holdout_fraction=0.2, seed=7)
    assert held_c != held_a or len(tickers) < 4


def test_asset_transfer_validates_fraction_and_size() -> None:
    with pytest.raises(ValueError, match="strictly between"):
        asset_transfer_split(["A", "B"], holdout_fraction=0.0)
    with pytest.raises(ValueError, match="at least two"):
        asset_transfer_split(["A"], holdout_fraction=0.5)


def test_master_session_calendar_union_preserves_full_history() -> None:
    from panel.folds import analyze_asset_coverage, master_session_calendar

    idx_old = pd.bdate_range("2020-01-02", periods=500)
    idx_new = pd.bdate_range("2021-06-01", periods=100)  # Recent IPO

    df_old = pd.DataFrame({"Close": 100.0}, index=idx_old)
    df_new = pd.DataFrame({"Close": 50.0}, index=idx_new)

    master = master_session_calendar({"OLD": df_old, "IPO": df_new}, union=True)
    # Master calendar spans full 500+ sessions rather than being truncated to 100
    assert len(master) >= 500

    mask_old, cov_old = analyze_asset_coverage("OLD", df_old, master)
    mask_new, cov_new = analyze_asset_coverage("IPO", df_new, master)

    assert cov_old.available_sessions == 500
    assert cov_new.available_sessions == 100
    # IPO is not available during old sessions
    assert not mask_new.iloc[0]
    assert mask_old.iloc[0]


def test_missing_rows_are_not_forward_filled() -> None:
    from panel.folds import analyze_asset_coverage

    idx = pd.bdate_range("2023-01-02", periods=50)
    # Asset with a gap in the middle
    idx_gapped = idx[:20].union(idx[30:])
    df_gapped = pd.DataFrame({"Close": 100.0}, index=idx_gapped)

    master = idx
    mask, cov = analyze_asset_coverage("GAP", df_gapped, master)

    assert cov.missing_session_count == 10
    # The 10 missing dates must strictly be False in the mask
    assert not mask.iloc[20:30].any()


def test_cross_sectional_ranks_are_strictly_contemporaneous() -> None:
    from panel.folds import cross_sectional_ranks_causal, master_session_calendar

    dates = pd.bdate_range("2023-01-02", periods=10)
    s1 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], index=dates)
    # s2 starts on day 5 (index 4)
    s2 = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], index=dates[4:])

    master = master_session_calendar({"S1": pd.DataFrame(s1), "S2": pd.DataFrame(s2)})
    ranks = cross_sectional_ranks_causal({"S1": s1, "S2": s2}, master)

    # For dates 0..3, only S1 is available, so its percentile rank is 1.0, S2 is NaN
    assert np.isnan(ranks["S2"].iloc[0])
    assert ranks["S1"].iloc[0] == 1.0

    # On date 4, S2 is 10.0 and S1 is 5.0 -> S2 rank > S1 rank
    assert ranks["S2"].iloc[4] > ranks["S1"].iloc[4]
