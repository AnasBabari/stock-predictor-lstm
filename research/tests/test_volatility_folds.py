from __future__ import annotations

import numpy as np
import pytest
from volatility_forecasting.contracts import VolatilityForecastProtocol
from volatility_forecasting.data import VolatilityPanelExamples
from volatility_forecasting.folds import (
    build_inner_training_split,
    build_prospective_development_fold_plan,
    build_volatility_fold_plan,
    select_asset_holdouts,
)


def _examples(sessions: int = 90) -> VolatilityPanelExamples:
    tickers = ("AAA", "BBB", "CCC", "DDD", "NMM", "MSFT")
    dates = np.arange(
        np.datetime64("2025-01-02"),
        np.datetime64("2025-01-02") + np.timedelta64(sessions, "D"),
    )
    ticker_rows = np.repeat(tickers, sessions)
    origin_dates = np.tile(dates, len(tickers))
    rows = len(ticker_rows)
    horizons = (1, 3, 7)
    return VolatilityPanelExamples(
        features=np.ones((rows, 22, 2), dtype=np.float32),
        baseline_variance=np.ones((rows, 3), dtype=np.float32) * 0.01,
        realized_variance=np.ones((rows, 3), dtype=np.float32) * 0.02,
        cumulative_returns=np.zeros((rows, 3), dtype=np.float32),
        direction_classes=np.ones((rows, 3), dtype=np.int64),
        tickers=ticker_rows,
        origin_dates=origin_dates,
        origin_closes=np.ones(rows, dtype=np.float64) * 100,
        horizons=horizons,
        feature_names=("a", "b"),
    )


def _protocol() -> VolatilityForecastProtocol:
    return VolatilityForecastProtocol(
        horizons=(1, 3, 7),
        feature_names=("a", "b"),
        window_size=22,
        folds=3,
        embargo_sessions=7,
        minimum_train_sessions=20,
        validation_sessions=8,
        early_stopping_sessions=5,
        temporal_holdout_sessions=10,
        asset_holdout_fraction=0.25,
    )


def test_asset_split_is_deterministic_and_reserves_acceptance_tickers() -> None:
    tickers = np.array(["AAA", "BBB", "CCC", "DDD", "NMM", "MSFT"])
    left = select_asset_holdouts(tickers, fraction=0.25, seed=11)
    right = select_asset_holdouts(tickers, fraction=0.25, seed=11)
    assert left == right
    assert {"NMM", "MSFT"}.issubset(left[1])
    assert set(left[0]).isdisjoint(left[1])


def test_fold_plan_has_expanding_training_and_embargo_gap() -> None:
    examples = _examples()
    protocol = _protocol()
    plan = build_volatility_fold_plan(examples, protocol, asset_split_seed=8)
    assert len(plan.folds) == 3
    assert {"NMM", "MSFT"}.issubset(plan.asset_holdout_tickers)

    prior_train_rows = 0
    unique_dates = np.unique(examples.origin_dates)
    for fold in plan.folds:
        assert len(fold.train_indices) > prior_train_rows
        prior_train_rows = len(fold.train_indices)
        train_end_position = int(np.flatnonzero(unique_dates == fold.train_end)[0])
        validation_start_position = int(np.flatnonzero(unique_dates == fold.validation_start)[0])
        assert validation_start_position - train_end_position - 1 == protocol.embargo_sessions
        assert set(examples.tickers[fold.train_indices]).isdisjoint(plan.asset_holdout_tickers)
        assert set(examples.tickers[fold.validation_indices]).isdisjoint(plan.asset_holdout_tickers)


def test_certification_rows_are_not_in_development_folds() -> None:
    examples = _examples()
    plan = build_volatility_fold_plan(examples, _protocol())
    development_rows = np.concatenate(
        [np.concatenate((fold.train_indices, fold.validation_indices)) for fold in plan.folds]
    )
    certification_rows = np.concatenate(
        (plan.temporal_certification_indices, plan.asset_transfer_certification_indices)
    )
    assert np.intersect1d(development_rows, certification_rows).size == 0
    assert set(examples.tickers[plan.temporal_certification_indices]).issubset(plan.train_tickers)
    assert set(examples.tickers[plan.asset_transfer_certification_indices]).issubset(
        plan.asset_holdout_tickers
    )


def test_prospective_plan_uses_all_historical_sessions_and_keeps_future_empty() -> None:
    examples = _examples()
    protocol = _protocol()
    cutoff = examples.origin_dates.max()
    prospective_start = cutoff + np.timedelta64(5, "D")
    plan = build_prospective_development_fold_plan(
        examples,
        protocol,
        development_cutoff=cutoff,
        prospective_certification_start=prospective_start,
    )

    assert plan.folds[-1].validation_end == cutoff
    assert plan.certification_start == prospective_start
    assert len(plan.temporal_certification_indices) == 0
    assert len(plan.asset_transfer_certification_indices) == 0
    assert {"NMM", "MSFT"}.issubset(plan.asset_holdout_tickers)


def test_prospective_plan_rejects_present_or_pre_cutoff_certification_rows() -> None:
    examples = _examples()
    protocol = _protocol()
    cutoff = examples.origin_dates.max()
    with pytest.raises(ValueError, match="must start after"):
        build_prospective_development_fold_plan(
            examples,
            protocol,
            development_cutoff=cutoff,
            prospective_certification_start=cutoff,
        )
    with pytest.raises(ValueError, match="already present"):
        build_prospective_development_fold_plan(
            examples,
            protocol,
            development_cutoff=cutoff - np.timedelta64(10, "D"),
            prospective_certification_start=cutoff - np.timedelta64(5, "D"),
        )


def test_inner_early_stopping_split_is_purged_and_never_uses_outer_validation() -> None:
    examples = _examples()
    protocol = _protocol()
    outer = build_volatility_fold_plan(examples, protocol).folds[0]
    inner = build_inner_training_split(examples, outer.train_indices, protocol)
    unique_dates = np.unique(examples.origin_dates[outer.train_indices])
    fit_end = int(np.flatnonzero(unique_dates == inner.fit_end)[0])
    early_start = int(np.flatnonzero(unique_dates == inner.early_stopping_start)[0])
    assert early_start - fit_end - 1 == protocol.embargo_sessions
    assert inner.early_stopping_end <= outer.train_end
    assert inner.early_stopping_end < outer.validation_start
    assert np.intersect1d(inner.early_stopping_indices, outer.validation_indices).size == 0
