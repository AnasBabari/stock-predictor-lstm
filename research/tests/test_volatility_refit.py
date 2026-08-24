from __future__ import annotations

import numpy as np
import pytest
from volatility_forecasting.contracts import VolatilityForecastProtocol
from volatility_forecasting.data import VolatilityPanelExamples
from volatility_forecasting.folds import VolatilityFoldPlan
from volatility_forecasting.refit import certification_development_split, derive_epoch_budget


def test_epoch_budget_is_conservative_median_of_fold_choices() -> None:
    record = {"folds": [{"best_epoch": value} for value in (5, 8, 9, 4, 8)]}
    assert derive_epoch_budget(record) == 8
    assert derive_epoch_budget({"folds": [{"best_epoch": value} for value in (3, 9, 4, 8)]}) == 4
    with pytest.raises(ValueError, match="at least three"):
        derive_epoch_budget({"folds": [{"best_epoch": 2}]})
    with pytest.raises(ValueError, match="invalid best epoch"):
        derive_epoch_budget({"folds": [{"best_epoch": 2}, {"best_epoch": 3}, {"best_epoch": 0}]})


def test_certification_split_excludes_unseen_assets_and_reserved_dates() -> None:
    protocol = VolatilityForecastProtocol(
        horizons=(1,),
        folds=3,
        embargo_sessions=2,
        early_stopping_sessions=5,
        temporal_holdout_sessions=5,
        validation_sessions=5,
        minimum_train_sessions=10,
    )
    dates = np.arange(30, dtype="timedelta64[D]") + np.datetime64("2025-01-01")
    origin_dates = np.repeat(dates, 2)
    tickers = np.tile(np.asarray(("TRAIN", "NMM")), len(dates))
    rows = len(tickers)
    examples = VolatilityPanelExamples(
        features=np.zeros((rows, 60, 1), dtype=np.float32),
        baseline_variance=np.ones((rows, 1), dtype=np.float32),
        realized_variance=np.ones((rows, 1), dtype=np.float32),
        cumulative_returns=np.zeros((rows, 1), dtype=np.float32),
        direction_classes=np.ones((rows, 1), dtype=np.int64),
        tickers=tickers,
        origin_dates=origin_dates,
        origin_closes=np.full(rows, 100.0),
        horizons=(1,),
        feature_names=("x",),
    )
    certification_start = dates[-5]
    plan = VolatilityFoldPlan(
        folds=(),
        train_tickers=("TRAIN",),
        asset_holdout_tickers=("NMM",),
        temporal_certification_indices=np.flatnonzero(
            (tickers == "TRAIN") & (origin_dates >= certification_start)
        ),
        asset_transfer_certification_indices=np.flatnonzero(
            (tickers == "NMM") & (origin_dates >= certification_start)
        ),
        certification_start=certification_start,
    )
    split = certification_development_split(examples, plan, protocol)
    selected = np.concatenate((split.fit_indices, split.early_stopping_indices))
    assert set(examples.tickers[selected]) == {"TRAIN"}
    assert np.all(examples.origin_dates[selected] < certification_start)
    assert split.fit_end < split.early_stopping_start
