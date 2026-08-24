from __future__ import annotations

import numpy as np
from volatility_forecasting.contracts import VolatilityForecastProtocol, VolatilityPromotionGate
from volatility_forecasting.data import VolatilityPanelExamples
from volatility_forecasting.evaluation import (
    assess_promotion,
    cluster_losses_by_session,
    evaluate_tcn_development,
    moving_block_ratio_upper_bound,
)
from volatility_forecasting.folds import build_volatility_fold_plan
from volatility_forecasting.model import BaselineResidualTCNConfig, TorchTrainingConfig


def _metric(
    relative_qlike: float,
    relative_crps: float = 0.95,
    coverage: float = 0.8,
    return_mae: float = 0.95,
    return_rmse: float = 0.95,
):
    return {
        "relative_qlike": relative_qlike,
        "relative_gaussian_crps": 1.25,
        "relative_variance_only_gaussian_crps": relative_crps,
        "variance_only_coverage_80": coverage,
        "relative_return_mae": return_mae,
        "relative_return_rmse": return_rmse,
    }


def test_moving_block_ratio_is_below_one_for_uniformly_better_losses() -> None:
    baseline = np.linspace(0.2, 0.5, 100)
    candidate = baseline * 0.8
    upper = moving_block_ratio_upper_bound(
        candidate,
        baseline,
        resamples=200,
        block_length=7,
        seed=5,
    )
    assert upper < 1.0


def test_loss_clustering_equal_weights_sessions_not_ticker_rows() -> None:
    losses = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0],
        ]
    )
    dates = np.array(
        ["2025-01-02", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05", "2025-01-06"],
        dtype="datetime64[D]",
    )
    clustered, sessions = cluster_losses_by_session(losses, dates)
    np.testing.assert_allclose(clustered[0], [2.0, 3.0])
    np.testing.assert_allclose(clustered[1:], losses[2:])
    assert len(sessions) == 5


def _loss_dates(rows: int) -> np.ndarray:
    return np.datetime64("2020-01-01") + np.arange(rows).astype("timedelta64[D]")


def test_promotion_reports_independent_volatility_and_distribution_verdicts() -> None:
    rng = np.random.default_rng(2)
    baseline = rng.uniform(0.2, 0.5, size=(300, 1))
    candidate = baseline * 0.75
    pooled = (_metric(0.75),)
    folds = tuple((_metric(value),) for value in (0.7, 0.8, 0.75, 0.78, 0.79))
    decision = assess_promotion(
        pooled_metrics=pooled,
        fold_metrics=folds,
        candidate_qlike_losses=candidate,
        baseline_qlike_losses=baseline,
        loss_dates=_loss_dates(len(candidate)),
        horizons=(7,),
        gate=VolatilityPromotionGate(),
        resamples=200,
    )[0]
    assert decision.promoted
    assert decision.volatility_promoted
    assert decision.return_distribution_promoted
    assert decision.return_location_promoted
    assert not decision.direction_promoted
    assert decision.reasons == ()
    assert decision.folds_beating_baseline == 5
    assert decision.holm_significant


def test_promotion_fails_closed_when_coverage_or_fold_is_bad() -> None:
    rng = np.random.default_rng(3)
    baseline = rng.uniform(0.2, 0.5, size=(300, 1))
    candidate = baseline * 0.75
    pooled = (_metric(0.75, coverage=0.95),)
    folds = tuple((_metric(value),) for value in (0.7, 0.8, 0.75, 0.78, 1.2))
    decision = assess_promotion(
        pooled_metrics=pooled,
        fold_metrics=folds,
        candidate_qlike_losses=candidate,
        baseline_qlike_losses=baseline,
        loss_dates=_loss_dates(len(candidate)),
        horizons=(7,),
        resamples=200,
    )[0]
    assert not decision.promoted
    assert any("fold" in reason for reason in decision.reasons)
    assert not decision.return_distribution_promoted
    assert any("coverage" in reason for reason in decision.return_distribution_reasons)


def test_bad_auxiliary_return_head_does_not_veto_promoted_volatility() -> None:
    rng = np.random.default_rng(11)
    baseline = rng.uniform(0.2, 0.5, size=(300, 1))
    candidate = baseline * 0.7
    pooled = (_metric(0.7, return_mae=1.05, return_rmse=1.08),)
    folds = tuple(
        (_metric(value, return_mae=1.03, return_rmse=1.04),)
        for value in (0.72, 0.71, 0.75, 0.76, 0.74)
    )
    decision = assess_promotion(
        pooled_metrics=pooled,
        fold_metrics=folds,
        candidate_qlike_losses=candidate,
        baseline_qlike_losses=baseline,
        loss_dates=_loss_dates(len(candidate)),
        horizons=(7,),
        resamples=200,
    )[0]
    assert decision.volatility_promoted
    assert decision.promoted
    assert not decision.return_location_promoted
    assert any("MAE" in reason for reason in decision.return_location_reasons)
    assert any("RMSE" in reason for reason in decision.return_location_reasons)


def test_bad_interval_crps_does_not_veto_realized_volatility() -> None:
    rng = np.random.default_rng(17)
    baseline = rng.uniform(0.2, 0.5, size=(300, 1))
    candidate = baseline * 0.7
    pooled = (_metric(0.7, relative_crps=1.02),)
    folds = tuple((_metric(value),) for value in (0.72, 0.71, 0.75, 0.76, 0.74))
    decision = assess_promotion(
        pooled_metrics=pooled,
        fold_metrics=folds,
        candidate_qlike_losses=candidate,
        baseline_qlike_losses=baseline,
        loss_dates=_loss_dates(len(candidate)),
        horizons=(7,),
        resamples=200,
    )[0]
    assert decision.volatility_promoted
    assert decision.promoted
    assert not decision.return_distribution_promoted
    assert any("CRPS" in reason for reason in decision.return_distribution_reasons)


def test_tiny_development_run_produces_disjoint_oof_evidence() -> None:
    rng = np.random.default_rng(12)
    tickers = ("AAA", "BBB", "CCC", "DDD", "NMM", "MSFT")
    sessions = 90
    dates = np.arange(
        np.datetime64("2025-01-02"),
        np.datetime64("2025-01-02") + np.timedelta64(sessions, "D"),
    )
    ticker_rows = np.repeat(tickers, sessions)
    origin_dates = np.tile(dates, len(tickers))
    rows = len(ticker_rows)
    horizons = (1, 3, 7)
    features = rng.normal(size=(rows, 22, 4)).astype(np.float32)
    baseline = np.exp(rng.normal(-8.0, 0.2, size=(rows, 3))).astype(np.float32)
    realized = baseline * np.exp(0.1 * features[:, -1, :1])
    realized = np.repeat(realized[:, :1], 3, axis=1).astype(np.float32)
    returns = rng.normal(size=(rows, 3)).astype(np.float32) * np.sqrt(realized)
    direction = np.where(returns < -0.001, 0, np.where(returns > 0.001, 2, 1)).astype(np.int64)
    examples = VolatilityPanelExamples(
        features=features,
        baseline_variance=baseline,
        realized_variance=realized,
        cumulative_returns=returns,
        direction_classes=direction,
        tickers=ticker_rows,
        origin_dates=origin_dates,
        origin_closes=np.ones(rows) * 100,
        horizons=horizons,
        feature_names=("a", "b", "c", "d"),
    )
    protocol = VolatilityForecastProtocol(
        horizons=horizons,
        feature_names=examples.feature_names,
        window_size=22,
        folds=3,
        embargo_sessions=7,
        minimum_train_sessions=20,
        validation_sessions=8,
        early_stopping_sessions=5,
        temporal_holdout_sessions=10,
        asset_holdout_fraction=0.25,
    )
    plan = build_volatility_fold_plan(examples, protocol)
    evaluation = evaluate_tcn_development(
        examples,
        plan,
        protocol,
        model_config=BaselineResidualTCNConfig(
            feature_count=4,
            horizon_count=3,
            channels=8,
            dilations=(1,),
            dropout=0.0,
        ),
        training_config=TorchTrainingConfig(
            maximum_epochs=1,
            patience=1,
            batch_size=64,
            use_amp=False,
        ),
        promotion_gate=VolatilityPromotionGate(minimum_folds_beating_baseline=2),
        seed=4,
        device="cpu",
        resamples=100,
    )
    assert len(evaluation.folds) == 3
    assert len(evaluation.pooled_metrics) == 3
    assert len(evaluation.promotion) == 3
    assert len(np.unique(evaluation.oof_indices)) == len(evaluation.oof_indices)
    assert evaluation.predictions.variance.shape == (len(evaluation.oof_indices), 3)
