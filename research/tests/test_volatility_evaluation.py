from __future__ import annotations

import numpy as np
from volatility_forecasting.contracts import VolatilityForecastProtocol, VolatilityPromotionGate
from volatility_forecasting.data import VolatilityPanelExamples
from volatility_forecasting.evaluation import (
    assess_promotion,
    evaluate_tcn_development,
    moving_block_ratio_upper_bound,
)
from volatility_forecasting.folds import build_volatility_fold_plan
from volatility_forecasting.model import BaselineResidualTCNConfig, TorchTrainingConfig


def _metric(relative_qlike: float, relative_crps: float = 0.95, coverage: float = 0.8):
    return {
        "relative_qlike": relative_qlike,
        "relative_gaussian_crps": relative_crps,
        "coverage_80": coverage,
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


def test_promotion_requires_every_probabilistic_and_stability_gate() -> None:
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
        horizons=(7,),
        gate=VolatilityPromotionGate(),
        resamples=200,
    )[0]
    assert decision.promoted
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
        horizons=(7,),
        resamples=200,
    )[0]
    assert not decision.promoted
    assert any("coverage" in reason for reason in decision.reasons)
    assert any("fold" in reason for reason in decision.reasons)


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
