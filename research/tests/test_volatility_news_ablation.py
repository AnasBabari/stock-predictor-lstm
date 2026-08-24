from __future__ import annotations

import numpy as np
import pytest
from volatility_forecasting.news_ablation import assess_news_ablation


def _dates(sessions: int, assets: int = 2) -> np.ndarray:
    unique = np.datetime64("2020-01-01") + np.arange(sessions).astype("timedelta64[D]")
    return np.repeat(unique, assets)


def test_news_ablation_promotes_only_paired_incremental_improvement() -> None:
    dates = _dates(120)
    rng = np.random.default_rng(12)
    market = rng.uniform(0.2, 0.5, size=(len(dates), 2))
    candidate = market * 0.75
    decisions = assess_news_ablation(
        candidate_qlike_losses=candidate,
        market_qlike_losses=market,
        origin_dates=dates,
        candidate_fold_relative_qlike=np.full((5, 2), 0.60),
        market_fold_relative_qlike=np.full((5, 2), 0.80),
        candidate_promoted_vs_har=(True, True),
        horizons=(1, 7),
        resamples=200,
        seed=9,
    )
    assert all(decision.promoted for decision in decisions)
    assert all(decision.relative_qlike_to_market == pytest.approx(0.75) for decision in decisions)


def test_news_ablation_fails_closed_when_candidate_does_not_beat_har() -> None:
    dates = _dates(120)
    market = np.ones((len(dates), 1))
    candidate = market * 0.75
    decision = assess_news_ablation(
        candidate_qlike_losses=candidate,
        market_qlike_losses=market,
        origin_dates=dates,
        candidate_fold_relative_qlike=np.full((5, 1), 0.75),
        market_fold_relative_qlike=np.ones((5, 1)),
        candidate_promoted_vs_har=(False,),
        horizons=(7,),
        resamples=200,
    )[0]
    assert not decision.promoted
    assert any("HAR" in reason for reason in decision.reasons)


def test_news_ablation_rejects_misaligned_fold_evidence() -> None:
    dates = _dates(20)
    losses = np.ones((len(dates), 1))
    with pytest.raises(ValueError, match="fold evidence"):
        assess_news_ablation(
            candidate_qlike_losses=losses,
            market_qlike_losses=losses,
            origin_dates=dates,
            candidate_fold_relative_qlike=np.ones((4, 1)),
            market_fold_relative_qlike=np.ones((5, 1)),
            candidate_promoted_vs_har=(True,),
            horizons=(7,),
            resamples=100,
        )
