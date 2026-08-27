"""Paired expanding-fold evidence for v8 market-plus-news candidates."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .contracts import VolatilityForecastProtocol
from .data import VolatilityPanelExamples
from .evaluation import evaluate_tcn_development
from .folds import VolatilityFoldPlan
from .metrics import qlike_losses
from .model import BaselineResidualTCNConfig, TorchTrainingConfig, VolatilityLossWeights
from .news_ablation import NewsHorizonAblationDecision, assess_news_ablation


@dataclass(frozen=True)
class V8NewsSeedAblationEvidence:
    seed: int
    promoted: bool
    horizons: tuple[NewsHorizonAblationDecision, ...]


def evaluate_v8_news_ablation(
    *,
    examples: VolatilityPanelExamples,
    fold_plan: VolatilityFoldPlan,
    protocol: VolatilityForecastProtocol,
    news_features: np.ndarray,
    seeds: tuple[int, ...],
    market_architecture: BaselineResidualTCNConfig,
    training_config: TorchTrainingConfig,
    loss_weights: VolatilityLossWeights | None = None,
    device: str = "cuda",
    resamples: int = 1000,
) -> tuple[V8NewsSeedAblationEvidence, ...]:
    """Compare matched market/news models without reading sealed test rows."""

    news = np.asarray(news_features, dtype=np.float32)
    if news.ndim != 2 or news.shape[0] != len(examples.features) or news.shape[1] < 1:
        raise ValueError("v8 news ablation requires a non-empty aligned feature matrix")
    if not np.isfinite(news).all():
        raise ValueError("v8 news ablation features must be finite")
    if market_architecture.news_feature_count:
        raise ValueError("paired market architecture must not include news features")
    news_architecture = replace(
        market_architecture,
        news_feature_count=news.shape[1],
    )
    evidence: list[V8NewsSeedAblationEvidence] = []
    for seed in seeds:
        market = evaluate_tcn_development(
            examples,
            fold_plan,
            protocol,
            model_config=market_architecture,
            training_config=training_config,
            loss_weights=loss_weights,
            seed=seed,
            device=device,
            resamples=resamples,
        )
        fused = evaluate_tcn_development(
            examples,
            fold_plan,
            protocol,
            model_config=news_architecture,
            training_config=training_config,
            loss_weights=loss_weights,
            seed=seed,
            device=device,
            resamples=resamples,
            news_features=news,
        )
        if not np.array_equal(market.oof_indices, fused.oof_indices):
            raise RuntimeError("paired news evaluation produced different OOF identities")
        market_fold_relative = np.asarray(
            [[float(row["relative_qlike"]) for row in fold.metrics] for fold in market.folds],
            dtype=np.float64,
        )
        fused_fold_relative = np.asarray(
            [[float(row["relative_qlike"]) for row in fold.metrics] for fold in fused.folds],
            dtype=np.float64,
        )
        indices = fused.oof_indices
        decisions = assess_news_ablation(
            candidate_qlike_losses=qlike_losses(
                fused.predictions.variance,
                examples.realized_variance[indices],
            ),
            market_qlike_losses=qlike_losses(
                market.predictions.variance,
                examples.realized_variance[indices],
            ),
            origin_dates=examples.origin_dates[indices],
            candidate_fold_relative_qlike=fused_fold_relative,
            market_fold_relative_qlike=market_fold_relative,
            candidate_promoted_vs_har=tuple(
                decision.volatility_promoted for decision in fused.promotion
            ),
            horizons=protocol.horizons,
            resamples=resamples,
            seed=20260827 + seed,
        )
        evidence.append(
            V8NewsSeedAblationEvidence(
                seed=seed,
                promoted=all(decision.promoted for decision in decisions),
                horizons=decisions,
            )
        )
    return tuple(evidence)
