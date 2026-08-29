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
    negative_controls: tuple[V8NewsNegativeControlEvidence, ...]


@dataclass(frozen=True)
class V8NewsNegativeControlEvidence:
    name: str
    transformation: str
    horizons: tuple[NewsHorizonAblationDecision, ...]

    @property
    def rejected(self) -> bool:
        """True when real news beats this control at every required horizon."""
        return all(decision.promoted for decision in self.horizons)


def build_v8_news_negative_controls(
    examples: VolatilityPanelExamples,
    news_features: np.ndarray,
    *,
    seed: int = 20260827,
    delay_days: int = 7,
) -> dict[str, np.ndarray]:
    """Build causal controls without moving future information into the past.

    Cross-sectional shuffling permutes complete feature rows only among assets
    sharing an origin timestamp. Delayed news uses the most recent same-asset
    row at or before ``origin - delay_days``. Both retain the exact row shape
    and are safe to pass through the same expanding folds as the real matrix.
    """
    news = np.asarray(news_features, dtype=np.float32)
    if news.ndim != 2 or news.shape[0] != len(examples.features) or news.shape[1] < 1:
        raise ValueError("v8 news controls require a non-empty aligned feature matrix")
    if not np.isfinite(news).all():
        raise ValueError("v8 news control inputs must be finite")
    if delay_days < 1:
        raise ValueError("v8 delayed-news control requires a positive delay")

    rng = np.random.default_rng(seed)
    shuffled = np.empty_like(news)
    dates = np.asarray(examples.origin_dates, dtype="datetime64[ns]")
    tickers = np.asarray(examples.tickers, dtype=str)
    for origin in np.unique(dates):
        rows = np.flatnonzero(dates == origin)
        shuffled[rows] = news[rng.permutation(rows)]

    delayed = np.zeros_like(news)
    delay = np.timedelta64(delay_days, "D")
    for ticker in np.unique(tickers):
        rows = np.flatnonzero(tickers == ticker)
        order = rows[np.argsort(dates[rows], kind="stable")]
        ticker_dates = dates[order]
        for row in order:
            available = int(np.searchsorted(ticker_dates, dates[row] - delay, side="right")) - 1
            if available >= 0:
                delayed[row] = news[order[available]]

    return {
        "cross_sectionally_shuffled_news": shuffled,
        "news_delayed_seven_days": delayed,
    }


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
    controls = build_v8_news_negative_controls(examples, news)
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
        control_rows: list[V8NewsNegativeControlEvidence] = []
        for control_index, (control_name, control_features) in enumerate(controls.items()):
            control = evaluate_tcn_development(
                examples,
                fold_plan,
                protocol,
                model_config=news_architecture,
                training_config=training_config,
                loss_weights=loss_weights,
                seed=seed,
                device=device,
                resamples=resamples,
                news_features=control_features,
            )
            if not np.array_equal(fused.oof_indices, control.oof_indices):
                raise RuntimeError("news negative control produced different OOF identities")
            control_fold_relative = np.asarray(
                [[float(row["relative_qlike"]) for row in fold.metrics] for fold in control.folds],
                dtype=np.float64,
            )
            control_decisions = assess_news_ablation(
                candidate_qlike_losses=qlike_losses(
                    fused.predictions.variance,
                    examples.realized_variance[indices],
                ),
                market_qlike_losses=qlike_losses(
                    control.predictions.variance,
                    examples.realized_variance[indices],
                ),
                origin_dates=examples.origin_dates[indices],
                candidate_fold_relative_qlike=fused_fold_relative,
                market_fold_relative_qlike=control_fold_relative,
                candidate_promoted_vs_har=tuple(
                    decision.volatility_promoted for decision in fused.promotion
                ),
                horizons=protocol.horizons,
                resamples=resamples,
                seed=20260827 + seed + (control_index + 1) * 1000,
            )
            control_rows.append(
                V8NewsNegativeControlEvidence(
                    name=control_name,
                    transformation=(
                        "permute complete rows among assets at the same origin timestamp"
                        if control_name == "cross_sectionally_shuffled_news"
                        else "use latest same-asset news row available at least seven days earlier"
                    ),
                    horizons=control_decisions,
                )
            )
        evidence.append(
            V8NewsSeedAblationEvidence(
                seed=seed,
                promoted=all(decision.promoted for decision in decisions)
                and all(control.rejected for control in control_rows),
                horizons=decisions,
                negative_controls=tuple(control_rows),
            )
        )
    return tuple(evidence)
