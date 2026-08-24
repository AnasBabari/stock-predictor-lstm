"""Paired promotion gate for market-only versus market-plus-news models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.panel.selection import diebold_mariano_hac, holm_correction

from .evaluation import cluster_losses_by_session, moving_block_ratio_upper_bound


@dataclass(frozen=True)
class NewsAblationGate:
    """Conservative initial defaults for incremental news promotion."""

    maximum_relative_qlike_to_market: float = 0.995
    minimum_folds_beating_market: int = 4
    maximum_worst_fold_relative_qlike_to_market: float = 1.05
    significance_level: float = 0.05

    def __post_init__(self) -> None:
        if not 0 < self.maximum_relative_qlike_to_market < 1:
            raise ValueError("news QLIKE gate must require an incremental improvement")
        if self.minimum_folds_beating_market < 1:
            raise ValueError("news fold gate must be positive")
        if self.maximum_worst_fold_relative_qlike_to_market < 1:
            raise ValueError("news worst-fold guardrail cannot be below one")
        if not 0 < self.significance_level < 1:
            raise ValueError("news significance level must be in (0, 1)")


@dataclass(frozen=True)
class NewsHorizonAblationDecision:
    horizon: int
    promoted: bool
    reasons: tuple[str, ...]
    relative_qlike_to_market: float
    relative_qlike_upper_95: float
    folds_beating_market: int
    worst_fold_relative_qlike_to_market: float
    dm_statistic: float
    dm_p_value: float
    holm_significant: bool
    candidate_promoted_vs_har: bool


def assess_news_ablation(
    *,
    candidate_qlike_losses: np.ndarray,
    market_qlike_losses: np.ndarray,
    origin_dates: np.ndarray,
    candidate_fold_relative_qlike: np.ndarray,
    market_fold_relative_qlike: np.ndarray,
    candidate_promoted_vs_har: tuple[bool, ...],
    horizons: tuple[int, ...],
    gate: NewsAblationGate | None = None,
    resamples: int = 1000,
    seed: int = 42,
) -> tuple[NewsHorizonAblationDecision, ...]:
    """Require paired, session-clustered evidence that news adds information."""
    settings = gate or NewsAblationGate()
    candidate = np.asarray(candidate_qlike_losses, dtype=np.float64)
    market = np.asarray(market_qlike_losses, dtype=np.float64)
    candidate_folds = np.asarray(candidate_fold_relative_qlike, dtype=np.float64)
    market_folds = np.asarray(market_fold_relative_qlike, dtype=np.float64)
    expected_columns = len(horizons)
    if candidate.shape != market.shape or candidate.ndim != 2:
        raise ValueError("news and market QLIKE losses must be matched matrices")
    if candidate.shape[1] != expected_columns:
        raise ValueError("news loss horizon count does not match the protocol")
    if (
        candidate_folds.shape != market_folds.shape
        or candidate_folds.ndim != 2
        or candidate_folds.shape[1] != expected_columns
    ):
        raise ValueError("news and market fold evidence must be matched by horizon")
    if len(candidate_promoted_vs_har) != expected_columns:
        raise ValueError("news candidate baseline verdicts do not match the protocol")
    if not np.isfinite(candidate_folds).all() or not np.isfinite(market_folds).all():
        raise ValueError("news fold evidence must be finite")
    if np.any(market_folds <= 0):
        raise ValueError("market fold QLIKE ratios must be positive")

    clustered_candidate, sessions = cluster_losses_by_session(candidate, origin_dates)
    clustered_market, market_sessions = cluster_losses_by_session(market, origin_dates)
    if not np.array_equal(sessions, market_sessions):
        raise ValueError("news and market loss sessions do not match")
    fold_ratios = candidate_folds / market_folds

    dm_rows: list[tuple[float, float]] = []
    upper_bounds: list[float] = []
    for column, horizon in enumerate(horizons):
        dm_rows.append(
            diebold_mariano_hac(
                clustered_candidate[:, column],
                clustered_market[:, column],
                max_lag=max(1, horizon - 1),
            )
        )
        upper_bounds.append(
            moving_block_ratio_upper_bound(
                clustered_candidate[:, column],
                clustered_market[:, column],
                resamples=resamples,
                block_length=max(5, horizon),
                seed=seed + column,
            )
        )
    holm = holm_correction([row[1] for row in dm_rows], alpha=settings.significance_level)

    decisions: list[NewsHorizonAblationDecision] = []
    for column, horizon in enumerate(horizons):
        denominator = float(np.mean(clustered_market[:, column]))
        relative = (
            float(np.mean(clustered_candidate[:, column])) / denominator
            if denominator > 0
            else float("inf")
        )
        folds_beating = int(np.sum(fold_ratios[:, column] < 1.0))
        worst_fold = float(np.max(fold_ratios[:, column]))
        dm_statistic, dm_p_value = dm_rows[column]
        reasons: list[str] = []
        if not candidate_promoted_vs_har[column]:
            reasons.append("market-plus-news candidate did not clear the matched HAR gate")
        if relative >= settings.maximum_relative_qlike_to_market:
            reasons.append("news did not clear the incremental pooled QLIKE gate")
        if upper_bounds[column] >= 1.0:
            reasons.append("news bootstrap upper confidence bound did not beat market-only")
        if folds_beating < settings.minimum_folds_beating_market:
            reasons.append("too few expanding folds improved on market-only")
        if worst_fold > settings.maximum_worst_fold_relative_qlike_to_market:
            reasons.append("news worst-fold degradation exceeded the guardrail")
        if dm_statistic >= 0 or not holm[column]:
            reasons.append("paired news improvement is not Holm-significant")
        decisions.append(
            NewsHorizonAblationDecision(
                horizon=horizon,
                promoted=not reasons,
                reasons=tuple(reasons),
                relative_qlike_to_market=relative,
                relative_qlike_upper_95=upper_bounds[column],
                folds_beating_market=folds_beating,
                worst_fold_relative_qlike_to_market=worst_fold,
                dm_statistic=float(dm_statistic),
                dm_p_value=float(dm_p_value),
                holm_significant=bool(holm[column]),
                candidate_promoted_vs_har=bool(candidate_promoted_vs_har[column]),
            )
        )
    return tuple(decisions)
