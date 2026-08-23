"""Cross-sectional expanding-fold evaluation and champion selection for Protocol V3.

Enforces 7 pre-registered development selection gates:
1. mean_spearman_ic > 0
2. holm_adjusted_p <= 0.05 (familywise over all (horizon, candidate) pairs)
3. mean_ic_ci_lower_95 > 0 (moving-block bootstrap)
4. fold_positive_fraction >= 0.80 (>= 4 of 5 folds have positive mean IC)
5. prediction_row_coverage >= 0.90
6. valid_ic_session_coverage >= 0.90
7. median_daily_breadth >= 30

If no candidate passes all 7 gates, status = 'abstain_no_robust_rank_signal'.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from panel.folds import CalendarFold
from panel.v3_candidates import BaseV3Candidate
from panel.v3_metrics import (
    SessionICMetrics,
    compute_session_rank_ic,
    evaluate_session_ic_statistics,
    holm_bonferroni_family,
)


@dataclass(frozen=True)
class V3CandidateFoldResult:
    fold_index: int
    train_sessions: int
    val_sessions: int
    mean_spearman_ic: float
    median_spearman_ic: float
    valid_sessions: int
    total_val_sessions: int


@dataclass(frozen=True)
class V3CandidateEvidence:
    candidate_name: str
    horizon: int
    overall_metrics: SessionICMetrics
    fold_metrics: list[V3CandidateFoldResult]
    positive_fold_count: int
    positive_fold_fraction: float

    # Raw out-of-fold daily IC series for audit
    daily_ic: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["overall_metrics"] = self.overall_metrics.to_dict()
        d["fold_metrics"] = [asdict(f) for f in self.fold_metrics]
        return d


@dataclass(frozen=True)
class V3SelectionDecision:
    horizon: int
    candidate: str | None
    status: str  # "selected" or "abstain_no_robust_rank_signal"
    mean_spearman_ic: float | None = None
    mean_ic_ci_lower_95: float | None = None
    mean_ic_ci_upper_95: float | None = None
    hac_t_stat: float | None = None
    raw_hac_p: float | None = None
    holm_adjusted_p: float | None = None
    positive_fold_count: int | None = None
    positive_fold_fraction: float | None = None
    valid_ic_session_coverage: float | None = None
    prediction_row_coverage: float | None = None
    median_daily_breadth: float | None = None
    candidate_hyperparameters: dict[str, Any] | None = None
    passed_gates: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> V3SelectionDecision:
        return cls(
            horizon=int(data["horizon"]),
            candidate=data.get("candidate"),
            status=str(data.get("status", "abstain_no_robust_rank_signal")),
            mean_spearman_ic=data.get("mean_spearman_ic"),
            mean_ic_ci_lower_95=data.get("mean_ic_ci_lower_95"),
            mean_ic_ci_upper_95=data.get("mean_ic_ci_upper_95"),
            hac_t_stat=data.get("hac_t_stat"),
            raw_hac_p=data.get("raw_hac_p"),
            holm_adjusted_p=data.get("holm_adjusted_p"),
            positive_fold_count=data.get("positive_fold_count"),
            positive_fold_fraction=data.get("positive_fold_fraction"),
            valid_ic_session_coverage=data.get("valid_ic_session_coverage"),
            prediction_row_coverage=data.get("prediction_row_coverage"),
            median_daily_breadth=data.get("median_daily_breadth"),
            candidate_hyperparameters=data.get("candidate_hyperparameters"),
            passed_gates=list(data.get("passed_gates", [])),
            reasons=list(data.get("reasons", [])),
        )


def evaluate_v3_candidate_on_folds(
    candidate: BaseV3Candidate,
    horizon: int,
    folds: list[CalendarFold],
    dev_features: dict[str, pd.DataFrame],
    dev_targets: dict[str, pd.Series],
    *,
    min_daily_asset_count: int = 30,
    resamples: int = 2000,
    seed: int = 42,
) -> V3CandidateEvidence:
    """Evaluates a candidate across expanding calendar folds and aggregates out-of-fold IC."""
    oof_scores_list: list[pd.DataFrame] = []
    fold_results: list[V3CandidateFoldResult] = []

    for fold in folds:
        # Slice train features and targets with explicit date/index-based alignment
        train_features: dict[str, pd.DataFrame] = {}
        train_targets: dict[str, pd.Series] = {}
        for t, df in dev_features.items():
            feat_mask = (df.index >= fold.train_start) & (df.index <= fold.train_end)
            if np.any(feat_mask):
                feat_slice = df.loc[feat_mask]
                train_features[t] = feat_slice
                if t in dev_targets:
                    tgt_series = dev_targets[t]
                    # Date/index-based alignment: strictly align target on feature timestamps
                    train_targets[t] = tgt_series.reindex(feat_slice.index)

        # Fit candidate on train data
        candidate.fit(train_features, train_targets)

        # Slice val features
        val_features: dict[str, pd.DataFrame] = {}
        for t, df in dev_features.items():
            val_mask = (df.index >= fold.val_start) & (df.index <= fold.val_end)
            if np.any(val_mask):
                val_features[t] = df.loc[val_mask]

        # Predict on val features
        val_scores = candidate.predict(val_features)
        oof_scores_list.append(val_scores)

        # Compute fold-specific IC with explicit date alignment
        val_targets_dict: dict[str, pd.Series] = {}
        for t in dev_targets:
            tgt_series = dev_targets[t]
            val_t_mask = (tgt_series.index >= fold.val_start) & (tgt_series.index <= fold.val_end)
            if np.any(val_t_mask):
                val_targets_dict[t] = tgt_series.loc[val_t_mask]
        val_targets_df = pd.DataFrame(val_targets_dict).reindex(val_scores.index)
        fold_ic, _ = compute_session_rank_ic(
            val_scores, val_targets_df, min_daily_asset_count=min_daily_asset_count
        )
        v_ic = fold_ic.dropna()
        f_mean = float(v_ic.mean()) if len(v_ic) > 0 else 0.0
        f_med = float(v_ic.median()) if len(v_ic) > 0 else 0.0

        fold_results.append(
            V3CandidateFoldResult(
                fold_index=fold.fold_index,
                train_sessions=fold.n_train_sessions,
                val_sessions=fold.n_val_sessions,
                mean_spearman_ic=f_mean,
                median_spearman_ic=f_med,
                valid_sessions=len(v_ic),
                total_val_sessions=len(fold_ic),
            )
        )

    # Combine OOF scores across folds
    combined_scores = pd.concat(oof_scores_list).sort_index()
    combined_targets_df = pd.DataFrame(dev_targets).reindex(combined_scores.index)

    overall_metrics = evaluate_session_ic_statistics(
        combined_scores,
        combined_targets_df,
        horizon,
        min_daily_asset_count=min_daily_asset_count,
        resamples=resamples,
        seed=seed,
    )

    daily_ic_series, _ = compute_session_rank_ic(
        combined_scores, combined_targets_df, min_daily_asset_count=min_daily_asset_count
    )
    daily_ic_dict = {str(d): float(v) for d, v in daily_ic_series.dropna().items()}

    pos_folds = sum(1 for f in fold_results if f.mean_spearman_ic > 0)
    pos_fraction = float(pos_folds / max(1, len(fold_results)))

    return V3CandidateEvidence(
        candidate_name=candidate.name,
        horizon=horizon,
        overall_metrics=overall_metrics,
        fold_metrics=fold_results,
        positive_fold_count=pos_folds,
        positive_fold_fraction=pos_fraction,
        daily_ic=daily_ic_dict,
    )


def select_v3_champions(
    evidence_by_pair: dict[tuple[int, str], V3CandidateEvidence],
    candidate_objects: dict[str, BaseV3Candidate],
    candidate_order: list[str],
    horizons: list[int],
    *,
    alpha: float = 0.05,
    min_positive_fold_fraction: float = 0.80,
    min_prediction_coverage: float = 0.90,
    min_ic_session_coverage: float = 0.90,
    min_daily_asset_count: int = 30,
) -> dict[int, V3SelectionDecision]:
    """Selects champion candidates across horizons using Holm multiple-testing correction."""
    # 1. Collect all raw HAC p-values across (horizon, candidate_name) family
    raw_p_values: dict[tuple[int, str], float] = {
        key: ev.overall_metrics.raw_one_sided_hac_p for key, ev in evidence_by_pair.items()
    }

    # 2. Compute Holm step-down correction over the entire candidate x horizon family
    holm_results = holm_bonferroni_family(raw_p_values, alpha=alpha)

    decisions: dict[int, V3SelectionDecision] = {}

    for h in horizons:
        eligible_candidates: list[tuple[str, V3CandidateEvidence, float]] = []

        for cand_name in candidate_order:
            key = (h, cand_name)
            if key not in evidence_by_pair:
                continue

            ev = evidence_by_pair[key]
            m = ev.overall_metrics
            reject_null, adj_p = holm_results[key]

            # Gate 1: Mean IC > 0
            g1 = m.mean_spearman_ic > 0.0
            # Gate 2: Holm-adjusted HAC p <= alpha
            g2 = reject_null and adj_p <= alpha
            # Gate 3: Moving-block bootstrap lower 95% CI > 0
            g3 = m.mean_ic_ci_lower_95 > 0.0
            # Gate 4: At least 4 of 5 fold mean ICs > 0
            g4 = ev.positive_fold_fraction >= min_positive_fold_fraction
            # Gate 5: Prediction row coverage >= 0.90
            g5 = m.prediction_row_coverage >= min_prediction_coverage
            # Gate 6: Valid IC session coverage >= 0.90
            g6 = m.ic_session_coverage >= min_ic_session_coverage
            # Gate 7: Median daily asset breadth >= 30
            g7 = m.median_daily_asset_breadth >= min_daily_asset_count

            if g1 and g2 and g3 and g4 and g5 and g6 and g7:
                eligible_candidates.append((cand_name, ev, adj_p))

        if not eligible_candidates:
            decisions[h] = V3SelectionDecision(
                horizon=h,
                candidate=None,
                status="abstain_no_robust_rank_signal",
                reasons=["No candidate satisfied all 7 pre-registered development rank gates."],
            )
        else:
            # Ranking criteria among eligible:
            # Primary: Highest bootstrap lower 95% CI bound (stability)
            # Tie-breaker 1: Highest mean IC
            # Tie-breaker 2: Candidate config order index
            def _rank_key(item: tuple[str, V3CandidateEvidence, float]) -> tuple[float, float, int]:
                name, ev, _ = item
                ci_lo = ev.overall_metrics.mean_ic_ci_lower_95
                mean_ic = ev.overall_metrics.mean_spearman_ic
                order_idx = -candidate_order.index(name) if name in candidate_order else -999
                return (ci_lo, mean_ic, order_idx)

            best_name, best_ev, best_adj_p = max(eligible_candidates, key=_rank_key)
            best_m = best_ev.overall_metrics

            cand_obj = candidate_objects[best_name]
            hyperparams = cand_obj.to_dict()

            passed_gates = [
                f"mean_spearman_ic({best_m.mean_spearman_ic:.4f} > 0)",
                f"holm_hac_p({best_adj_p:.4f} <= {alpha:.2f})",
                f"bootstrap_lower_95({best_m.mean_ic_ci_lower_95:.4f} > 0)",
                f"fold_stability({best_ev.positive_fold_fraction:.2f} >= {min_positive_fold_fraction:.2f})",
                f"prediction_coverage({best_m.prediction_row_coverage:.2f} >= {min_prediction_coverage:.2f})",
                f"session_coverage({best_m.ic_session_coverage:.2f} >= {min_ic_session_coverage:.2f})",
                f"median_breadth({best_m.median_daily_asset_breadth:.1f} >= {min_daily_asset_count})",
            ]

            decisions[h] = V3SelectionDecision(
                horizon=h,
                candidate=best_name,
                status="selected",
                mean_spearman_ic=best_m.mean_spearman_ic,
                mean_ic_ci_lower_95=best_m.mean_ic_ci_lower_95,
                mean_ic_ci_upper_95=best_m.mean_ic_ci_upper_95,
                hac_t_stat=best_m.hac_t_stat,
                raw_hac_p=best_m.raw_one_sided_hac_p,
                holm_adjusted_p=best_adj_p,
                positive_fold_count=best_ev.positive_fold_count,
                positive_fold_fraction=best_ev.positive_fold_fraction,
                valid_ic_session_coverage=best_m.ic_session_coverage,
                prediction_row_coverage=best_m.prediction_row_coverage,
                median_daily_breadth=best_m.median_daily_asset_breadth,
                candidate_hyperparameters=hyperparams,
                passed_gates=passed_gates,
                reasons=[f"Selected as highest-confidence ranking candidate for horizon {h}d."],
            )

    return decisions
