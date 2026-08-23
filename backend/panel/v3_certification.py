"""Prospective holdout certification for Protocol V3 (global-cert-v3).

Features:
1. Prospective holdout boundary strictly post 2026-08-21.
2. Pre-maturity lockout: Before 252 origin sessions + max(horizons) mature sessions,
   certification returns 'locked_waiting_for_maturity' without calculating or leaking
   any performance metrics.
3. Dual-population evaluation: Evaluates reference universe D and held-out transfer assets H.
4. Independent multiplicity-corrected Holm gate evaluation across selected horizons.
5. Immutable artifact protection once holdout is opened.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from panel.cross_sectional import compute_cross_sectional_ranks, compute_relative_forward_returns
from panel.v3_candidates import BaseV3Candidate
from panel.v3_metrics import (
    SessionICMetrics,
    evaluate_session_ic_statistics,
    holm_bonferroni_family,
)
from panel.v3_selection import V3SelectionDecision

V3_CERTIFICATION_PROTOCOL_VERSION = "global-cert-v3"
DEFAULT_DEVELOPMENT_CUTOFF = "2026-08-21"
DEFAULT_PROSPECTIVE_ORIGINS = 252


@dataclass(frozen=True)
class V3CertificationGateConfig:
    protocol_version: str = V3_CERTIFICATION_PROTOCOL_VERSION
    development_cutoff: str = DEFAULT_DEVELOPMENT_CUTOFF
    prospective_origin_sessions: int = DEFAULT_PROSPECTIVE_ORIGINS

    # Mandatory Temporal IC Gates
    require_temporal_mean_ic_positive: bool = True
    require_temporal_bootstrap_lower_bound_positive: bool = True
    require_temporal_holm_hac_significance: bool = True
    min_temporal_prediction_coverage: float = 0.90
    min_temporal_ic_session_coverage: float = 0.90
    min_temporal_daily_breadth: int = 30

    # Mandatory Transfer IC Gates
    require_transfer_mean_ic_positive: bool = True
    require_transfer_bootstrap_lower_bound_positive: bool = True
    require_transfer_holm_hac_significance: bool = True
    min_transfer_prediction_coverage: float = 0.90
    min_transfer_ic_session_coverage: float = 0.90
    min_transfer_daily_breadth: int = 30

    family_alpha: float = 0.05
    resamples: int = 2000
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> V3CertificationGateConfig:
        return cls(
            protocol_version=str(data.get("protocol_version", V3_CERTIFICATION_PROTOCOL_VERSION)),
            development_cutoff=str(data.get("development_cutoff", DEFAULT_DEVELOPMENT_CUTOFF)),
            prospective_origin_sessions=int(
                data.get("prospective_origin_sessions", DEFAULT_PROSPECTIVE_ORIGINS)
            ),
            require_temporal_mean_ic_positive=bool(
                data.get("require_temporal_mean_ic_positive", True)
            ),
            require_temporal_bootstrap_lower_bound_positive=bool(
                data.get("require_temporal_bootstrap_lower_bound_positive", True)
            ),
            require_temporal_holm_hac_significance=bool(
                data.get("require_temporal_holm_hac_significance", True)
            ),
            min_temporal_prediction_coverage=float(
                data.get("min_temporal_prediction_coverage", 0.90)
            ),
            min_temporal_ic_session_coverage=float(
                data.get("min_temporal_ic_session_coverage", 0.90)
            ),
            min_temporal_daily_breadth=int(data.get("min_temporal_daily_breadth", 30)),
            require_transfer_mean_ic_positive=bool(
                data.get("require_transfer_mean_ic_positive", True)
            ),
            require_transfer_bootstrap_lower_bound_positive=bool(
                data.get("require_transfer_bootstrap_lower_bound_positive", True)
            ),
            require_transfer_holm_hac_significance=bool(
                data.get("require_transfer_holm_hac_significance", True)
            ),
            min_transfer_prediction_coverage=float(
                data.get("min_transfer_prediction_coverage", 0.90)
            ),
            min_transfer_ic_session_coverage=float(
                data.get("min_transfer_ic_session_coverage", 0.90)
            ),
            min_transfer_daily_breadth=int(data.get("min_transfer_daily_breadth", 30)),
            family_alpha=float(data.get("family_alpha", 0.05)),
            resamples=int(data.get("resamples", 2000)),
            seed=int(data.get("seed", 42)),
        )


@dataclass(frozen=True)
class V3HorizonCertificationDecision:
    horizon: int
    candidate_name: str | None
    decision: str  # "pass", "fail", "not_selected", "abstain"

    # Temporal Metrics (Reference Universe D)
    temporal_mean_ic: float | None = None
    temporal_mean_ic_ci_lower_95: float | None = None
    temporal_mean_ic_ci_upper_95: float | None = None
    temporal_hac_t_stat: float | None = None
    temporal_raw_hac_p: float | None = None
    temporal_holm_hac_p: float | None = None
    temporal_prediction_coverage: float | None = None
    temporal_session_coverage: float | None = None
    temporal_median_breadth: float | None = None

    # Transfer Metrics (Held-out Assets H)
    transfer_mean_ic: float | None = None
    transfer_mean_ic_ci_lower_95: float | None = None
    transfer_mean_ic_ci_upper_95: float | None = None
    transfer_hac_t_stat: float | None = None
    transfer_raw_hac_p: float | None = None
    transfer_holm_hac_p: float | None = None
    transfer_prediction_coverage: float | None = None
    transfer_session_coverage: float | None = None
    transfer_median_breadth: float | None = None

    passed_gates: list[str] = field(default_factory=list)
    failed_gates: list[str] = field(default_factory=list)
    gate_config: dict[str, Any] = field(default_factory=dict)
    certified_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> V3HorizonCertificationDecision:
        return cls(
            horizon=int(data["horizon"]),
            candidate_name=data.get("candidate_name"),
            decision=str(data.get("decision", "fail")),
            temporal_mean_ic=data.get("temporal_mean_ic"),
            temporal_mean_ic_ci_lower_95=data.get("temporal_mean_ic_ci_lower_95"),
            temporal_mean_ic_ci_upper_95=data.get("temporal_mean_ic_ci_upper_95"),
            temporal_hac_t_stat=data.get("temporal_hac_t_stat"),
            temporal_raw_hac_p=data.get("temporal_raw_hac_p"),
            temporal_holm_hac_p=data.get("temporal_holm_hac_p"),
            temporal_prediction_coverage=data.get("temporal_prediction_coverage"),
            temporal_session_coverage=data.get("temporal_session_coverage"),
            temporal_median_breadth=data.get("temporal_median_breadth"),
            transfer_mean_ic=data.get("transfer_mean_ic"),
            transfer_mean_ic_ci_lower_95=data.get("transfer_mean_ic_ci_lower_95"),
            transfer_mean_ic_ci_upper_95=data.get("transfer_mean_ic_ci_upper_95"),
            transfer_hac_t_stat=data.get("transfer_hac_t_stat"),
            transfer_raw_hac_p=data.get("transfer_raw_hac_p"),
            transfer_holm_hac_p=data.get("transfer_holm_hac_p"),
            transfer_prediction_coverage=data.get("transfer_prediction_coverage"),
            transfer_session_coverage=data.get("transfer_session_coverage"),
            transfer_median_breadth=data.get("transfer_median_breadth"),
            passed_gates=list(data.get("passed_gates", [])),
            failed_gates=list(data.get("failed_gates", [])),
            gate_config=dict(data.get("gate_config", {})),
            certified_at=str(data.get("certified_at", "")),
        )


@dataclass(frozen=True)
class V3ProspectiveMaturityStatus:
    is_mature: bool
    development_cutoff: str
    prospective_origin_sessions: int
    required_total_sessions: int
    observed_prospective_sessions: int
    remaining_sessions: int
    max_horizon: int
    prospective_dates: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_prospective_holdout_maturity(
    master_calendar: list[pd.Timestamp] | list[str],
    development_cutoff: str = DEFAULT_DEVELOPMENT_CUTOFF,
    prospective_origin_sessions: int = DEFAULT_PROSPECTIVE_ORIGINS,
    max_horizon: int = 30,
) -> V3ProspectiveMaturityStatus:
    """Checks whether sufficient post-cutoff sessions exist to mature the prospective holdout.

    Required sessions:
        prospective_origin_sessions + max_horizon
    """
    cutoff_ts = pd.Timestamp(development_cutoff)
    calendar_ts = [pd.Timestamp(d) for d in master_calendar]
    prospective_dates = [d for d in calendar_ts if d > cutoff_ts]

    n_observed = len(prospective_dates)
    required_total = prospective_origin_sessions + max_horizon
    remaining = max(0, required_total - n_observed)
    is_mature = n_observed >= required_total

    return V3ProspectiveMaturityStatus(
        is_mature=is_mature,
        development_cutoff=development_cutoff,
        prospective_origin_sessions=prospective_origin_sessions,
        required_total_sessions=required_total,
        observed_prospective_sessions=n_observed,
        remaining_sessions=remaining,
        max_horizon=max_horizon,
        prospective_dates=[d.strftime("%Y-%m-%d") for d in prospective_dates],
    )


def evaluate_v3_prospective_certification(
    frozen_candidates_by_horizon: dict[int, BaseV3Candidate],
    selection_decisions: dict[int, V3SelectionDecision],
    all_panels: dict[str, pd.DataFrame],
    master_calendar: list[pd.Timestamp] | list[str],
    dev_tickers: list[str] | set[str],
    transfer_tickers: list[str] | set[str],
    gate_config: V3CertificationGateConfig,
    *,
    open_locked_holdout: bool = False,
) -> dict[str, Any]:
    """Evaluates prospective holdout for Protocol V3.

    Returns:
        Certification result dictionary. If holdout is immature or not opened,
        returns metadata-only status with zero metric calculations.
    """
    horizons = sorted(selection_decisions.keys())
    max_h = max(horizons) if horizons else 30

    maturity = check_prospective_holdout_maturity(
        master_calendar,
        development_cutoff=gate_config.development_cutoff,
        prospective_origin_sessions=gate_config.prospective_origin_sessions,
        max_horizon=max_h,
    )

    if not maturity.is_mature:
        return {
            "certification_protocol_version": gate_config.protocol_version,
            "status": "locked_waiting_for_maturity",
            "decision": "locked_waiting_for_maturity",
            "maturity_status": maturity.to_dict(),
            "gate_config": gate_config.to_dict(),
            "certified_horizons": [],
            "decisions": {},
        }

    if not open_locked_holdout:
        return {
            "certification_protocol_version": gate_config.protocol_version,
            "status": "locked_mature_ready_to_open",
            "decision": "locked_mature_ready_to_open",
            "maturity_status": maturity.to_dict(),
            "gate_config": gate_config.to_dict(),
            "certified_horizons": [],
            "decisions": {},
        }

    # Slice evaluation window: exactly the first prospective_origin_sessions after cutoff
    eval_dates = [
        pd.Timestamp(d)
        for d in maturity.prospective_dates[: gate_config.prospective_origin_sessions]
    ]
    eval_date_set = set(eval_dates)

    # 1. Compute causal cross-sectional rank features across all tickers with strict D/H isolation
    ranked_panels = compute_cross_sectional_ranks(
        all_panels,
        dev_tickers=dev_tickers,
        min_reference_assets=gate_config.min_temporal_daily_breadth,
    )

    # Slice features to evaluation origin dates
    eval_features_d: dict[str, pd.DataFrame] = {
        t: ranked_panels[t].loc[ranked_panels[t].index.isin(eval_date_set)]
        for t in dev_tickers
        if t in ranked_panels
    }
    eval_features_h: dict[str, pd.DataFrame] = {
        t: ranked_panels[t].loc[ranked_panels[t].index.isin(eval_date_set)]
        for t in transfer_tickers
        if t in ranked_panels
    }

    # Collect raw metrics for multiplicity correction across selected horizons
    raw_temporal_p: dict[int, float] = {}
    raw_transfer_p: dict[int, float] = {}
    temp_metrics_by_h: dict[int, SessionICMetrics] = {}
    trans_metrics_by_h: dict[int, SessionICMetrics] = {}

    selected_horizons = [
        h
        for h, d in selection_decisions.items()
        if d.status == "selected" and d.candidate is not None
    ]

    for h in selected_horizons:
        cand = frozen_candidates_by_horizon[h]

        # Predict scores
        pred_scores_d = cand.predict(eval_features_d)
        pred_scores_h = cand.predict(eval_features_h)

        # Compute relative forward returns
        _, rel_targets = compute_relative_forward_returns(
            all_panels,
            h,
            dev_tickers=dev_tickers,
            min_reference_assets=gate_config.min_temporal_daily_breadth,
        )

        rel_targets_d = pd.DataFrame(
            {
                t: rel_targets[t].loc[rel_targets[t].index.isin(eval_date_set)]
                for t in dev_tickers
                if t in rel_targets
            }
        )
        rel_targets_h = pd.DataFrame(
            {
                t: rel_targets[t].loc[rel_targets[t].index.isin(eval_date_set)]
                for t in transfer_tickers
                if t in rel_targets
            }
        )

        m_d = evaluate_session_ic_statistics(
            pred_scores_d,
            rel_targets_d,
            h,
            min_daily_asset_count=gate_config.min_temporal_daily_breadth,
            resamples=gate_config.resamples,
            seed=gate_config.seed,
        )
        m_h = evaluate_session_ic_statistics(
            pred_scores_h,
            rel_targets_h,
            h,
            min_daily_asset_count=gate_config.min_transfer_daily_breadth,
            resamples=gate_config.resamples,
            seed=gate_config.seed,
        )

        temp_metrics_by_h[h] = m_d
        trans_metrics_by_h[h] = m_h
        raw_temporal_p[h] = m_d.raw_one_sided_hac_p
        raw_transfer_p[h] = m_h.raw_one_sided_hac_p

    # Multiplicity correction over selected horizons
    holm_temp = holm_bonferroni_family(
        {(h, "temporal"): p for h, p in raw_temporal_p.items()}, alpha=gate_config.family_alpha
    )
    holm_trans = holm_bonferroni_family(
        {(h, "transfer"): p for h, p in raw_transfer_p.items()}, alpha=gate_config.family_alpha
    )

    decisions: dict[str, Any] = {}
    certified_horizons: list[int] = []

    for h in horizons:
        sel = selection_decisions[h]
        if sel.status != "selected" or sel.candidate is None:
            decisions[str(h)] = V3HorizonCertificationDecision(
                horizon=h,
                candidate_name=None,
                decision="not_selected",
                gate_config=gate_config.to_dict(),
            ).to_dict()
            continue

        m_d = temp_metrics_by_h[h]
        m_h = trans_metrics_by_h[h]
        temp_reject, temp_adj_p = holm_temp[(h, "temporal")]
        trans_reject, trans_adj_p = holm_trans[(h, "transfer")]

        passed_gates: list[str] = []
        failed_gates: list[str] = []

        # Temporal Gates
        if m_d.mean_spearman_ic > 0:
            passed_gates.append(f"temporal_mean_ic({m_d.mean_spearman_ic:.4f} > 0)")
        else:
            failed_gates.append(f"temporal_mean_ic({m_d.mean_spearman_ic:.4f} <= 0)")

        if m_d.mean_ic_ci_lower_95 > 0:
            passed_gates.append(f"temporal_bootstrap_lower_95({m_d.mean_ic_ci_lower_95:.4f} > 0)")
        else:
            failed_gates.append(f"temporal_bootstrap_lower_95({m_d.mean_ic_ci_lower_95:.4f} <= 0)")

        if temp_reject:
            passed_gates.append(
                f"temporal_holm_hac_p({temp_adj_p:.4f} <= {gate_config.family_alpha:.2f})"
            )
        else:
            failed_gates.append(
                f"temporal_holm_hac_p({temp_adj_p:.4f} > {gate_config.family_alpha:.2f})"
            )

        if m_d.prediction_row_coverage >= gate_config.min_temporal_prediction_coverage:
            passed_gates.append(
                f"temporal_prediction_coverage({m_d.prediction_row_coverage:.2f} >= {gate_config.min_temporal_prediction_coverage:.2f})"
            )
        else:
            failed_gates.append(
                f"temporal_prediction_coverage({m_d.prediction_row_coverage:.2f} < {gate_config.min_temporal_prediction_coverage:.2f})"
            )

        if m_d.ic_session_coverage >= gate_config.min_temporal_ic_session_coverage:
            passed_gates.append(
                f"temporal_session_coverage({m_d.ic_session_coverage:.2f} >= {gate_config.min_temporal_ic_session_coverage:.2f})"
            )
        else:
            failed_gates.append(
                f"temporal_session_coverage({m_d.ic_session_coverage:.2f} < {gate_config.min_temporal_ic_session_coverage:.2f})"
            )

        if m_d.median_daily_asset_breadth >= gate_config.min_temporal_daily_breadth:
            passed_gates.append(
                f"temporal_breadth({m_d.median_daily_asset_breadth:.1f} >= {gate_config.min_temporal_daily_breadth})"
            )
        else:
            failed_gates.append(
                f"temporal_breadth({m_d.median_daily_asset_breadth:.1f} < {gate_config.min_temporal_daily_breadth})"
            )

        # Transfer Gates
        if m_h.mean_spearman_ic > 0:
            passed_gates.append(f"transfer_mean_ic({m_h.mean_spearman_ic:.4f} > 0)")
        else:
            failed_gates.append(f"transfer_mean_ic({m_h.mean_spearman_ic:.4f} <= 0)")

        if m_h.mean_ic_ci_lower_95 > 0:
            passed_gates.append(f"transfer_bootstrap_lower_95({m_h.mean_ic_ci_lower_95:.4f} > 0)")
        else:
            failed_gates.append(f"transfer_bootstrap_lower_95({m_h.mean_ic_ci_lower_95:.4f} <= 0)")

        if trans_reject:
            passed_gates.append(
                f"transfer_holm_hac_p({trans_adj_p:.4f} <= {gate_config.family_alpha:.2f})"
            )
        else:
            failed_gates.append(
                f"transfer_holm_hac_p({trans_adj_p:.4f} > {gate_config.family_alpha:.2f})"
            )

        if m_h.prediction_row_coverage >= gate_config.min_transfer_prediction_coverage:
            passed_gates.append(
                f"transfer_prediction_coverage({m_h.prediction_row_coverage:.2f} >= {gate_config.min_transfer_prediction_coverage:.2f})"
            )
        else:
            failed_gates.append(
                f"transfer_prediction_coverage({m_h.prediction_row_coverage:.2f} < {gate_config.min_transfer_prediction_coverage:.2f})"
            )

        if m_h.ic_session_coverage >= gate_config.min_transfer_ic_session_coverage:
            passed_gates.append(
                f"transfer_session_coverage({m_h.ic_session_coverage:.2f} >= {gate_config.min_transfer_ic_session_coverage:.2f})"
            )
        else:
            failed_gates.append(
                f"transfer_session_coverage({m_h.ic_session_coverage:.2f} < {gate_config.min_transfer_ic_session_coverage:.2f})"
            )

        if m_h.median_daily_asset_breadth >= gate_config.min_transfer_daily_breadth:
            passed_gates.append(
                f"transfer_breadth({m_h.median_daily_asset_breadth:.1f} >= {gate_config.min_transfer_daily_breadth})"
            )
        else:
            failed_gates.append(
                f"transfer_breadth({m_h.median_daily_asset_breadth:.1f} < {gate_config.min_transfer_daily_breadth})"
            )

        decision_str = "pass" if len(failed_gates) == 0 else "fail"
        if decision_str == "pass":
            certified_horizons.append(h)

        decisions[str(h)] = V3HorizonCertificationDecision(
            horizon=h,
            candidate_name=sel.candidate,
            decision=decision_str,
            temporal_mean_ic=m_d.mean_spearman_ic,
            temporal_mean_ic_ci_lower_95=m_d.mean_ic_ci_lower_95,
            temporal_mean_ic_ci_upper_95=m_d.mean_ic_ci_upper_95,
            temporal_hac_t_stat=m_d.hac_t_stat,
            temporal_raw_hac_p=m_d.raw_one_sided_hac_p,
            temporal_holm_hac_p=temp_adj_p,
            temporal_prediction_coverage=m_d.prediction_row_coverage,
            temporal_session_coverage=m_d.ic_session_coverage,
            temporal_median_breadth=m_d.median_daily_asset_breadth,
            transfer_mean_ic=m_h.mean_spearman_ic,
            transfer_mean_ic_ci_lower_95=m_h.mean_ic_ci_lower_95,
            transfer_mean_ic_ci_upper_95=m_h.mean_ic_ci_upper_95,
            transfer_hac_t_stat=m_h.hac_t_stat,
            transfer_raw_hac_p=m_h.raw_one_sided_hac_p,
            transfer_holm_hac_p=trans_adj_p,
            transfer_prediction_coverage=m_h.prediction_row_coverage,
            transfer_session_coverage=m_h.ic_session_coverage,
            transfer_median_breadth=m_h.median_daily_asset_breadth,
            passed_gates=passed_gates,
            failed_gates=failed_gates,
            gate_config=gate_config.to_dict(),
        ).to_dict()

    if not selected_horizons:
        global_decision = "no_horizons_selected"
    elif len(certified_horizons) == len(horizons):
        global_decision = "pass_all"
    elif len(certified_horizons) > 0:
        global_decision = "partial"
    else:
        global_decision = "fail_all"

    return {
        "certification_protocol_version": gate_config.protocol_version,
        "status": "holdout_opened",
        "decision": global_decision,
        "maturity_status": maturity.to_dict(),
        "gate_config": gate_config.to_dict(),
        "certified_horizons": certified_horizons,
        "decisions": decisions,
    }
