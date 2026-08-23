"""Locked holdout certification for global forecasting models (Slice-10 / Certification Gate).

Evaluates frozen champion models on:
1. Untouched Temporal Holdout (last N sessions of master calendar)
2. Untouched Asset-Transfer Holdout (tickers completely excluded from development)

Enforces strict statistical gates before any release artifact is approved.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from panel.candidates import REGISTRY, CandidateTargets
from panel.features import DeployableFeatureContract
from panel.selection import SelectionDecision


@dataclass(frozen=True)
class CertificationGateConfig:
    """Predeclared certification gate configuration for Protocol V2.

    Every field declared here either directly participates in the pass/fail decision
    or is explicitly documented as a descriptive diagnostic.
    """

    # Temporal holdout non-degradation gates
    require_temporal_relative_rmse: bool = True
    max_temporal_relative_rmse: float = 1.00

    require_temporal_relative_mae: bool = True
    max_temporal_relative_mae: float = 1.00

    # Asset-transfer holdout non-degradation gates (mandatory by default in Protocol V2)
    require_transfer_relative_rmse: bool = True
    max_transfer_relative_rmse: float = 1.00

    require_transfer_relative_mae: bool = True
    max_transfer_relative_mae: float = 1.00

    # Directional skill gate (vs majority baseline prevalence on evaluated non-neutral rows)
    require_direction_skill: bool = False
    min_direction_accuracy_delta_vs_majority: float = 0.00

    # Probabilistic calibration gate (only evaluated when direction_probabilities are available)
    require_probabilistic_direction: bool = False
    max_direction_brier: float | None = None

    protocol_version: str = "global-cert-v2"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CertificationGateConfig:
        return cls(
            require_temporal_relative_rmse=bool(data.get("require_temporal_relative_rmse", True)),
            max_temporal_relative_rmse=float(data.get("max_temporal_relative_rmse", 1.00)),
            require_temporal_relative_mae=bool(data.get("require_temporal_relative_mae", True)),
            max_temporal_relative_mae=float(data.get("max_temporal_relative_mae", 1.00)),
            require_transfer_relative_rmse=bool(data.get("require_transfer_relative_rmse", True)),
            max_transfer_relative_rmse=float(data.get("max_transfer_relative_rmse", 1.00)),
            require_transfer_relative_mae=bool(data.get("require_transfer_relative_mae", True)),
            max_transfer_relative_mae=float(data.get("max_transfer_relative_mae", 1.00)),
            require_direction_skill=bool(data.get("require_direction_skill", False)),
            min_direction_accuracy_delta_vs_majority=float(
                data.get("min_direction_accuracy_delta_vs_majority", 0.00)
            ),
            require_probabilistic_direction=bool(
                data.get("require_probabilistic_direction", False)
            ),
            max_direction_brier=(
                float(data["max_direction_brier"])
                if data.get("max_direction_brier") is not None
                else None
            ),
            protocol_version=str(data.get("protocol_version", "global-cert-v2")),
        )


@dataclass(frozen=True)
class CertificationDecision:
    horizon: int
    candidate_name: str
    decision: str  # "pass", "fail", "abstain"

    # Point error metrics
    temporal_relative_rmse: float
    temporal_relative_mae: float
    transfer_relative_rmse: float
    transfer_relative_mae: float

    # Direction diagnostics (descriptive unless require_direction_skill is True)
    temporal_direction_acc: float
    positive_prevalence: float
    subset_positive_prevalence: float
    majority_class_accuracy: float
    direction_accuracy_delta_vs_majority: float
    direction_coverage: float
    balanced_accuracy: float

    # Probabilistic diagnostics (None if candidate is return-only)
    temporal_brier: float | None
    direction_probability_status: str  # "evaluated" | "not_available" | "partial" | "invalid"
    direction_probability_coverage: float

    # Gate audit trail
    passed_gates: list[str] = field(default_factory=list)
    failed_gates: list[str] = field(default_factory=list)
    gate_config: dict[str, Any] = field(default_factory=dict)
    certification_protocol_version: str = "global-cert-v2"

    temporal_sessions: int = 0
    transfer_ticker_count: int = 0
    certified_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CertificationDecision:
        pos_prev = float(data.get("positive_prevalence", 0.5))
        return cls(
            horizon=int(data.get("horizon", 0)),
            candidate_name=str(data.get("candidate_name", "none")),
            decision=str(data.get("decision", "abstain")),
            temporal_relative_rmse=float(data.get("temporal_relative_rmse", 1.0)),
            temporal_relative_mae=float(data.get("temporal_relative_mae", 1.0)),
            transfer_relative_rmse=float(data.get("transfer_relative_rmse", 1.0)),
            transfer_relative_mae=float(data.get("transfer_relative_mae", 1.0)),
            temporal_direction_acc=float(data.get("temporal_direction_acc", 0.5)),
            positive_prevalence=pos_prev,
            subset_positive_prevalence=float(data.get("subset_positive_prevalence", pos_prev)),
            majority_class_accuracy=float(data.get("majority_class_accuracy", 0.5)),
            direction_accuracy_delta_vs_majority=float(
                data.get("direction_accuracy_delta_vs_majority", 0.0)
            ),
            direction_coverage=float(data.get("direction_coverage", 1.0)),
            balanced_accuracy=float(data.get("balanced_accuracy", 0.5)),
            temporal_brier=(
                float(data["temporal_brier"]) if data.get("temporal_brier") is not None else None
            ),
            direction_probability_status=str(
                data.get("direction_probability_status", "not_available")
            ),
            direction_probability_coverage=float(
                data.get(
                    "direction_probability_coverage",
                    1.0 if data.get("direction_probability_status") == "evaluated" else 0.0,
                )
            ),
            passed_gates=list(data.get("passed_gates", [])),
            failed_gates=list(data.get("failed_gates", [])),
            gate_config=dict(data.get("gate_config", {})),
            certification_protocol_version=str(
                data.get("certification_protocol_version", "global-cert-v1")
            ),
            temporal_sessions=int(data.get("temporal_sessions", 0)),
            transfer_ticker_count=int(data.get("transfer_ticker_count", 0)),
            certified_at=str(data.get("certified_at", datetime.now(UTC).isoformat())),
        )


def evaluate_locked_certification(
    *,
    horizon: int,
    champion_decision: SelectionDecision,
    universe_data: dict[str, pd.DataFrame],
    features_by_ticker: dict[str, pd.DataFrame],
    temporal_holdout_dates: pd.DatetimeIndex,
    asset_transfer_tickers: list[str],
    dev_train_tickers: list[str],
    seed: int = 42,
    gate_config: CertificationGateConfig | None = None,
) -> CertificationDecision:
    """Evaluate a selected champion on the locked temporal and asset holdouts under Protocol V2."""
    if gate_config is None:
        gate_config = CertificationGateConfig()

    cand_name = champion_decision.candidate_name
    if (
        champion_decision.status == "experimental_no_demonstrated_edge"
        or not cand_name
        or cand_name not in REGISTRY
    ):
        return CertificationDecision(
            horizon=horizon,
            candidate_name=cand_name or "none",
            decision="abstain",
            temporal_relative_rmse=1.0,
            temporal_relative_mae=1.0,
            transfer_relative_rmse=1.0,
            transfer_relative_mae=1.0,
            temporal_direction_acc=0.5,
            positive_prevalence=0.5,
            subset_positive_prevalence=0.5,
            majority_class_accuracy=0.5,
            direction_accuracy_delta_vs_majority=0.0,
            direction_coverage=0.0,
            balanced_accuracy=0.5,
            temporal_brier=None,
            direction_probability_status="not_available",
            direction_probability_coverage=0.0,
            passed_gates=[],
            failed_gates=["no_champion_selected"],
            gate_config=gate_config.to_dict(),
            certification_protocol_version=gate_config.protocol_version,
            temporal_sessions=len(temporal_holdout_dates),
            transfer_ticker_count=len(asset_transfer_tickers),
        )

    contract = DeployableFeatureContract()
    feature_cols = [
        c
        for c in features_by_ticker[dev_train_tickers[0]].columns
        if c in contract.feature_names
        and pd.api.types.is_numeric_dtype(features_by_ticker[dev_train_tickers[0]][c])
    ]

    # 1. Prepare temporal holdout evaluation rows (evaluated on development tickers during temporal holdout)
    temp_cand_losses: list[float] = []
    temp_base_losses: list[float] = []
    actual_ups: list[bool] = []
    pred_ups: list[bool | None] = []
    brier_losses: list[float] = []
    valid_prob_count: int = 0
    invalid_prob_count: int = 0

    # Fit candidate model on development data
    x_dev_train: list[np.ndarray] = []
    y_dev_train: list[float] = []
    d_dev_train: list[int] = []

    for ticker in dev_train_tickers:
        f = features_by_ticker[ticker]
        c = universe_data[ticker]["Close"]
        cumret = np.log(c.shift(-horizon) / c)
        feat_vals = f[feature_cols].to_numpy(dtype=float)

        # Use rows prior to temporal holdout
        dev_dates = f.index[~f.index.isin(temporal_holdout_dates)]
        if len(dev_dates) < contract.window_size + horizon + 1:
            continue

        for t in range(contract.window_size, len(dev_dates) - horizon):
            w = feat_vals[t - contract.window_size : t]
            tgt = cumret.iloc[t - 1]
            if np.isfinite(tgt) and np.isfinite(w).all():
                x_dev_train.append(w)
                y_dev_train.append(float(tgt))
                d_dev_train.append(1 if abs(tgt) < 0.005 else (2 if tgt > 0 else 0))

    if not x_dev_train:
        raise ValueError("Insufficient training rows for certification model fit.")

    model = REGISTRY[cand_name](seed)
    targets_tr = CandidateTargets(
        cumulative_returns=np.asarray(y_dev_train, dtype=np.float32),
        direction_classes=np.asarray(d_dev_train, dtype=int),
    )
    model.fit(np.stack(x_dev_train), targets_tr)

    # Evaluate on temporal holdout for dev_train_tickers
    for ticker in dev_train_tickers:
        f = features_by_ticker[ticker]
        c = universe_data[ticker]["Close"]
        cumret = np.log(c.shift(-horizon) / c)
        feat_vals = f[feature_cols].to_numpy(dtype=float)

        for t_date in temporal_holdout_dates:
            if t_date not in f.index:
                continue
            idx = f.index.get_loc(t_date)
            if idx < contract.window_size or idx >= len(f) - horizon:
                continue
            w = feat_vals[idx - contract.window_size : idx]
            tgt = cumret.iloc[idx - 1]
            if np.isfinite(tgt) and np.isfinite(w).all():
                pred = model.predict(np.expand_dims(w, axis=0))
                pt = float(pred.return_point[0]) if pred.return_point is not None else 0.0

                # Blend with persistence if alpha < 1.0 (persistence point return is 0)
                blended_pt = float(champion_decision.alpha * pt)
                temp_cand_losses.append(abs(blended_pt - float(tgt)))
                temp_base_losses.append(abs(float(tgt)))

                is_up_actual = bool(float(tgt) > 0)
                actual_ups.append(is_up_actual)

                # Classify non-zero blended forecast direction (abstain if zero/near-zero)
                if blended_pt > 1e-9:
                    pred_ups.append(True)
                elif blended_pt < -1e-9:
                    pred_ups.append(False)
                else:
                    pred_ups.append(None)

                # Probabilistic Brier if genuine, valid probabilities exist
                if pred.direction_probabilities is not None:
                    raw_probs = np.asarray(pred.direction_probabilities, dtype=float).squeeze()
                    is_valid_shape = raw_probs.ndim == 1 and raw_probs.size in (2, 3)
                    is_finite = bool(np.isfinite(raw_probs).all()) if is_valid_shape else False
                    in_bounds = (
                        bool((raw_probs >= 0.0).all() and (raw_probs <= 1.0).all())
                        if is_finite
                        else False
                    )
                    sums_to_one = (
                        bool(abs(float(np.sum(raw_probs)) - 1.0) < 1e-4) if in_bounds else False
                    )

                    if is_valid_shape and is_finite and in_bounds and sums_to_one:
                        p_up = float(raw_probs[2]) if raw_probs.size == 3 else float(raw_probs[1])
                        y_up_num = 1.0 if is_up_actual else 0.0
                        brier_losses.append((p_up - y_up_num) ** 2)
                        valid_prob_count += 1
                    else:
                        invalid_prob_count += 1

    # 2. Evaluate on asset-transfer holdout
    transfer_cand_losses: list[float] = []
    transfer_base_losses: list[float] = []

    for ticker in asset_transfer_tickers:
        if ticker not in features_by_ticker or ticker not in universe_data:
            continue
        f = features_by_ticker[ticker]
        c = universe_data[ticker]["Close"]
        cumret = np.log(c.shift(-horizon) / c)
        feat_vals = f[feature_cols].to_numpy(dtype=float)

        for t in range(contract.window_size, len(f) - horizon):
            w = feat_vals[t - contract.window_size : t]
            tgt = cumret.iloc[t - 1]
            if np.isfinite(tgt) and np.isfinite(w).all():
                pred = model.predict(np.expand_dims(w, axis=0))
                pt = float(pred.return_point[0]) if pred.return_point is not None else 0.0
                blended_pt = float(champion_decision.alpha * pt)
                transfer_cand_losses.append(abs(blended_pt - float(tgt)))
                transfer_base_losses.append(abs(float(tgt)))

    # Compute aggregate metrics
    c_arr = np.asarray(temp_cand_losses, dtype=float)
    b_arr = np.asarray(temp_base_losses, dtype=float)
    t_c_arr = np.asarray(transfer_cand_losses, dtype=float)
    t_b_arr = np.asarray(transfer_base_losses, dtype=float)

    temp_rel_mae = (
        float(np.mean(c_arr) / max(1e-12, float(np.mean(b_arr)))) if len(c_arr) > 0 else 1.0
    )
    temp_rel_rmse = (
        float(np.sqrt(np.mean(c_arr**2)) / max(1e-12, float(np.sqrt(np.mean(b_arr**2)))))
        if len(c_arr) > 0
        else 1.0
    )
    trans_rel_mae = (
        float(np.mean(t_c_arr) / max(1e-12, float(np.mean(t_b_arr)))) if len(t_c_arr) > 0 else 1.0
    )
    trans_rel_rmse = (
        float(np.sqrt(np.mean(t_c_arr**2)) / max(1e-12, float(np.sqrt(np.mean(t_b_arr**2)))))
        if len(t_c_arr) > 0
        else 1.0
    )

    # Direction metrics (overall prevalence vs non-neutral subset baseline)
    total_eval_rows = len(actual_ups)
    pos_prevalence = float(sum(actual_ups) / total_eval_rows) if total_eval_rows > 0 else 0.5

    non_neutrals = [(p, a) for p, a in zip(pred_ups, actual_ups, strict=False) if p is not None]
    direction_coverage = float(len(non_neutrals) / total_eval_rows) if total_eval_rows > 0 else 0.0

    if non_neutrals:
        subset_actual_ups = [a for _, a in non_neutrals]
        subset_pos_prev = float(sum(subset_actual_ups) / len(subset_actual_ups))
        majority_acc = max(subset_pos_prev, 1.0 - subset_pos_prev)
        temp_dir_acc = float(np.mean([p == a for p, a in non_neutrals]))
        dir_delta_vs_majority = float(temp_dir_acc - majority_acc)

        # Balanced accuracy on evaluated subset: 0.5 * (TPR + TNR)
        tp = sum(1 for p, a in non_neutrals if p is True and a is True)
        fn = sum(1 for p, a in non_neutrals if p is False and a is True)
        tn = sum(1 for p, a in non_neutrals if p is False and a is False)
        fp = sum(1 for p, a in non_neutrals if p is True and a is False)
        tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.5
        tnr = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.5
        balanced_acc = float(0.5 * (tpr + tnr))
    else:
        subset_pos_prev = 0.5
        majority_acc = 0.5
        temp_dir_acc = 0.5
        dir_delta_vs_majority = 0.0
        balanced_acc = 0.5

    # Probability status & Brier score
    prob_coverage = float(valid_prob_count / total_eval_rows) if total_eval_rows > 0 else 0.0
    if invalid_prob_count > 0:
        prob_status = "invalid"
        temp_brier = None
    elif valid_prob_count == 0:
        prob_status = "not_available"
        temp_brier = None
    elif valid_prob_count == total_eval_rows:
        prob_status = "evaluated"
        temp_brier = float(np.mean(brier_losses)) if brier_losses else None
    else:
        prob_status = "partial"
        temp_brier = float(np.mean(brier_losses)) if brier_losses else None

    # Evaluate gates
    passed_gates: list[str] = []
    failed_gates: list[str] = []

    if gate_config.require_temporal_relative_rmse:
        if temp_rel_rmse <= gate_config.max_temporal_relative_rmse:
            passed_gates.append(
                f"temporal_relative_rmse({temp_rel_rmse:.4f} <= {gate_config.max_temporal_relative_rmse:.4f})"
            )
        else:
            failed_gates.append(
                f"temporal_relative_rmse({temp_rel_rmse:.4f} > {gate_config.max_temporal_relative_rmse:.4f})"
            )

    if gate_config.require_temporal_relative_mae:
        if temp_rel_mae <= gate_config.max_temporal_relative_mae:
            passed_gates.append(
                f"temporal_relative_mae({temp_rel_mae:.4f} <= {gate_config.max_temporal_relative_mae:.4f})"
            )
        else:
            failed_gates.append(
                f"temporal_relative_mae({temp_rel_mae:.4f} > {gate_config.max_temporal_relative_mae:.4f})"
            )

    if gate_config.require_transfer_relative_rmse:
        if trans_rel_rmse <= gate_config.max_transfer_relative_rmse:
            passed_gates.append(
                f"transfer_relative_rmse({trans_rel_rmse:.4f} <= {gate_config.max_transfer_relative_rmse:.4f})"
            )
        else:
            failed_gates.append(
                f"transfer_relative_rmse({trans_rel_rmse:.4f} > {gate_config.max_transfer_relative_rmse:.4f})"
            )

    if gate_config.require_transfer_relative_mae:
        if trans_rel_mae <= gate_config.max_transfer_relative_mae:
            passed_gates.append(
                f"transfer_relative_mae({trans_rel_mae:.4f} <= {gate_config.max_transfer_relative_mae:.4f})"
            )
        else:
            failed_gates.append(
                f"transfer_relative_mae({trans_rel_mae:.4f} > {gate_config.max_transfer_relative_mae:.4f})"
            )

    if gate_config.require_direction_skill:
        if dir_delta_vs_majority >= gate_config.min_direction_accuracy_delta_vs_majority:
            passed_gates.append(
                f"direction_skill_delta({dir_delta_vs_majority:.4f} >= {gate_config.min_direction_accuracy_delta_vs_majority:.4f})"
            )
        else:
            failed_gates.append(
                f"direction_skill_delta({dir_delta_vs_majority:.4f} < {gate_config.min_direction_accuracy_delta_vs_majority:.4f})"
            )

    if gate_config.require_probabilistic_direction:
        if (
            prob_status == "evaluated"
            and temp_brier is not None
            and (
                gate_config.max_direction_brier is None
                or temp_brier <= gate_config.max_direction_brier
            )
        ):
            passed_gates.append(
                f"probabilistic_brier({temp_brier:.4f}"
                + (
                    f" <= {gate_config.max_direction_brier:.4f})"
                    if gate_config.max_direction_brier is not None
                    else ")"
                )
            )
        else:
            failed_gates.append(
                f"probabilistic_direction(status={prob_status}, coverage={prob_coverage:.2f}, required=True)"
            )

    decision = "pass" if len(failed_gates) == 0 else "fail"

    return CertificationDecision(
        horizon=horizon,
        candidate_name=cand_name,
        decision=decision,
        temporal_relative_rmse=temp_rel_rmse,
        temporal_relative_mae=temp_rel_mae,
        transfer_relative_rmse=trans_rel_rmse,
        transfer_relative_mae=trans_rel_mae,
        temporal_direction_acc=temp_dir_acc,
        positive_prevalence=pos_prevalence,
        subset_positive_prevalence=subset_pos_prev,
        majority_class_accuracy=majority_acc,
        direction_accuracy_delta_vs_majority=dir_delta_vs_majority,
        direction_coverage=direction_coverage,
        balanced_accuracy=balanced_acc,
        temporal_brier=temp_brier,
        direction_probability_status=prob_status,
        direction_probability_coverage=prob_coverage,
        passed_gates=passed_gates,
        failed_gates=failed_gates,
        gate_config=gate_config.to_dict(),
        certification_protocol_version=gate_config.protocol_version,
        temporal_sessions=len(temporal_holdout_dates),
        transfer_ticker_count=len(asset_transfer_tickers),
    )
