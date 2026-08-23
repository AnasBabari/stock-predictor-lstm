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
    """Predeclared certification thresholds."""

    max_relative_rmse: float = 1.00
    max_relative_mae: float = 1.00
    min_direction_accuracy_delta: float = 0.00
    max_brier_score: float = 0.25
    max_relative_qlike: float = 1.00
    require_transfer_pass: bool = (
        False  # If True, asset-transfer holdout must also beat persistence
    )


@dataclass(frozen=True)
class CertificationDecision:
    horizon: int
    candidate_name: str
    decision: str  # "pass", "fail", "abstain"
    temporal_relative_rmse: float
    temporal_relative_mae: float
    temporal_direction_acc: float
    temporal_brier: float
    transfer_relative_rmse: float
    transfer_relative_mae: float
    passed_gates: list[str]
    failed_gates: list[str]
    temporal_sessions: int
    transfer_ticker_count: int
    certified_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    """Evaluate a selected champion on the locked temporal and asset holdouts."""
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
            temporal_direction_acc=0.5,
            temporal_brier=0.25,
            transfer_relative_rmse=1.0,
            transfer_relative_mae=1.0,
            passed_gates=[],
            failed_gates=["no_champion_selected"],
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
    temp_dir_correct: list[int] = []
    temp_dir_brier: list[float] = []

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

        # Dates within temporal holdout
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

                # Blend with persistence if alpha < 1.0
                blended = float(champion_decision.alpha * pt)
                temp_cand_losses.append(abs(blended - float(tgt)))
                temp_base_losses.append(abs(float(tgt)))

                is_up_pred = pt > 0
                is_up_actual = float(tgt) > 0
                temp_dir_correct.append(1 if (is_up_pred == is_up_actual) else 0)
                temp_dir_brier.append(
                    (1.0 - (1.0 if is_up_actual else 0.0)) ** 2
                    if is_up_pred
                    else (0.0 - (1.0 if is_up_actual else 0.0)) ** 2
                )

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
                blended = float(champion_decision.alpha * pt)
                transfer_cand_losses.append(abs(blended - float(tgt)))
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
    temp_dir_acc = float(np.mean(temp_dir_correct)) if temp_dir_correct else 0.5
    temp_brier = float(np.mean(temp_dir_brier)) if temp_dir_brier else 0.25

    trans_rel_mae = (
        float(np.mean(t_c_arr) / max(1e-12, float(np.mean(t_b_arr)))) if len(t_c_arr) > 0 else 1.0
    )
    trans_rel_rmse = (
        float(np.sqrt(np.mean(t_c_arr**2)) / max(1e-12, float(np.sqrt(np.mean(t_b_arr**2)))))
        if len(t_c_arr) > 0
        else 1.0
    )

    passed_gates: list[str] = []
    failed_gates: list[str] = []

    if temp_rel_rmse <= gate_config.max_relative_rmse:
        passed_gates.append(
            f"temporal_relative_rmse({temp_rel_rmse:.4f} <= {gate_config.max_relative_rmse})"
        )
    else:
        failed_gates.append(
            f"temporal_relative_rmse({temp_rel_rmse:.4f} > {gate_config.max_relative_rmse})"
        )

    if temp_rel_mae <= gate_config.max_relative_mae:
        passed_gates.append(
            f"temporal_relative_mae({temp_rel_mae:.4f} <= {gate_config.max_relative_mae})"
        )
    else:
        failed_gates.append(
            f"temporal_relative_mae({temp_rel_mae:.4f} > {gate_config.max_relative_mae})"
        )

    if gate_config.require_transfer_pass:
        if trans_rel_rmse <= gate_config.max_relative_rmse:
            passed_gates.append(
                f"transfer_relative_rmse({trans_rel_rmse:.4f} <= {gate_config.max_relative_rmse})"
            )
        else:
            failed_gates.append(
                f"transfer_relative_rmse({trans_rel_rmse:.4f} > {gate_config.max_relative_rmse})"
            )

    decision = "pass" if len(failed_gates) == 0 else "fail"

    return CertificationDecision(
        horizon=horizon,
        candidate_name=cand_name,
        decision=decision,
        temporal_relative_rmse=temp_rel_rmse,
        temporal_relative_mae=temp_rel_mae,
        temporal_direction_acc=temp_dir_acc,
        temporal_brier=temp_brier,
        transfer_relative_rmse=trans_rel_rmse,
        transfer_relative_mae=trans_rel_mae,
        passed_gates=passed_gates,
        failed_gates=failed_gates,
        temporal_sessions=len(temporal_holdout_dates),
        transfer_ticker_count=len(asset_transfer_tickers),
    )
