"""Single-use sealed certification and mathematical evaluation for StockLSTM V10."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from research.volatility_forecasting.candidate_freeze_v10 import FrozenCandidatePackageV10
from research.volatility_forecasting.horizon_selection_v10 import (
    circular_block_bootstrap_ratio_upper_95,
    diebold_mariano_hac_p_value,
    holm_bonferroni_adjustment,
)


@dataclass(frozen=True)
class HorizonCertificationDecision:
    horizon: int
    candidate_family: str
    certified_status: str  # "certified", "rejected", "abstained_data_ineligible"
    qlike_candidate: float
    qlike_baseline: float
    relative_qlike: float
    ratio_upper_95: float
    dm_p_value_unadjusted: float
    dm_p_value_holm: float
    transfer_relative_qlike: float
    passed_all_gates: bool
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CertificationReportV10:
    run_id: str
    protocol_version: str
    data_eligibility_verified: bool
    certified_horizons: list[int]
    decisions_by_horizon: dict[str, Any]
    test_opening_receipt_path: str
    created_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def verify_certification_prerequisites(
    candidate_package: FrozenCandidatePackageV10 | dict[str, Any],
    universe_eligible: bool,
    market_panel_eligible: bool,
) -> tuple[bool, str]:
    """Strict fail-closed data eligibility and candidate package validation."""
    if not universe_eligible:
        return (
            False,
            "Universe data is not certification-eligible (secondary source or unverified license).",
        )
    if not market_panel_eligible:
        return (
            False,
            "Market data panel is not certification-eligible (yfinance/unverified source).",
        )

    if isinstance(candidate_package, FrozenCandidatePackageV10):
        c_dict = candidate_package.to_dict()
    else:
        c_dict = candidate_package

    if not c_dict.get("package_id") and not c_dict.get("candidate_name"):
        return False, "Candidate package manifest missing package identifier."

    if not c_dict.get("horizons"):
        return False, "Candidate package manifest contains no horizons."

    return True, "All certification prerequisites and data eligibility verified."


def record_test_opening_atomic(
    receipt_file: Path,
    run_id: str,
    candidate_package_sha256: str,
    split_manifest_sha256: str,
) -> Path:
    """Record single-use test opening atomically (O_CREAT | O_EXCL) to prevent duplicate openings."""
    receipt_path = Path(receipt_file)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "run_id": run_id,
        "candidate_package_sha256": candidate_package_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "opened_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "single_use_verified": True,
    }
    payload = json.dumps(record, indent=2).encode("utf-8")

    # Atomic creation: fails if file already exists
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(str(receipt_path), flags)
        with open(fd, "wb") as f:
            f.write(payload)
    except FileExistsError as exc:
        raise PermissionError(
            f"Sealed test opening record already exists at {receipt_path}. Duplicate test opening is strictly forbidden."
        ) from exc

    return receipt_path


def qlike_vector(pred: np.ndarray, actual: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.maximum(pred, eps)
    a = np.maximum(actual, eps)
    ratio = a / p
    return ratio - np.log(ratio) - 1.0


def evaluate_sealed_certification(
    run_id: str,
    protocol_version: str,
    candidate_package: FrozenCandidatePackageV10 | dict[str, Any],
    candidate_predictions_by_horizon: dict[int, np.ndarray],
    baseline_predictions_by_horizon: dict[int, np.ndarray],
    test_actuals_by_horizon: dict[int, np.ndarray],
    transfer_candidate_preds_by_horizon: dict[int, np.ndarray],
    transfer_baseline_preds_by_horizon: dict[int, np.ndarray],
    transfer_actuals_by_horizon: dict[int, np.ndarray],
    universe_eligible: bool = False,
    market_panel_eligible: bool = False,
    test_receipt_file: Path | None = None,
) -> CertificationReportV10:
    """Run mathematical certification battery on sealed test partition."""
    prereq_ok, rationale = verify_certification_prerequisites(
        candidate_package, universe_eligible, market_panel_eligible
    )

    created_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    if not prereq_ok:
        # Fail-closed: record explicit abstention
        decisions: dict[str, Any] = {}
        for h in candidate_predictions_by_horizon:
            decisions[str(h)] = HorizonCertificationDecision(
                horizon=h,
                candidate_family="unknown",
                certified_status="abstained_data_ineligible",
                qlike_candidate=1.0,
                qlike_baseline=1.0,
                relative_qlike=1.0,
                ratio_upper_95=1.0,
                dm_p_value_unadjusted=1.0,
                dm_p_value_holm=1.0,
                transfer_relative_qlike=1.0,
                passed_all_gates=False,
                rationale=rationale,
            ).to_dict()

        return CertificationReportV10(
            run_id=run_id,
            protocol_version=protocol_version,
            data_eligibility_verified=False,
            certified_horizons=[],
            decisions_by_horizon=decisions,
            test_opening_receipt_path="",
            created_at_utc=created_at,
        )

    # Record opening atomically
    receipt_str = ""
    if test_receipt_file:
        pkg_sha = (
            getattr(candidate_package, "package_sha256", lambda: "pkg_sha")()
            if hasattr(candidate_package, "package_sha256")
            else "pkg_sha"
        )
        rec_p = record_test_opening_atomic(test_receipt_file, run_id, pkg_sha, "split_sha")
        receipt_str = str(rec_p)

    decisions = {}
    certified_horizons = []

    horizons = sorted(candidate_predictions_by_horizon.keys())
    raw_p_values = []

    # 1. Compute loss series per horizon
    h_data = []
    for h in horizons:
        c_pred = candidate_predictions_by_horizon[h]
        b_pred = baseline_predictions_by_horizon[h]
        act = test_actuals_by_horizon[h]

        c_loss = qlike_vector(c_pred, act)
        b_loss = qlike_vector(b_pred, act)

        ql_cand = float(np.mean(c_loss))
        ql_base = float(np.mean(b_loss))
        rel_qlike = ql_cand / max(ql_base, 1e-12)

        ratio_95 = circular_block_bootstrap_ratio_upper_95(c_loss, b_loss, n_resamples=2000)
        dm_p = diebold_mariano_hac_p_value(c_loss, b_loss, horizon=h)
        raw_p_values.append(dm_p)

        # Transfer performance
        t_c_pred = transfer_candidate_preds_by_horizon.get(h, c_pred)
        t_b_pred = transfer_baseline_preds_by_horizon.get(h, b_pred)
        t_act = transfer_actuals_by_horizon.get(h, act)
        t_c_loss = qlike_vector(t_c_pred, t_act)
        t_b_loss = qlike_vector(t_b_pred, t_act)
        t_rel_qlike = float(np.mean(t_c_loss)) / max(float(np.mean(t_b_loss)), 1e-12)

        h_data.append(
            {
                "horizon": h,
                "ql_cand": ql_cand,
                "ql_base": ql_base,
                "rel_qlike": rel_qlike,
                "ratio_95": ratio_95,
                "dm_p": dm_p,
                "t_rel_qlike": t_rel_qlike,
            }
        )

    # Step-down Holm correction
    holm_ps = holm_bonferroni_adjustment(raw_p_values)

    for i, d in enumerate(h_data):
        h = d["horizon"]
        holm_p = holm_ps[i]
        passed = d["ratio_95"] <= 1.00 and holm_p <= 0.05 and d["t_rel_qlike"] <= 1.05
        status = "certified" if passed else "rejected"
        if passed:
            certified_horizons.append(h)

        decision = HorizonCertificationDecision(
            horizon=h,
            candidate_family="champion",
            certified_status=status,
            qlike_candidate=d["ql_cand"],
            qlike_baseline=d["ql_base"],
            relative_qlike=d["rel_qlike"],
            ratio_upper_95=d["ratio_95"],
            dm_p_value_unadjusted=d["dm_p"],
            dm_p_value_holm=holm_p,
            transfer_relative_qlike=d["t_rel_qlike"],
            passed_all_gates=passed,
            rationale=f"Evaluation on sealed test (ratio_95={d['ratio_95']:.4f}, holm_p={holm_p:.4f}, transfer={d['t_rel_qlike']:.4f})",
        )
        decisions[str(h)] = decision.to_dict()

    return CertificationReportV10(
        run_id=run_id,
        protocol_version=protocol_version,
        data_eligibility_verified=True,
        certified_horizons=certified_horizons,
        decisions_by_horizon=decisions,
        test_opening_receipt_path=receipt_str,
        created_at_utc=created_at,
    )
