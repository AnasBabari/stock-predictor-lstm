"""One-shot sealed certification runner for frozen forecast candidate packages."""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

OutcomeType = Literal[
    "certified_learned_candidate",
    "separately_certified_baseline",
    "abstention",
]


@dataclass(frozen=True)
class FrozenCandidatePackage:
    family: str
    target_contract_version: str
    feature_schema_sha256: str
    checkpoint_sha256: str
    validation_relative_loss: float
    dm_p_value_vs_baseline: float
    is_frozen: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SealedCertificationReport:
    candidate_family: str
    target_contract_version: str
    outcome: OutcomeType
    is_certified: bool
    test_loss_candidate: float
    test_loss_baseline: float
    relative_loss: float
    test_opened_timestamp: str
    report_digest_sha256: str
    rejection_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SealedCertificationRunner:
    """Evaluates frozen candidate package against sealed test partition with one-shot audit locking."""

    @staticmethod
    def certify_candidate(
        package: FrozenCandidatePackage,
        sealed_test_candidate_losses: np.ndarray,
        sealed_test_baseline_losses: np.ndarray,
        alpha_threshold: float = 0.05,
    ) -> SealedCertificationReport:
        if not package.is_frozen:
            raise ValueError("Candidate package must be frozen before sealed test evaluation.")

        t_opened = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        loss_cand = float(np.mean(sealed_test_candidate_losses))
        loss_base = float(np.mean(sealed_test_baseline_losses))
        rel_loss = loss_cand / max(loss_base, 1e-8)

        outcome: OutcomeType = "abstention"
        is_certified = False
        rejection_reason = None

        if rel_loss < 1.0 and package.dm_p_value_vs_baseline < alpha_threshold:
            outcome = "certified_learned_candidate"
            is_certified = True
        elif rel_loss >= 1.0:
            outcome = "abstention"
            is_certified = False
            rejection_reason = f"Candidate test loss {loss_cand:.6f} exceeded baseline {loss_base:.6f} (rel={rel_loss:.4f})"
        else:
            outcome = "separately_certified_baseline"
            is_certified = False
            rejection_reason = "Candidate failed statistical significance threshold vs baseline."

        # Cryptographic report digest
        hasher = hashlib.sha256()
        hasher.update(package.checkpoint_sha256.encode("utf-8"))
        hasher.update(package.feature_schema_sha256.encode("utf-8"))
        hasher.update(f"{outcome}_{rel_loss:.6f}_{t_opened}".encode())
        report_digest = hasher.hexdigest()

        return SealedCertificationReport(
            candidate_family=package.family,
            target_contract_version=package.target_contract_version,
            outcome=outcome,
            is_certified=is_certified,
            test_loss_candidate=round(loss_cand, 6),
            test_loss_baseline=round(loss_base, 6),
            relative_loss=round(rel_loss, 4),
            test_opened_timestamp=t_opened,
            report_digest_sha256=report_digest,
            rejection_reason=rejection_reason,
        )
