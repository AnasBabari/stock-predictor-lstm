"""Unit tests for SealedCertificationRunner."""

import numpy as np

from research.volatility_forecasting.certification_runner_v10 import (
    FrozenCandidatePackage,
    SealedCertificationRunner,
)


def test_certification_promotes_statistically_superior_candidate():
    pkg = FrozenCandidatePackage(
        family="causal_tcn",
        target_contract_version="price-return-distribution-v1",
        feature_schema_sha256="abc123schema",
        checkpoint_sha256="chk456hash",
        validation_relative_loss=0.88,
        dm_p_value_vs_baseline=0.008,
    )

    cand_losses = np.full(100, 0.015)
    base_losses = np.full(100, 0.020)

    report = SealedCertificationRunner.certify_candidate(pkg, cand_losses, base_losses)
    assert report.is_certified is True
    assert report.outcome == "certified_learned_candidate"
    assert report.relative_loss == 0.75
    assert len(report.report_digest_sha256) == 64


def test_certification_rejects_inferior_candidate():
    pkg = FrozenCandidatePackage(
        family="unanchored_lstm",
        target_contract_version="price-return-distribution-v1",
        feature_schema_sha256="abc123schema",
        checkpoint_sha256="chk456hash",
        validation_relative_loss=1.12,
        dm_p_value_vs_baseline=0.85,
    )

    cand_losses = np.full(100, 0.025)
    base_losses = np.full(100, 0.020)

    report = SealedCertificationRunner.certify_candidate(pkg, cand_losses, base_losses)
    assert report.is_certified is False
    assert report.outcome == "abstention"
    assert report.rejection_reason is not None
