"""Protocol-only foundation for the Verified Model Release V12 generation.

The VMR-V12 package defines schemas and fail-closed gate evaluators.  It does
not acquire data, train models, retrieve public randomness, open a holdout, or
create a release.  Those operations belong to later, separately reviewed
boundaries.
"""

from .canonical import CanonicalizationError, canonical_bytes, canonical_digest, canonical_json
from .gates import (
    GateResult,
    ReleaseEligibility,
    evaluate_benchmark_integrity,
    evaluate_candidate_precommitment,
    evaluate_data_usage,
    evaluate_evaluation_validity,
    evaluate_release_eligibility,
    evaluate_release_provenance,
    evaluate_scientific_acceptance,
    reject_v11_2_evidence,
)
from .ledger import EvaluationLedger, EvaluationLedgerError
from .policies import (
    FALLBACK_TRIGGER_CONDITIONS,
    GATE_IDS,
    PUBLIC_RANDOMNESS_POLICY,
    TERMINAL_EVENT_POLICY,
    TERMINAL_EVENT_POLICY_VERSION,
    VMR_V12_PRODUCTION_ABSTENTION,
    VMR_V12_PROTOCOL,
    VMR_V12_SUCCESS_STATUS,
)
from .protocol import load_protocol_metadata, protocol_manifest
from .schemas import ProtocolValidationError

__all__ = [
    "GATE_IDS",
    "FALLBACK_TRIGGER_CONDITIONS",
    "PUBLIC_RANDOMNESS_POLICY",
    "TERMINAL_EVENT_POLICY",
    "TERMINAL_EVENT_POLICY_VERSION",
    "VMR_V12_PROTOCOL",
    "VMR_V12_PRODUCTION_ABSTENTION",
    "VMR_V12_SUCCESS_STATUS",
    "ProtocolValidationError",
    "CanonicalizationError",
    "load_protocol_metadata",
    "protocol_manifest",
    "GateResult",
    "ReleaseEligibility",
    "EvaluationLedger",
    "EvaluationLedgerError",
    "canonical_bytes",
    "canonical_digest",
    "canonical_json",
    "evaluate_data_usage",
    "evaluate_benchmark_integrity",
    "evaluate_candidate_precommitment",
    "evaluate_evaluation_validity",
    "evaluate_scientific_acceptance",
    "evaluate_release_provenance",
    "evaluate_release_eligibility",
    "reject_v11_2_evidence",
]
