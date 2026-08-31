"""Immutable VMR-V12 identities and policy data.

These are protocol metadata, not evidence.  All eligibility decisions still
require later evidence artifacts and the gate evaluators in :mod:`gates`.
"""

from __future__ import annotations

VMR_V12_PROTOCOL = "VMR-V12"
VMR_V12_BENCHMARK = "USE64-HIST-v1"
VMR_V12_UNIVERSE_DESIGN = "HISTORICAL_FIXED_V1"
TERMINAL_EVENT_POLICY_VERSION = "TERMINAL_EVENT_POLICY_V1"
VMR_V12_PRODUCTION_ABSTENTION = "abstain_no_verified_release"
VMR_V12_SUCCESS_STATUS = "verified_release"
VMR_V12_SCHEMA_VERSION = 1
VMR_V12_INTEGRITY_STATEMENT = (
    "Automated integrity checks prove internal consistency and protocol compliance; "
    "they do not independently prove the truth of an untrusted historical source."
)
VMR_V12_SCOPE = {
    "data_acquisition": "out_of_scope_protocol_only",
    "training": "out_of_scope_protocol_only",
    "evaluation": "out_of_scope_protocol_only",
    "release": "out_of_scope_protocol_only",
    "production_status": VMR_V12_PRODUCTION_ABSTENTION,
}

GATE_IDS = (
    "G1_DATA_USAGE_COMPLIANT",
    "G2_BENCHMARK_INTEGRITY_VERIFIED",
    "G3_CANDIDATE_PRECOMMITTED",
    "G4_OFFICIAL_EVALUATION_VALID",
    "G5_SCIENTIFIC_ACCEPTANCE",
    "G6_RELEASE_PROVENANCE_VERIFIED",
)

FORBIDDEN_PROVENANCE_MARKERS = (
    "simulated",
    "demo",
    "placeholder",
    "permission_pending",
    "copied_diagnostic",
    "self_attested",
    "fake",
    "mock",
    "test_only",
    "locally_generated",
    "previously_opened",
    "invalidated",
    "simulated_rekor",
)

FALLBACK_TRIGGER_CONDITIONS = (
    "expected_pulse_unavailable_after_committed_deadline",
    "signature_invalid",
    "chain_verification_failure",
    "canonicalization_failure",
    "precommitted_provider_unavailable",
)

TERMINAL_EVENT_POLICY = {
    "version": TERMINAL_EVENT_POLICY_VERSION,
    "rules": (
        "Permanent security identity survives ticker and name changes.",
        "Splits and distributions follow the declared adjustment source.",
        "Securities remain benchmark members after historical selection.",
        "A security contributes observations only while valid data exist.",
        "Origins requiring unavailable post-terminal labels are right-censored.",
        "Missing post-delisting prices are never treated as unchanged or zero.",
        "Terminal returns are used only when supplied by an approved authoritative source.",
        "Missing vendor observations generate explicit quality flags.",
        "No security is removed merely because it later failed, merged, or delisted.",
        "Exclusions operate at affected observation/origin level, not retrospectively at security level.",
        "Counts and reasons for all censored observations are reported by security and event type.",
    ),
}

PUBLIC_RANDOMNESS_POLICY = {
    "primary": "drand",
    "secondary": "NIST Beacon V2",
    "primary_chain_or_root": "drand_public_chain_root",
    "secondary_chain_or_root": "nist_beacon_v2_public_root",
    "retrieval_endpoint_policy": "provider_endpoint_precommitted_in_candidate",
    "signature_verification": "verify_signed_pulse_against_provider_root",
    "canonical_pulse_encoding": "provider_payload_canonical_utf8_v1",
    "domain_separation": "VMR-V12/USE64-HIST-v1/partition-selection",
    "hash_to_selection": "sha256_digest_modulo_preconstructed_block_count_v1",
    "selection": "preconstructed_temporal_asset_blocks_only",
    "fallback_trigger_conditions": list(FALLBACK_TRIGGER_CONDITIONS),
    "timeout_wait_policy": "wait_until_committed_deadline_then_failover",
    "no_fallback_after_observation": True,
    "fallback_requires_precommitted_objective_failure": True,
    "unfavourable_partition_is_never_a_fallback": True,
    "individual_row_selection_forbidden": True,
}

VMR_V12_PROTOCOL_METADATA = {
    "schema_version": VMR_V12_SCHEMA_VERSION,
    "protocol": VMR_V12_PROTOCOL,
    "benchmark": VMR_V12_BENCHMARK,
    "universe_design": VMR_V12_UNIVERSE_DESIGN,
    "terminal_policy": TERMINAL_EVENT_POLICY_VERSION,
    "previous_generation": {
        "version": "V11.2",
        "status": "INVALIDATED_OPENED",
        "eligible_for_v12": False,
    },
    "production": {
        "default_status": VMR_V12_PRODUCTION_ABSTENTION,
        "success_status": VMR_V12_SUCCESS_STATUS,
    },
    "gates": list(GATE_IDS),
    "randomness": PUBLIC_RANDOMNESS_POLICY,
    "terminal_event_policy_rules": list(TERMINAL_EVENT_POLICY["rules"]),
    "scope": VMR_V12_SCOPE,
    "integrity_statement": VMR_V12_INTEGRITY_STATEMENT,
    "not_third_party_certification": True,
}
