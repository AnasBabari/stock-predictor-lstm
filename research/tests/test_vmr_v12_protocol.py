"""Adversarial protocol-boundary tests for VMR-V12.

These tests exercise schemas and gate composition only.  They intentionally do
not acquire data, fetch beacons, train models, open holdouts, or build a
release.
"""

from __future__ import annotations

from copy import deepcopy

import pytest
from volatility_forecasting.vmr_v12.canonical import CanonicalizationError, canonical_digest
from volatility_forecasting.vmr_v12.gates import (
    GateResult,
    evaluate_benchmark_integrity,
    evaluate_candidate_precommitment,
    evaluate_data_usage,
    evaluate_evaluation_validity,
    evaluate_release_eligibility,
    evaluate_release_provenance,
    evaluate_scientific_acceptance,
    reject_v11_2_evidence,
)
from volatility_forecasting.vmr_v12.ledger import EvaluationLedger, EvaluationLedgerError
from volatility_forecasting.vmr_v12.policies import GATE_IDS, PUBLIC_RANDOMNESS_POLICY
from volatility_forecasting.vmr_v12.protocol import (
    DEFAULT_PROTOCOL_PATH,
    load_protocol_metadata,
    protocol_manifest,
    validate_protocol_metadata,
)
from volatility_forecasting.vmr_v12.schemas import (
    ProtocolValidationError,
    validate_data_usage_record,
    validate_evaluation_record,
    validate_terminal_policy,
)

DIGEST = "a" * 64


def data_usage() -> dict[str, object]:
    return {
        "protocol": "VMR-V12",
        "provider": "licensed_vendor_v1",
        "dataset_id": "ohlcv-eod-v1",
        "acquired_at": "2017-12-29T22:00:00Z",
        "raw_artifact_sha256": DIGEST,
        "rights_basis": {
            "type": "provider_terms",
            "document_sha256": DIGEST,
            "terms_version": "2026-01",
            "retrieved_at": "2026-08-31T12:00:00Z",
        },
        "rights_review": {
            "reviewed_by": "repository_owner",
            "reviewed_at": "2026-08-31T12:05:00Z",
            "document_sha256": DIGEST,
            "terms_version": "2026-01",
            "scope": "VMR-V12",
        },
        "permitted_uses": {
            "historical_analysis": True,
            "model_training": True,
            "derived_metrics_publication": True,
            "prediction_deployment": True,
            "weights_distribution": True,
            "raw_data_redistribution": False,
        },
        "transformations": [],
        "derived_artifacts": [],
    }


def universe() -> dict[str, object]:
    return {
        "protocol": "VMR-V12",
        "benchmark": "USE64-HIST-v1",
        "universe_design": "HISTORICAL_FIXED_V1",
        "terminal_policy": "TERMINAL_EVENT_POLICY_V1",
        "terminal_event_policy": {
            "version": "TERMINAL_EVENT_POLICY_V1",
            "rules": [
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
            ],
        },
        "selection_timestamp": "2017-12-29T21:00:00Z",
        "evaluation_start_timestamp": "2018-01-02T14:30:00Z",
        "selection_input_hashes": [DIGEST],
        "selection_input_timestamps": ["2017-12-29T20:00:00Z"],
        "selection_code_commit": "selection-v1",
        "ordered_universe_hash": DIGEST,
        "permanent_security_ids": [f"SEC{i:02d}" for i in range(64)],
        "selection_criteria": {"fixed": True},
        "eligible_exchanges": ["XNYS", "XNAS"],
        "security_types": ["common_stock"],
        "liquidity_thresholds": {"minimum_sessions": 250},
        "history_thresholds": {"minimum_sessions": 250},
        "deduplication_rules": {"key": "permanent_security_id"},
        "corporate_action_policy": {"adjustment": "declared_source"},
        "missing_data_policy": {
            "action": "quality_flag_and_right_censor",
            "post_terminal_fill": "forbidden",
            "terminal_return_source": "approved_authoritative_only",
            "retrospective_security_deletion": "forbidden",
            "censor_counting": "by_security_and_event_type",
        },
        "survivorship_policy": {
            "future_survival_used_for_selection": False,
            "future_delisting_used_for_selection": False,
            "future_liquidity_used_for_selection": False,
            "future_membership_used_for_selection": False,
            "future_corporate_status_used_for_selection": False,
            "retain_post_selection_failures": True,
        },
        "integrity_policy": {
            "strictly_increasing_timestamps": True,
            "no_feature_after_origin": True,
            "train_only_transforms": True,
            "purge_sessions": 30,
            "embargo_sessions": 30,
            "max_horizon_sessions": 7,
            "asset_transfer_excluded_from_fit": True,
            "corporate_action_consistent": True,
            "explicit_missing_data_handling": True,
            "deterministic_reconstruction": True,
        },
    }


def candidate() -> dict[str, object]:
    catalogue = ["block-1", "block-2"]
    payload: dict[str, object] = {
        "protocol_version": "VMR-V12",
        "benchmark_version": "USE64-HIST-v1",
        "terminal_policy_version": "TERMINAL_EVENT_POLICY_V1",
        "model_implementation_version": "model-v12.0",
        "code_commit": "abc123",
        "training_data_sha256": DIGEST,
        "universe_sha256": DIGEST,
        "feature_schema_sha256": DIGEST,
        "target_schema_sha256": DIGEST,
        "split_construction_version": "split-v1",
        "selection_algorithm_version": "selection-v1",
        "training_configuration": {"epochs": 10, "seed": 42},
        "random_seeds": [41, 42, 43],
        "candidate_weights_sha256": DIGEST,
        "baseline_definitions": {
            "persistence": "persistence-v1",
            "HAR": "har-v1",
        },
        "acceptance_thresholds": {"relative_rmse": 0.95},
        "statistical_test_specification": {"bootstrap": {"replicates": 1000}},
        "release_format_version": "release-v1",
        "public_randomness_policy": deepcopy(PUBLIC_RANDOMNESS_POLICY),
        "evaluation_partition_catalogue": catalogue,
        "evaluation_partition_catalogue_sha256": canonical_digest(catalogue),
        "anchor": {"provider": "sigstore", "bundle_sha256": DIGEST, "verified": True},
    }
    payload["candidate_sha256"] = canonical_digest(payload)
    return payload


def evaluation(candidate_sha: str, record_type: str = "official") -> dict[str, object]:
    payload: dict[str, object] = {
        "candidate_sha256": candidate_sha,
        "protocol_version": "VMR-V12",
        "benchmark_version": "USE64-HIST-v1",
        "randomness_provider": "drand",
        "provider_chain_or_root": PUBLIC_RANDOMNESS_POLICY["primary_chain_or_root"],
        "randomness_identifier": "round-100",
        "raw_randomness_sha256": DIGEST,
        "canonical_randomness_sha256": DIGEST,
        "selected_partition": "block-2",
        "selection_algorithm_version": "selection-v1",
        "evaluation_code_sha256": DIGEST,
        "evaluation_environment": "ci-ubuntu-python311",
        "randomness_event_at": "2026-09-01T11:00:00Z",
        "started_at": "2026-09-01T12:00:00Z",
        "completed_at": "2026-09-01T12:10:00Z",
        "metrics_artifact_sha256": DIGEST,
        "record_type": record_type,
        "fallback_used": False,
    }
    if record_type == "reproduction":
        payload["reference_to_original_official_record"] = DIGEST
    payload["evaluation_record_sha256"] = canonical_digest(payload)
    return payload


def scientific(*, pass_model: bool = True) -> dict[str, object]:
    return {
        "protocol": "VMR-V12",
        "test_was_precommitted": True,
        "evaluation_complete": True,
        "no_leakage_detected": True,
        "model_beats_required_baseline": pass_model,
        "confidence_requirement_passed": True,
        "calibration_requirement_passed": True,
        "no_material_asset_group_regression": True,
        "required_baselines": [
            "persistence",
            "HAR",
            "EWMA",
            "strongest_eligible_development_baseline",
        ],
        "acceptance_thresholds": {"relative_rmse_max": 0.95},
        "metrics": {"rmse": 1.0, "relative_rmse": 0.9},
        "per_horizon": {"1": {"rmse": 1.0}},
        "subgroups": {"XNYS": {"rmse": 1.0}},
        "confidence_intervals": {"rmse": {"low": 0.8, "high": 1.2}},
        "paired_loss_comparisons": {"persistence": {"relative_rmse": 0.9}},
        "diebold_mariano": {"persistence": {"p_value": 0.04}},
        "multiple_horizon_correction": {"method": "holm", "alpha": 0.05},
        "coverage": {"prediction_interval": 0.9},
        "volatility_regimes": {"low": {"relative_rmse": 0.9}},
        "liquidity_groups": {"large": {"relative_rmse": 0.9}},
        "asset_transfer": {"held_out": {"relative_rmse": 0.9}},
    }


def training_and_release() -> tuple[dict[str, object], dict[str, object]]:
    training = {
        "protocol": "VMR-V12",
        "candidate_sha256": DIGEST,
        "git_commit": "abc123",
        "dirty_worktree": False,
        "environment_lock_sha256": DIGEST,
        "python_version": "3.11.9",
        "cuda_runtime_version": "12.4",
        "gpu_identity": "RTX",
        "training_command": "run-v12",
        "training_configuration_sha256": DIGEST,
        "training_dataset_sha256": DIGEST,
        "universe_sha256": DIGEST,
        "feature_schema_sha256": DIGEST,
        "seed": 42,
        "model_output_sha256": DIGEST,
        "training_log_sha256": DIGEST,
        "training_metrics_sha256": DIGEST,
        "started_at": "2026-08-31T10:00:00Z",
        "completed_at": "2026-08-31T11:00:00Z",
    }
    training["training_manifest_sha256"] = canonical_digest(training)
    release = {
        "protocol": "VMR-V12",
        "archive_sha256": DIGEST,
        "manifest_sha256": DIGEST,
        "model_output_sha256": DIGEST,
        "candidate_sha256": DIGEST,
        "training_manifest_sha256": training["training_manifest_sha256"],
        "training_dataset_sha256": training["training_dataset_sha256"],
        "universe_sha256": training["universe_sha256"],
        "feature_schema_sha256": training["feature_schema_sha256"],
        "repository": "AnasBabari/stock-predictor-lstm",
        "workflow": "release-v12",
        "commit": "abc123",
        "actions_sha_pinned": True,
        "job_permissions_minimal": True,
        "id_token_write_scoped": True,
        "production_environment_protected": True,
        "repository_identity_verified": True,
        "workflow_identity_verified": True,
        "untrusted_pr_signing": False,
        "attestation_verified": True,
        "simulated": False,
        "cosign_bundle_verified": True,
    }
    return training, release


def test_protocol_metadata_is_frozen_and_loaded() -> None:
    metadata = load_protocol_metadata(DEFAULT_PROTOCOL_PATH)
    assert metadata["protocol"] == "VMR-V12"
    assert metadata["benchmark"] == "USE64-HIST-v1"
    assert metadata["production"]["default_status"] == "abstain_no_verified_release"
    assert protocol_manifest()["terminal_event_policy"]["version"] == "TERMINAL_EVENT_POLICY_V1"


def test_protocol_metadata_rejects_scope_or_terminal_policy_drift() -> None:
    metadata = load_protocol_metadata(DEFAULT_PROTOCOL_PATH)
    metadata["scope"] = {**metadata["scope"], "training": "enabled"}
    with pytest.raises(ProtocolValidationError):
        validate_protocol_metadata(metadata)

    with pytest.raises(ProtocolValidationError):
        validate_terminal_policy({"version": "TERMINAL_EVENT_POLICY_V1", "rules": []})

    manifest = protocol_manifest()
    manifest["gates"].clear()
    assert protocol_manifest()["gates"] == list(GATE_IDS)
    manifest = protocol_manifest()
    assert manifest["protocol_sha256"] == canonical_digest(
        {key: value for key, value in manifest.items() if key != "protocol_sha256"}
    )


def test_canonical_serialization_is_order_independent_and_rejects_nonfinite() -> None:
    assert canonical_digest({"b": 2, "a": 1}) == canonical_digest({"a": 1, "b": 2})
    with pytest.raises(CanonicalizationError, match="non-finite"):
        canonical_digest({"x": float("nan")})
    with pytest.raises(CanonicalizationError, match="non-finite"):
        canonical_digest({"x": float("inf")})
    with pytest.raises(CanonicalizationError, match="unsupported"):
        canonical_digest({"x": b"bytes"})
    with pytest.raises(CanonicalizationError, match="UTF-8"):
        canonical_digest({"x": "\ud800"})


def test_gate_result_cannot_be_manually_constructed_or_overridden() -> None:
    with pytest.raises(TypeError, match="gate evaluator"):
        GateResult(GATE_IDS[0])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("permitted_uses", {"historical_analysis": True}),
        ("rights_basis", {"type": "provider_terms"}),
    ],
)
def test_g1_missing_rights_evidence_fails(field: str, value: object) -> None:
    payload = data_usage()
    payload[field] = value
    result = evaluate_data_usage(payload)
    assert not result.passed
    assert result.gate == "G1_DATA_USAGE_COMPLIANT"


def test_g1_standalone_verified_flag_is_not_evidence() -> None:
    payload = {"permission_status": "VERIFIED", "source_is_external": True}
    with pytest.raises(ProtocolValidationError):
        validate_data_usage_record(payload)
    assert not evaluate_data_usage(payload).passed


@pytest.mark.parametrize(
    ("use", "expected"),
    [
        ("model_training", False),
        ("prediction_deployment", False),
        ("historical_analysis", False),
    ],
)
def test_g1_required_use_rights_are_explicitly_enforced(use: str, expected: bool) -> None:
    payload = data_usage()
    payload["permitted_uses"][use] = expected  # type: ignore[index]
    assert not evaluate_data_usage(payload).passed


def test_g1_unknown_rights_basis_is_not_treated_as_permission() -> None:
    payload = data_usage()
    payload["rights_basis"]["type"] = "unknown_terms"  # type: ignore[index]
    assert not evaluate_data_usage(payload).passed


@pytest.mark.parametrize(
    "marker",
    [
        "simulated",
        "demo",
        "permission_pending",
        "self_attested",
        "copied_diagnostic",
        "fake",
        "test_only",
        "placeholder",
        "mock",
        "locally_generated",
        "previously_opened",
        "invalidated",
        "simulated_rekor",
    ],
)
def test_g1_forbidden_markers_fail(marker: str) -> None:
    payload = data_usage()
    payload["transformations"] = [{"note": marker}]
    assert not evaluate_data_usage(payload).passed


def test_g1_forbidden_marker_keys_fail_in_nested_evidence() -> None:
    payload = data_usage()
    payload["transformations"] = [{"simulated": False}]
    assert not evaluate_data_usage(payload).passed


def test_g2_historical_snapshot_and_integrity_contract() -> None:
    payload = universe()
    assert evaluate_benchmark_integrity(payload).passed
    payload["selection_timestamp"] = "2018-01-03T00:00:00Z"
    assert not evaluate_benchmark_integrity(payload).passed
    payload = universe()
    payload["permanent_security_ids"] = list(payload["permanent_security_ids"][:-1]) + ["SEC00"]  # type: ignore[index]
    assert not evaluate_benchmark_integrity(payload).passed


@pytest.mark.parametrize(
    "field",
    [
        "future_survival_used_for_selection",
        "future_delisting_used_for_selection",
        "future_liquidity_used_for_selection",
        "future_membership_used_for_selection",
        "future_corporate_status_used_for_selection",
    ],
)
def test_g2_future_selection_information_fails(field: str) -> None:
    payload = universe()
    payload["survivorship_policy"][field] = True  # type: ignore[index]
    assert not evaluate_benchmark_integrity(payload).passed


def test_g2_purge_and_embargo_must_cover_horizon() -> None:
    payload = universe()
    payload["integrity_policy"]["purge_sessions"] = 6  # type: ignore[index]
    assert not evaluate_benchmark_integrity(payload).passed


def test_g2_selection_inputs_and_terminal_policy_are_time_and_rule_bound() -> None:
    payload = universe()
    payload["selection_input_timestamps"] = ["2018-01-03T00:00:00Z"]
    assert not evaluate_benchmark_integrity(payload).passed

    payload = universe()
    payload["selection_input_timestamps"] = []
    assert not evaluate_benchmark_integrity(payload).passed

    payload = universe()
    payload["terminal_event_policy"]["rules"][5] = (
        "Missing post-delisting prices are carried forward."  # type: ignore[index]
    )
    assert not evaluate_benchmark_integrity(payload).passed

    payload = universe()
    payload["missing_data_policy"]["post_terminal_fill"] = "carry_forward"  # type: ignore[index]
    assert not evaluate_benchmark_integrity(payload).passed

    payload = universe()
    payload["missing_data_policy"]["post_terminal_fill"] = "zero"  # type: ignore[index]
    assert not evaluate_benchmark_integrity(payload).passed
    payload = universe()
    payload["integrity_policy"]["embargo_sessions"] = 6  # type: ignore[index]
    assert not evaluate_benchmark_integrity(payload).passed


def test_g3_manifest_hash_changes_when_threshold_changes() -> None:
    payload = candidate()
    assert evaluate_candidate_precommitment(payload).passed
    payload["acceptance_thresholds"] = {"relative_rmse": 0.99}
    assert not evaluate_candidate_precommitment(payload).passed


def test_g3_manifest_hash_changes_when_randomness_policy_changes() -> None:
    payload = candidate()
    payload["public_randomness_policy"] = {**PUBLIC_RANDOMNESS_POLICY, "primary": "other"}
    assert not evaluate_candidate_precommitment(payload).passed


def test_g3_partition_catalogue_is_unique_and_canonically_bound() -> None:
    payload = candidate()
    payload["evaluation_partition_catalogue"] = ["block-1", "block-1"]
    payload["evaluation_partition_catalogue_sha256"] = canonical_digest(
        payload["evaluation_partition_catalogue"]
    )
    payload["candidate_sha256"] = canonical_digest(
        {key: value for key, value in payload.items() if key != "candidate_sha256"}
    )
    assert not evaluate_candidate_precommitment(payload).passed


def test_g3_unset_threshold_fails_closed() -> None:
    payload = candidate()
    payload["acceptance_thresholds"] = {"relative_rmse": "UNSET"}
    payload["candidate_sha256"] = canonical_digest(
        {key: value for key, value in payload.items() if key != "candidate_sha256"}
    )
    assert not evaluate_candidate_precommitment(payload).passed


def test_g4_requires_one_official_and_preserves_reproduction() -> None:
    candidate_hash = candidate()["candidate_sha256"]
    official = evaluation(candidate_hash)
    reproduction = evaluation(candidate_hash, "reproduction")
    reproduction["reference_to_original_official_record"] = official["evaluation_record_sha256"]
    reproduction["evaluation_record_sha256"] = canonical_digest(
        {key: value for key, value in reproduction.items() if key != "evaluation_record_sha256"}
    )
    assert evaluate_evaluation_validity(
        [official, reproduction],
        candidate_sha256=candidate_hash,
        candidate_manifest=candidate(),
    ).passed
    assert not evaluate_evaluation_validity(
        [official, official], candidate_sha256=candidate_hash
    ).passed


def test_g4_reproduction_cannot_replace_official() -> None:
    record = evaluation(candidate()["candidate_sha256"], "reproduction")
    assert not evaluate_evaluation_validity([record]).passed


def test_g4_partition_event_and_fallback_are_bound_to_policy() -> None:
    cand = candidate()
    candidate_hash = cand["candidate_sha256"]
    official = evaluation(candidate_hash)
    official["selected_partition"] = "not-precommitted"
    official["evaluation_record_sha256"] = canonical_digest(
        {key: value for key, value in official.items() if key != "evaluation_record_sha256"}
    )
    result = evaluate_evaluation_validity(
        [official], candidate_sha256=candidate_hash, candidate_manifest=cand
    )
    assert not result.passed

    before_event = evaluation(candidate_hash)
    before_event["started_at"] = "2026-09-01T10:00:00Z"
    before_event["evaluation_record_sha256"] = canonical_digest(
        {key: value for key, value in before_event.items() if key != "evaluation_record_sha256"}
    )
    with pytest.raises(ProtocolValidationError, match="randomness event"):
        validate_evaluation_record(before_event)

    fallback = evaluation(candidate_hash)
    fallback["fallback_used"] = True
    fallback["fallback_reason"] = "expected_pulse_unavailable_after_committed_deadline"
    fallback["randomness_provider"] = "nist_beacon_v2"
    fallback["provider_chain_or_root"] = PUBLIC_RANDOMNESS_POLICY["secondary_chain_or_root"]
    fallback["evaluation_record_sha256"] = canonical_digest(
        {key: value for key, value in fallback.items() if key != "evaluation_record_sha256"}
    )
    assert evaluate_evaluation_validity(
        [fallback], candidate_sha256=candidate_hash, candidate_manifest=cand
    ).passed

    bad_fallback = dict(fallback)
    bad_fallback["fallback_reason"] = "unfavourable_partition"
    bad_fallback["evaluation_record_sha256"] = canonical_digest(
        {key: value for key, value in bad_fallback.items() if key != "evaluation_record_sha256"}
    )
    assert not evaluate_evaluation_validity(
        [bad_fallback], candidate_sha256=candidate_hash, candidate_manifest=cand
    ).passed

    bad_chain = evaluation(candidate_hash)
    bad_chain["provider_chain_or_root"] = "other-chain"
    bad_chain["evaluation_record_sha256"] = canonical_digest(
        {key: value for key, value in bad_chain.items() if key != "evaluation_record_sha256"}
    )
    assert not evaluate_evaluation_validity(
        [bad_chain], candidate_sha256=candidate_hash, candidate_manifest=cand
    ).passed


def test_g4_ledger_is_append_only_and_rejects_duplicate_records() -> None:
    record = evaluation(candidate()["candidate_sha256"])
    ledger = EvaluationLedger()
    appended = ledger.append(record)
    assert ledger.records == ()
    assert len(appended.records) == 1
    with pytest.raises(EvaluationLedgerError, match="already exists"):
        appended.append(record)
    detached = appended.records[0]
    detached["selected_partition"] = "mutated"
    assert appended.records[0]["selected_partition"] == "block-2"


def test_g5_missing_threshold_and_failed_model_are_not_accepted() -> None:
    assert evaluate_scientific_acceptance(scientific()).passed
    assert not evaluate_scientific_acceptance(scientific(pass_model=False)).passed
    incomplete = scientific()
    incomplete["required_baselines"] = ["persistence"]
    assert not evaluate_scientific_acceptance(incomplete).passed
    missing_threshold = scientific()
    missing_threshold.pop("acceptance_thresholds")
    assert not evaluate_scientific_acceptance(missing_threshold).passed


def test_g5_nested_nonfinite_metrics_fail_closed() -> None:
    payload = scientific()
    payload["per_horizon"]["1"]["rmse"] = float("nan")  # type: ignore[index]
    assert not evaluate_scientific_acceptance(payload).passed


def test_g6_dirty_worktree_and_simulated_release_fail() -> None:
    training, release = training_and_release()
    assert evaluate_release_provenance(training, release).passed
    training["dirty_worktree"] = True
    assert not evaluate_release_provenance(training, release).passed
    training["dirty_worktree"] = False
    release["simulated"] = True
    assert not evaluate_release_provenance(training, release).passed


@pytest.mark.parametrize(
    "field",
    [
        "actions_sha_pinned",
        "job_permissions_minimal",
        "id_token_write_scoped",
        "production_environment_protected",
        "repository_identity_verified",
        "workflow_identity_verified",
    ],
)
def test_g6_release_build_controls_fail_closed(field: str) -> None:
    training, release = training_and_release()
    release[field] = False
    assert not evaluate_release_provenance(training, release).passed


def test_g6_untrusted_pull_request_signing_is_forbidden() -> None:
    training, release = training_and_release()
    release["untrusted_pr_signing"] = True
    assert not evaluate_release_provenance(training, release).passed


def test_g6_training_output_must_bind_release() -> None:
    training, release = training_and_release()
    release["model_output_sha256"] = "b" * 64
    assert not evaluate_release_provenance(training, release).passed


@pytest.mark.parametrize("field", ["training_dataset_sha256", "universe_sha256"])
def test_g6_training_identity_changes_fail_closed(field: str) -> None:
    training, release = training_and_release()
    training[field] = "b" * 64
    training["training_manifest_sha256"] = canonical_digest(
        {key: value for key, value in training.items() if key != "training_manifest_sha256"}
    )
    release["training_manifest_sha256"] = training["training_manifest_sha256"]
    assert not evaluate_release_provenance(training, release).passed


def test_v11_evidence_rejected_even_when_renamed() -> None:
    with pytest.raises(ProtocolValidationError, match="V11.2"):
        reject_v11_2_evidence({"protocol": "VMR-V12", "generation": "V11.2"})
    with pytest.raises(ProtocolValidationError, match="VMR-V12"):
        reject_v11_2_evidence({"generation": "new"})


def test_six_of_six_required_and_manual_eligibility_is_ignored() -> None:
    cand = candidate()
    cand_hash = cand["candidate_sha256"]
    training, release = training_and_release()
    training["candidate_sha256"] = cand_hash
    training["training_manifest_sha256"] = canonical_digest(
        {key: value for key, value in training.items() if key != "training_manifest_sha256"}
    )
    release["candidate_sha256"] = cand_hash
    release["training_manifest_sha256"] = training["training_manifest_sha256"]
    results = [
        evaluate_data_usage(data_usage()),
        evaluate_benchmark_integrity(universe()),
        evaluate_candidate_precommitment(cand),
        evaluate_evaluation_validity(
            [evaluation(cand_hash)], candidate_sha256=cand_hash, candidate_manifest=cand
        ),
        evaluate_scientific_acceptance(scientific()),
        evaluate_release_provenance(training, release),
    ]
    eligibility = evaluate_release_eligibility(results)
    assert eligibility.release_eligible
    assert len(eligibility.gates) == 6
    failed = list(results)
    failed[4] = evaluate_scientific_acceptance(scientific(pass_model=False))
    assert not evaluate_release_eligibility(failed).release_eligible
