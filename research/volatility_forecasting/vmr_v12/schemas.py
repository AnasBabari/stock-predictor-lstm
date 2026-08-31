"""Strict, evidence-oriented VMR-V12 schema validators.

The validators deliberately do not claim that data or legal documents are
true.  They verify shape, identity, hashes, timestamps, and explicit policy
declarations so gate evaluators can fail closed when evidence is absent.
"""

from __future__ import annotations

import datetime as dt
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .canonical import CanonicalizationError, canonical_digest
from .policies import (
    FALLBACK_TRIGGER_CONDITIONS,
    FORBIDDEN_PROVENANCE_MARKERS,
    PUBLIC_RANDOMNESS_POLICY,
    TERMINAL_EVENT_POLICY,
    TERMINAL_EVENT_POLICY_VERSION,
    VMR_V12_BENCHMARK,
    VMR_V12_PROTOCOL,
    VMR_V12_UNIVERSE_DESIGN,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,159}$")
_FORBIDDEN_TOKENS = {marker.lower() for marker in FORBIDDEN_PROVENANCE_MARKERS}


class ProtocolValidationError(ValueError):
    """Raised when a VMR-V12 schema or invariant is invalid."""


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolValidationError(f"{label} must be an object")
    return dict(value)


def _text(value: object, label: str, *, identifier: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolValidationError(f"{label} must be a non-empty string")
    if value != value.strip():
        raise ProtocolValidationError(f"{label} must not have surrounding whitespace")
    result = value.strip()
    if identifier and not _IDENTIFIER.fullmatch(result):
        raise ProtocolValidationError(f"{label} is not a valid identifier")
    return result


def _digest(value: object, label: str) -> str:
    result = _text(value, label).removeprefix("sha256:")
    if not _SHA256.fullmatch(result):
        raise ProtocolValidationError(f"{label} must be a lowercase SHA-256 digest")
    return result


def _timestamp(value: object, label: str) -> dt.datetime:
    text = _text(value, label)
    try:
        parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProtocolValidationError(f"{label} must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ProtocolValidationError(f"{label} must include a timezone")
    return parsed.astimezone(dt.UTC)


def _bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise ProtocolValidationError(f"{label} must be a boolean")
    return value


def _list(value: object, label: str, *, non_empty: bool = True) -> list[Any]:
    if not isinstance(value, list) or (non_empty and not value):
        raise ProtocolValidationError(f"{label} must be a {'non-empty ' if non_empty else ''}list")
    return value


def _strict_keys(
    payload: Mapping[str, Any], required: set[str], allowed: set[str], label: str
) -> None:
    missing = sorted(required - set(payload))
    unknown = sorted(set(payload) - allowed)
    if missing:
        raise ProtocolValidationError(f"{label} missing required fields: {missing}")
    if unknown:
        raise ProtocolValidationError(f"{label} contains unsupported fields: {unknown}")


def _tokens(value: object) -> set[str]:
    if isinstance(value, Mapping):
        result: set[str] = set()
        for child in value.values():
            result.update(_tokens(child))
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result = set()
        for child in value:
            result.update(_tokens(child))
        return result
    text = str(value).lower()
    return {token for token in re.split(r"[^a-z0-9_]+", text) if token}


def _ensure_finite_json(value: object, label: str) -> None:
    """Reject non-JSON values and every non-finite number in an evidence tree."""

    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProtocolValidationError(f"{label} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str) or not key:
                raise ProtocolValidationError(f"{label} contains an invalid object key")
            _ensure_finite_json(child, f"{label}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for index, child in enumerate(value):
            _ensure_finite_json(child, f"{label}[{index}]")
        return
    raise ProtocolValidationError(f"{label} contains unsupported value {type(value).__name__}")


def _ensure_numeric_tree(value: object, label: str) -> None:
    """Require a non-empty tree whose leaves are finite numeric thresholds."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        if isinstance(value, Mapping):
            if not value:
                raise ProtocolValidationError(f"{label} cannot be empty")
            for key, child in value.items():
                if not isinstance(key, str) or not key.strip():
                    raise ProtocolValidationError(f"{label} contains an invalid key")
                _ensure_numeric_tree(child, f"{label}.{key}")
            return
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            if not value:
                raise ProtocolValidationError(f"{label} cannot be empty")
            for index, child in enumerate(value):
                _ensure_numeric_tree(child, f"{label}[{index}]")
            return
        raise ProtocolValidationError(f"{label} must contain only finite numeric values")
    if isinstance(value, float) and not math.isfinite(value):
        raise ProtocolValidationError(f"{label} contains a non-finite number")


def _canonical_digest(value: object, label: str) -> str:
    try:
        return canonical_digest(value)
    except CanonicalizationError as exc:
        raise ProtocolValidationError(f"{label} is not canonically serializable") from exc


def _marker_key_tokens(value: object, *, root: bool = False) -> set[str]:
    if isinstance(value, Mapping):
        result: set[str] = set()
        for key, child in value.items():
            if isinstance(key, str) and not (root and key.lower() == "simulated"):
                result.update(_tokens(key))
            result.update(_marker_key_tokens(child, root=False))
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result = set()
        for child in value:
            result.update(_marker_key_tokens(child, root=False))
        return result
    return set()


def reject_forbidden_markers(payload: object) -> None:
    found = sorted((_tokens(payload) | _marker_key_tokens(payload, root=True)) & _FORBIDDEN_TOKENS)
    if found:
        raise ProtocolValidationError(f"forbidden provenance markers: {found}")


def validate_data_usage_record(record: object) -> dict[str, Any]:
    payload = _object(record, "G1 data-usage record")
    reject_previous_generation(payload)
    required = {
        "protocol",
        "provider",
        "dataset_id",
        "acquired_at",
        "raw_artifact_sha256",
        "rights_basis",
        "rights_review",
        "permitted_uses",
        "transformations",
        "derived_artifacts",
    }
    _strict_keys(payload, required, required, "G1 data-usage record")
    reject_forbidden_markers(payload)
    if payload["protocol"] != VMR_V12_PROTOCOL:
        raise ProtocolValidationError("data-usage record must declare protocol VMR-V12")
    _text(payload["provider"], "provider", identifier=True)
    _text(payload["dataset_id"], "dataset_id", identifier=True)
    _timestamp(payload["acquired_at"], "acquired_at")
    _digest(payload["raw_artifact_sha256"], "raw_artifact_sha256")

    rights_basis = _object(payload["rights_basis"], "rights_basis")
    _strict_keys(
        rights_basis,
        {"type", "document_sha256", "terms_version", "retrieved_at"},
        {"type", "document_sha256", "terms_version", "retrieved_at"},
        "rights_basis",
    )
    _text(rights_basis["type"], "rights_basis.type", identifier=True)
    if rights_basis["type"] not in {
        "provider_terms",
        "license_agreement",
        "vendor_contract",
        "public_domain",
    }:
        raise ProtocolValidationError("rights_basis.type is not a recognized rights source")
    _digest(rights_basis["document_sha256"], "rights_basis.document_sha256")
    _text(rights_basis["terms_version"], "rights_basis.terms_version")
    rights_retrieved_at = _timestamp(rights_basis["retrieved_at"], "rights_basis.retrieved_at")

    review = _object(payload["rights_review"], "rights_review")
    _strict_keys(
        review,
        {"reviewed_by", "reviewed_at", "document_sha256", "terms_version", "scope"},
        {"reviewed_by", "reviewed_at", "document_sha256", "terms_version", "scope"},
        "rights_review",
    )
    _text(review["reviewed_by"], "rights_review.reviewed_by")
    reviewed_at = _timestamp(review["reviewed_at"], "rights_review.reviewed_at")
    if reviewed_at < rights_retrieved_at:
        raise ProtocolValidationError("rights review cannot predate rights-basis retrieval")
    _digest(review["document_sha256"], "rights_review.document_sha256")
    _text(review["terms_version"], "rights_review.terms_version")
    if _text(review["scope"], "rights_review.scope") != VMR_V12_PROTOCOL:
        raise ProtocolValidationError("rights_review.scope must be VMR-V12")
    if review["document_sha256"] != rights_basis["document_sha256"]:
        raise ProtocolValidationError("rights review must bind the rights-basis document")
    if review["terms_version"] != rights_basis["terms_version"]:
        raise ProtocolValidationError("rights review must bind the rights-basis terms version")

    uses = _object(payload["permitted_uses"], "permitted_uses")
    expected_uses = {
        "historical_analysis",
        "model_training",
        "derived_metrics_publication",
        "prediction_deployment",
        "weights_distribution",
        "raw_data_redistribution",
    }
    _strict_keys(uses, expected_uses, expected_uses, "permitted_uses")
    for name, value in uses.items():
        _bool(value, f"permitted_uses.{name}")
    for name in ("historical_analysis", "model_training", "prediction_deployment"):
        if uses[name] is not True:
            raise ProtocolValidationError(f"required usage is not permitted: {name}")

    transformations = _list(payload["transformations"], "transformations", non_empty=False)
    derived = _list(payload["derived_artifacts"], "derived_artifacts", non_empty=False)
    for index, item in enumerate(transformations + derived):
        if not isinstance(item, Mapping):
            raise ProtocolValidationError(f"evidence artifact entry {index} must be an object")
    return payload


def validate_terminal_policy(policy: object) -> dict[str, Any]:
    payload = _object(policy, "terminal-event policy")
    _strict_keys(payload, {"version", "rules"}, {"version", "rules"}, "terminal-event policy")
    if payload.get("version") != TERMINAL_EVENT_POLICY_VERSION:
        raise ProtocolValidationError("terminal-event policy version is unsupported")
    rules = _list(payload.get("rules"), "terminal-event policy rules")
    if tuple(rules) != TERMINAL_EVENT_POLICY["rules"]:
        raise ProtocolValidationError("terminal-event policy rules do not match the frozen policy")
    return payload


def validate_universe_manifest(manifest: object) -> dict[str, Any]:
    payload = _object(manifest, "G2 universe manifest")
    reject_previous_generation(payload)
    required = {
        "protocol",
        "benchmark",
        "universe_design",
        "terminal_policy",
        "terminal_event_policy",
        "selection_timestamp",
        "evaluation_start_timestamp",
        "selection_input_hashes",
        "selection_input_timestamps",
        "selection_code_commit",
        "ordered_universe_hash",
        "permanent_security_ids",
        "selection_criteria",
        "eligible_exchanges",
        "security_types",
        "liquidity_thresholds",
        "history_thresholds",
        "deduplication_rules",
        "corporate_action_policy",
        "missing_data_policy",
        "survivorship_policy",
        "integrity_policy",
    }
    _strict_keys(payload, required, required, "G2 universe manifest")
    if payload["protocol"] != VMR_V12_PROTOCOL or payload["benchmark"] != VMR_V12_BENCHMARK:
        raise ProtocolValidationError("universe is not bound to VMR-V12/USE64-HIST-v1")
    if payload["universe_design"] != VMR_V12_UNIVERSE_DESIGN:
        raise ProtocolValidationError("universe must use HISTORICAL_FIXED_V1")
    if payload["terminal_policy"] != TERMINAL_EVENT_POLICY_VERSION:
        raise ProtocolValidationError("universe terminal policy is not TERMINAL_EVENT_POLICY_V1")
    validate_terminal_policy(payload["terminal_event_policy"])
    selection_time = _timestamp(payload["selection_timestamp"], "selection_timestamp")
    evaluation_start = _timestamp(
        payload["evaluation_start_timestamp"], "evaluation_start_timestamp"
    )
    if selection_time > evaluation_start:
        raise ProtocolValidationError("selection_timestamp must not be after evaluation start")
    hashes = _list(payload["selection_input_hashes"], "selection_input_hashes")
    for index, value in enumerate(hashes):
        _digest(value, f"selection_input_hashes[{index}]")
    input_timestamps = _list(payload["selection_input_timestamps"], "selection_input_timestamps")
    if len(input_timestamps) != len(hashes):
        raise ProtocolValidationError(
            "each selection input hash must have a corresponding as-of timestamp"
        )
    for index, value in enumerate(input_timestamps):
        if _timestamp(value, f"selection_input_timestamps[{index}]") > selection_time:
            raise ProtocolValidationError(
                "selection input information must not be newer than selection_timestamp"
            )
    _text(payload["selection_code_commit"], "selection_code_commit", identifier=True)
    _digest(payload["ordered_universe_hash"], "ordered_universe_hash")
    securities = _list(payload["permanent_security_ids"], "permanent_security_ids")
    if len(securities) != 64 or not all(
        isinstance(item, str) and item == item.strip() and _IDENTIFIER.fullmatch(item)
        for item in securities
    ):
        raise ProtocolValidationError("USE64-HIST-v1 requires exactly 64 security identifiers")
    if len(set(securities)) != len(securities):
        raise ProtocolValidationError("permanent security identifiers must be unique")
    for field in (
        "selection_criteria",
        "liquidity_thresholds",
        "history_thresholds",
        "deduplication_rules",
        "corporate_action_policy",
        "missing_data_policy",
        "integrity_policy",
    ):
        if not isinstance(payload[field], Mapping):
            raise ProtocolValidationError(f"{field} must be an object")
    for field in ("eligible_exchanges", "security_types"):
        values = _list(payload[field], field)
        if not all(isinstance(item, str) and item == item.strip() and item for item in values):
            raise ProtocolValidationError(f"{field} entries must be non-empty strings")

    deduplication = _object(payload["deduplication_rules"], "deduplication_rules")
    if deduplication.get("key") != "permanent_security_id":
        raise ProtocolValidationError("deduplication must use permanent security identifiers")

    corporate_actions = _object(payload["corporate_action_policy"], "corporate_action_policy")
    _strict_keys(
        corporate_actions,
        {"adjustment"},
        {"adjustment"},
        "corporate_action_policy",
    )
    if corporate_actions["adjustment"] not in {
        "declared_source",
        "authoritative_adjusted_ohlcv",
    }:
        raise ProtocolValidationError("corporate-action adjustment source is not approved")

    missing_data = _object(payload["missing_data_policy"], "missing_data_policy")
    _strict_keys(
        missing_data,
        {
            "action",
            "post_terminal_fill",
            "terminal_return_source",
            "retrospective_security_deletion",
            "censor_counting",
        },
        {
            "action",
            "post_terminal_fill",
            "terminal_return_source",
            "retrospective_security_deletion",
            "censor_counting",
        },
        "missing_data_policy",
    )
    expected_missing_data = {
        "action": "quality_flag_and_right_censor",
        "post_terminal_fill": "forbidden",
        "terminal_return_source": "approved_authoritative_only",
        "retrospective_security_deletion": "forbidden",
        "censor_counting": "by_security_and_event_type",
    }
    if missing_data != expected_missing_data:
        raise ProtocolValidationError("missing-data policy does not implement terminal-event rules")

    survivorship = _object(payload["survivorship_policy"], "survivorship_policy")
    expected_survivorship = {
        "future_survival_used_for_selection",
        "future_delisting_used_for_selection",
        "future_liquidity_used_for_selection",
        "future_membership_used_for_selection",
        "future_corporate_status_used_for_selection",
        "retain_post_selection_failures",
    }
    _strict_keys(survivorship, expected_survivorship, expected_survivorship, "survivorship_policy")
    for field in expected_survivorship:
        _bool(survivorship[field], f"survivorship_policy.{field}")
    for field in expected_survivorship - {"retain_post_selection_failures"}:
        if survivorship[field] is not False:
            raise ProtocolValidationError(f"{field} must be false")
    if survivorship["retain_post_selection_failures"] is not True:
        raise ProtocolValidationError("post-selection failures must be retained")

    integrity = payload["integrity_policy"]
    required_integrity = {
        "strictly_increasing_timestamps",
        "no_feature_after_origin",
        "train_only_transforms",
        "purge_sessions",
        "embargo_sessions",
        "max_horizon_sessions",
        "asset_transfer_excluded_from_fit",
        "corporate_action_consistent",
        "explicit_missing_data_handling",
        "deterministic_reconstruction",
    }
    _strict_keys(integrity, required_integrity, required_integrity, "integrity_policy")
    for field in required_integrity - {
        "purge_sessions",
        "embargo_sessions",
        "max_horizon_sessions",
    }:
        _bool(integrity[field], f"integrity_policy.{field}")
        if integrity[field] is not True:
            raise ProtocolValidationError(f"integrity_policy.{field} must be true")
    for field in ("purge_sessions", "embargo_sessions", "max_horizon_sessions"):
        value = integrity[field]
        if type(value) is not int or value < 1:
            raise ProtocolValidationError(f"integrity_policy.{field} must be a positive integer")
    if integrity["purge_sessions"] < integrity["max_horizon_sessions"]:
        raise ProtocolValidationError("purge window must cover maximum horizon")
    if integrity["embargo_sessions"] < integrity["max_horizon_sessions"]:
        raise ProtocolValidationError("embargo window must cover maximum horizon")
    return payload


def validate_candidate_manifest(manifest: object) -> dict[str, Any]:
    payload = _object(manifest, "G3 candidate manifest")
    reject_previous_generation(payload)
    required = {
        "protocol_version",
        "benchmark_version",
        "terminal_policy_version",
        "model_implementation_version",
        "code_commit",
        "training_data_sha256",
        "universe_sha256",
        "feature_schema_sha256",
        "target_schema_sha256",
        "split_construction_version",
        "selection_algorithm_version",
        "training_configuration",
        "random_seeds",
        "candidate_weights_sha256",
        "baseline_definitions",
        "acceptance_thresholds",
        "statistical_test_specification",
        "release_format_version",
        "public_randomness_policy",
        "evaluation_partition_catalogue",
        "evaluation_partition_catalogue_sha256",
        "candidate_sha256",
        "anchor",
    }
    _strict_keys(payload, required, required, "G3 candidate manifest")
    reject_forbidden_markers(payload)
    if payload["protocol_version"] != VMR_V12_PROTOCOL:
        raise ProtocolValidationError("candidate protocol_version must be VMR-V12")
    if payload["benchmark_version"] != VMR_V12_BENCHMARK:
        raise ProtocolValidationError("candidate benchmark_version is unsupported")
    if payload["terminal_policy_version"] != TERMINAL_EVENT_POLICY_VERSION:
        raise ProtocolValidationError("candidate terminal policy is unsupported")
    for field in (
        "model_implementation_version",
        "code_commit",
        "split_construction_version",
        "selection_algorithm_version",
        "release_format_version",
    ):
        _text(payload[field], field, identifier=True)
    for field in (
        "training_data_sha256",
        "universe_sha256",
        "feature_schema_sha256",
        "target_schema_sha256",
        "candidate_weights_sha256",
        "evaluation_partition_catalogue_sha256",
    ):
        _digest(payload[field], field)
    if not isinstance(payload["training_configuration"], Mapping):
        raise ProtocolValidationError("training_configuration must be an object")
    _ensure_finite_json(payload["training_configuration"], "training_configuration")
    seeds = _list(payload["random_seeds"], "random_seeds")
    if not seeds or not all(type(seed) is int and seed >= 0 for seed in seeds):
        raise ProtocolValidationError("random_seeds must contain non-negative integers")
    if len(set(seeds)) != len(seeds):
        raise ProtocolValidationError("random_seeds must be unique")
    for field in ("baseline_definitions", "statistical_test_specification"):
        if not isinstance(payload[field], Mapping) or not payload[field]:
            raise ProtocolValidationError(f"{field} must be a non-empty object")
        _ensure_finite_json(payload[field], field)
    catalogue = _list(payload["evaluation_partition_catalogue"], "evaluation_partition_catalogue")
    if not all(isinstance(item, str) and _IDENTIFIER.fullmatch(item) for item in catalogue):
        raise ProtocolValidationError("evaluation partition catalogue entries are invalid")
    if len(set(catalogue)) != len(catalogue):
        raise ProtocolValidationError("evaluation partition catalogue entries must be unique")
    if (
        _canonical_digest(catalogue, "evaluation_partition_catalogue")
        != payload["evaluation_partition_catalogue_sha256"]
    ):
        raise ProtocolValidationError("evaluation partition catalogue hash does not match contents")
    thresholds = _object(payload["acceptance_thresholds"], "acceptance_thresholds")
    if not thresholds or "unset" in {token for token in _tokens(thresholds)}:
        raise ProtocolValidationError("acceptance thresholds cannot be missing or UNSET")
    _ensure_numeric_tree(thresholds, "acceptance_thresholds")
    randomness = _object(payload["public_randomness_policy"], "public_randomness_policy")
    if randomness != PUBLIC_RANDOMNESS_POLICY:
        raise ProtocolValidationError("candidate randomness policy differs from VMR-V12 policy")
    anchor = _object(payload["anchor"], "candidate anchor")
    _strict_keys(
        anchor,
        {"provider", "bundle_sha256", "verified"},
        {"provider", "bundle_sha256", "verified"},
        "candidate anchor",
    )
    _text(anchor["provider"], "anchor.provider", identifier=True)
    if anchor["provider"].lower() in {"local", "self", "repository_owner"}:
        raise ProtocolValidationError("candidate anchor must identify an external provider")
    _digest(anchor["bundle_sha256"], "anchor.bundle_sha256")
    if _bool(anchor["verified"], "anchor.verified") is not True:
        raise ProtocolValidationError("candidate must have a verified external anchor")
    expected = dict(payload)
    declared = expected.pop("candidate_sha256")
    if _digest(declared, "candidate_sha256") != _canonical_digest(expected, "candidate manifest"):
        raise ProtocolValidationError("candidate_sha256 does not match canonical manifest contents")
    return payload


def validate_evaluation_record(record: object) -> dict[str, Any]:
    payload = _object(record, "G4 evaluation record")
    reject_previous_generation(payload)
    required = {
        "candidate_sha256",
        "protocol_version",
        "benchmark_version",
        "randomness_provider",
        "provider_chain_or_root",
        "randomness_identifier",
        "raw_randomness_sha256",
        "canonical_randomness_sha256",
        "selected_partition",
        "selection_algorithm_version",
        "evaluation_code_sha256",
        "evaluation_environment",
        "randomness_event_at",
        "started_at",
        "completed_at",
        "metrics_artifact_sha256",
        "evaluation_record_sha256",
        "record_type",
        "fallback_used",
    }
    allowed = required | {"reference_to_original_official_record", "fallback_reason"}
    _strict_keys(payload, required, allowed, "G4 evaluation record")
    reject_forbidden_markers(payload)
    if (
        payload["protocol_version"] != VMR_V12_PROTOCOL
        or payload["benchmark_version"] != VMR_V12_BENCHMARK
    ):
        raise ProtocolValidationError("evaluation record is not bound to VMR-V12 benchmark")
    _digest(payload["candidate_sha256"], "candidate_sha256")
    _text(payload["randomness_provider"], "randomness_provider", identifier=True)
    if payload["randomness_provider"] not in {"drand", "nist_beacon_v2"}:
        raise ProtocolValidationError("randomness provider is not in the frozen provider policy")
    _text(payload["provider_chain_or_root"], "provider_chain_or_root", identifier=True)
    _text(payload["randomness_identifier"], "randomness_identifier", identifier=True)
    _digest(payload["raw_randomness_sha256"], "raw_randomness_sha256")
    _digest(payload["canonical_randomness_sha256"], "canonical_randomness_sha256")
    _text(payload["selected_partition"], "selected_partition", identifier=True)
    _text(payload["selection_algorithm_version"], "selection_algorithm_version", identifier=True)
    _digest(payload["evaluation_code_sha256"], "evaluation_code_sha256")
    _text(payload["evaluation_environment"], "evaluation_environment", identifier=True)
    randomness_event = _timestamp(payload["randomness_event_at"], "randomness_event_at")
    started = _timestamp(payload["started_at"], "started_at")
    completed = _timestamp(payload["completed_at"], "completed_at")
    if started < randomness_event:
        raise ProtocolValidationError("evaluation must start after the committed randomness event")
    if completed < started:
        raise ProtocolValidationError("completed_at must not precede started_at")
    _digest(payload["metrics_artifact_sha256"], "metrics_artifact_sha256")
    declared_digest = _digest(payload["evaluation_record_sha256"], "evaluation_record_sha256")
    unsigned = dict(payload)
    unsigned.pop("evaluation_record_sha256")
    if declared_digest != _canonical_digest(unsigned, "evaluation record"):
        raise ProtocolValidationError("evaluation_record_sha256 does not match record contents")
    record_type = payload["record_type"]
    if record_type not in {"official", "reproduction"}:
        raise ProtocolValidationError("record_type must be official or reproduction")
    fallback_used = _bool(payload["fallback_used"], "fallback_used")
    fallback_reason = payload.get("fallback_reason")
    if fallback_used:
        if (
            not isinstance(fallback_reason, str)
            or fallback_reason not in FALLBACK_TRIGGER_CONDITIONS
        ):
            raise ProtocolValidationError(
                "fallback reason is not an objective precommitted failure"
            )
        if payload["randomness_provider"] != "nist_beacon_v2":
            raise ProtocolValidationError("fallback evaluations must use the secondary provider")
    elif payload["randomness_provider"] != "drand":
        raise ProtocolValidationError("non-fallback evaluations must use the primary provider")
    elif fallback_reason is not None:
        raise ProtocolValidationError(
            "fallback_reason must be absent or null when fallback_used is false"
        )
    if record_type == "reproduction":
        reference = payload.get("reference_to_original_official_record")
        _digest(reference, "reference_to_original_official_record")
    elif "reference_to_original_official_record" in payload:
        raise ProtocolValidationError("official record cannot reference another official record")
    return payload


def validate_scientific_acceptance(record: object) -> dict[str, Any]:
    payload = _object(record, "G5 scientific acceptance")
    reject_previous_generation(payload)
    required = {
        "protocol",
        "test_was_precommitted",
        "evaluation_complete",
        "no_leakage_detected",
        "model_beats_required_baseline",
        "confidence_requirement_passed",
        "calibration_requirement_passed",
        "no_material_asset_group_regression",
        "required_baselines",
        "acceptance_thresholds",
        "metrics",
        "per_horizon",
        "subgroups",
        "confidence_intervals",
        "paired_loss_comparisons",
        "diebold_mariano",
        "multiple_horizon_correction",
        "coverage",
        "volatility_regimes",
        "liquidity_groups",
        "asset_transfer",
    }
    _strict_keys(payload, required, required, "G5 scientific acceptance")
    if payload["protocol"] != VMR_V12_PROTOCOL:
        raise ProtocolValidationError("scientific acceptance must declare protocol VMR-V12")
    for field in required - {
        "protocol",
        "required_baselines",
        "acceptance_thresholds",
        "metrics",
        "per_horizon",
        "subgroups",
        "confidence_intervals",
        "paired_loss_comparisons",
        "diebold_mariano",
        "multiple_horizon_correction",
        "coverage",
        "volatility_regimes",
        "liquidity_groups",
        "asset_transfer",
    }:
        _bool(payload[field], f"scientific_acceptance.{field}")
    baselines = _list(payload["required_baselines"], "required_baselines")
    required_baseline_names = {
        "persistence",
        "HAR",
        "EWMA",
        "strongest_eligible_development_baseline",
    }
    if not required_baseline_names.issubset(set(baselines)):
        raise ProtocolValidationError("required baseline set is incomplete")
    thresholds = _object(
        payload["acceptance_thresholds"], "scientific_acceptance.acceptance_thresholds"
    )
    if "unset" in _tokens(thresholds):
        raise ProtocolValidationError("scientific acceptance thresholds cannot be UNSET")
    _ensure_numeric_tree(thresholds, "scientific_acceptance.acceptance_thresholds")
    metrics = _object(payload["metrics"], "scientific_acceptance.metrics")
    if not metrics:
        raise ProtocolValidationError("scientific acceptance metrics are missing")
    _ensure_numeric_tree(metrics, "scientific_acceptance.metrics")
    for field in ("per_horizon", "subgroups"):
        if not isinstance(payload[field], Mapping) or not payload[field]:
            raise ProtocolValidationError(f"{field} evidence is missing")
        _ensure_numeric_tree(payload[field], f"scientific_acceptance.{field}")
    for field in (
        "confidence_intervals",
        "paired_loss_comparisons",
        "diebold_mariano",
        "multiple_horizon_correction",
        "coverage",
        "volatility_regimes",
        "liquidity_groups",
        "asset_transfer",
    ):
        if not isinstance(payload[field], Mapping) or not payload[field]:
            raise ProtocolValidationError(f"{field} evidence is missing")
        _ensure_finite_json(payload[field], f"scientific_acceptance.{field}")
    return payload


def validate_training_provenance(record: object) -> dict[str, Any]:
    payload = _object(record, "G6 training provenance")
    reject_previous_generation(payload)
    required = {
        "protocol",
        "candidate_sha256",
        "git_commit",
        "dirty_worktree",
        "environment_lock_sha256",
        "python_version",
        "cuda_runtime_version",
        "gpu_identity",
        "training_command",
        "training_configuration_sha256",
        "training_dataset_sha256",
        "universe_sha256",
        "feature_schema_sha256",
        "seed",
        "model_output_sha256",
        "training_log_sha256",
        "training_metrics_sha256",
        "training_manifest_sha256",
        "started_at",
        "completed_at",
    }
    _strict_keys(payload, required, required, "G6 training provenance")
    if payload["protocol"] != VMR_V12_PROTOCOL:
        raise ProtocolValidationError("training provenance must declare protocol VMR-V12")
    _digest(payload["candidate_sha256"], "candidate_sha256")
    _text(payload["git_commit"], "git_commit", identifier=True)
    if _bool(payload["dirty_worktree"], "dirty_worktree") is not False:
        raise ProtocolValidationError("dirty training worktree is not release eligible")
    for field in (
        "environment_lock_sha256",
        "training_configuration_sha256",
        "training_dataset_sha256",
        "universe_sha256",
        "feature_schema_sha256",
        "model_output_sha256",
        "training_log_sha256",
        "training_metrics_sha256",
    ):
        _digest(payload[field], field)
    manifest_digest = _digest(payload["training_manifest_sha256"], "training_manifest_sha256")
    unsigned = dict(payload)
    unsigned.pop("training_manifest_sha256")
    if manifest_digest != _canonical_digest(unsigned, "training provenance"):
        raise ProtocolValidationError("training_manifest_sha256 does not match its contents")
    for field in ("python_version", "cuda_runtime_version", "gpu_identity", "training_command"):
        _text(payload[field], field)
    if type(payload["seed"]) is not int or payload["seed"] < 0:
        raise ProtocolValidationError("training seed must be a non-negative integer")
    started = _timestamp(payload["started_at"], "started_at")
    completed = _timestamp(payload["completed_at"], "completed_at")
    if completed < started:
        raise ProtocolValidationError("training completed_at must not precede started_at")
    return payload


def validate_release_provenance(record: object) -> dict[str, Any]:
    payload = _object(record, "G6 release provenance")
    reject_previous_generation(payload)
    required = {
        "protocol",
        "archive_sha256",
        "manifest_sha256",
        "model_output_sha256",
        "candidate_sha256",
        "training_manifest_sha256",
        "training_dataset_sha256",
        "universe_sha256",
        "feature_schema_sha256",
        "repository",
        "workflow",
        "commit",
        "actions_sha_pinned",
        "job_permissions_minimal",
        "id_token_write_scoped",
        "production_environment_protected",
        "repository_identity_verified",
        "workflow_identity_verified",
        "untrusted_pr_signing",
        "attestation_verified",
        "simulated",
        "cosign_bundle_verified",
    }
    _strict_keys(payload, required, required, "G6 release provenance")
    reject_forbidden_markers(payload)
    if payload["protocol"] != VMR_V12_PROTOCOL:
        raise ProtocolValidationError("release provenance must declare protocol VMR-V12")
    _digest(payload["archive_sha256"], "archive_sha256")
    _digest(payload["manifest_sha256"], "manifest_sha256")
    _digest(payload["model_output_sha256"], "model_output_sha256")
    _digest(payload["candidate_sha256"], "candidate_sha256")
    for field in (
        "training_manifest_sha256",
        "training_dataset_sha256",
        "universe_sha256",
        "feature_schema_sha256",
    ):
        _digest(payload[field], field)
    _text(payload["repository"], "repository", identifier=True)
    _text(payload["workflow"], "workflow", identifier=True)
    _text(payload["commit"], "commit", identifier=True)
    for field in (
        "actions_sha_pinned",
        "job_permissions_minimal",
        "id_token_write_scoped",
        "production_environment_protected",
        "repository_identity_verified",
        "workflow_identity_verified",
    ):
        if _bool(payload[field], field) is not True:
            raise ProtocolValidationError(f"release control {field} is not satisfied")
    if _bool(payload["untrusted_pr_signing"], "untrusted_pr_signing") is not False:
        raise ProtocolValidationError("untrusted pull requests cannot sign releases")
    if _bool(payload["attestation_verified"], "attestation_verified") is not True:
        raise ProtocolValidationError("release artifact attestation is not verified")
    if _bool(payload["simulated"], "simulated") is not False:
        raise ProtocolValidationError("simulated release provenance is forbidden")
    if _bool(payload["cosign_bundle_verified"], "cosign_bundle_verified") is not True:
        raise ProtocolValidationError("real Cosign bundle verification is required")
    return payload


def validate_randomness_policy(policy: object) -> dict[str, Any]:
    payload = _object(policy, "public-randomness policy")
    reject_previous_generation(payload)
    if payload != PUBLIC_RANDOMNESS_POLICY:
        raise ProtocolValidationError("public-randomness policy differs from the frozen V12 policy")
    return payload


def reject_previous_generation(payload: object) -> None:
    # Tokenization alone turns ``V11.2`` into ``v11`` and ``2``. Normalize
    # punctuation as well so renaming an old artifact cannot bypass the check.
    normalized = re.sub(r"[^a-z0-9]+", "_", str(payload).lower())
    tokens = _tokens(payload)
    if (
        re.search(r"(?:^|_)v11(?:_?2)?(?:_|$)", normalized)
        or "v11_2" in tokens
        or "invalidated_opened" in tokens
    ):
        raise ProtocolValidationError("V11.2 evidence is permanently invalidated for VMR-V12")
