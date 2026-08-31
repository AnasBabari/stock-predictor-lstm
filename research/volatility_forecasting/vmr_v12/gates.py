"""Fail-closed VMR-V12 gate computation.

Every gate computes its own result from validated evidence.  Callers cannot
promote a model by setting a free-form ``verified`` or ``release_eligible``
boolean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .policies import GATE_IDS, VMR_V12_PROTOCOL
from .schemas import (
    ProtocolValidationError,
    reject_previous_generation,
    validate_candidate_manifest,
    validate_data_usage_record,
    validate_evaluation_record,
    validate_release_provenance,
    validate_scientific_acceptance,
    validate_training_provenance,
    validate_universe_manifest,
)

_GATE_TOKEN = object()
_ELIGIBILITY_TOKEN = object()


@dataclass(frozen=True, init=False)
class GateResult:
    """Computed result for one gate.

    ``passed`` is derived from the immutable reason list and is deliberately
    not a public constructor argument.  Gate callers must use one of the
    evaluator functions below rather than supplying an override boolean.
    """

    gate: str
    passed: bool
    reasons: tuple[dict[str, str], ...]
    evidence_hashes: tuple[str, ...]

    def __init__(
        self,
        gate: str,
        reasons: tuple[dict[str, str], ...] = (),
        evidence_hashes: tuple[str, ...] = (),
        *,
        _token: object | None = None,
    ) -> None:
        if _token is not _GATE_TOKEN:
            raise TypeError("GateResult instances must come from a gate evaluator")
        if gate not in GATE_IDS:
            raise ValueError(f"unknown VMR-V12 gate: {gate}")
        object.__setattr__(self, "gate", gate)
        object.__setattr__(self, "reasons", tuple(dict(reason) for reason in reasons))
        object.__setattr__(self, "evidence_hashes", tuple(evidence_hashes))
        object.__setattr__(self, "passed", not reasons)

    @classmethod
    def _computed(
        cls,
        gate: str,
        reasons: tuple[dict[str, str], ...] = (),
        evidence_hashes: tuple[str, ...] = (),
    ) -> GateResult:
        return cls(gate, reasons, evidence_hashes, _token=_GATE_TOKEN)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "passed": self.passed,
            "protocol": VMR_V12_PROTOCOL,
            "reasons": [dict(reason) for reason in self.reasons],
            "evidence_hashes": list(self.evidence_hashes),
        }


@dataclass(frozen=True, init=False)
class ReleaseEligibility:
    """Aggregate of all six independently computed gate results."""

    gates: tuple[GateResult, ...]
    release_eligible: bool

    def __init__(
        self,
        gates: tuple[GateResult, ...],
        release_eligible: bool,
        *,
        _token: object | None = None,
    ) -> None:
        if _token is not _ELIGIBILITY_TOKEN:
            raise TypeError("ReleaseEligibility must be computed from all gates")
        object.__setattr__(self, "gates", tuple(gates))
        object.__setattr__(self, "release_eligible", release_eligible)

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": VMR_V12_PROTOCOL,
            "release_eligible": self.release_eligible,
            "gates": [gate.to_dict() for gate in self.gates],
        }


def _failed(gate: str, code: str, message: str) -> GateResult:
    return GateResult._computed(gate, ({"code": code, "message": message},))


def _hashes(*values: object) -> tuple[str, ...]:
    hashes: list[str] = []
    for value in values:
        if isinstance(value, str) and len(value) == 64:
            hashes.append(value)
    return tuple(hashes)


def evaluate_data_usage(record: object) -> GateResult:
    gate = GATE_IDS[0]
    try:
        payload = validate_data_usage_record(record)
    except ProtocolValidationError as exc:
        return _failed(gate, "DATA_USAGE_EVIDENCE_INVALID", str(exc))
    return GateResult._computed(
        gate,
        evidence_hashes=_hashes(
            payload["raw_artifact_sha256"], payload["rights_basis"]["document_sha256"]
        ),
    )


def evaluate_benchmark_integrity(manifest: object) -> GateResult:
    gate = GATE_IDS[1]
    try:
        payload = validate_universe_manifest(manifest)
    except ProtocolValidationError as exc:
        return _failed(gate, "BENCHMARK_INTEGRITY_INVALID", str(exc))
    integrity = payload["integrity_policy"]
    if any(
        integrity[field] is not True
        for field in (
            "strictly_increasing_timestamps",
            "no_feature_after_origin",
            "train_only_transforms",
            "asset_transfer_excluded_from_fit",
            "corporate_action_consistent",
            "explicit_missing_data_handling",
            "deterministic_reconstruction",
        )
    ):
        return _failed(
            gate, "INTEGRITY_ASSERTION_FAILED", "a required integrity assertion is false"
        )
    return GateResult._computed(
        gate,
        evidence_hashes=_hashes(
            payload["ordered_universe_hash"], *payload["selection_input_hashes"]
        ),
    )


def evaluate_candidate_precommitment(manifest: object) -> GateResult:
    gate = GATE_IDS[2]
    try:
        payload = validate_candidate_manifest(manifest)
    except ProtocolValidationError as exc:
        return _failed(gate, "CANDIDATE_PRECOMMITMENT_INVALID", str(exc))
    return GateResult._computed(
        gate,
        evidence_hashes=_hashes(
            payload["candidate_sha256"],
            payload["candidate_weights_sha256"],
            payload["training_data_sha256"],
            payload["universe_sha256"],
            payload["feature_schema_sha256"],
            payload["target_schema_sha256"],
        ),
    )


def evaluate_evaluation_validity(
    records: object,
    *,
    candidate_sha256: str | None = None,
    candidate_manifest: object | None = None,
) -> GateResult:
    gate = GATE_IDS[3]
    if not isinstance(records, (list, tuple)) or not records:
        return _failed(gate, "OFFICIAL_EVALUATION_MISSING", "evaluation records are required")
    validated: list[dict[str, Any]] = []
    for record in records:
        try:
            validated.append(validate_evaluation_record(record))
        except ProtocolValidationError as exc:
            return _failed(gate, "EVALUATION_RECORD_INVALID", str(exc))
    official = [record for record in validated if record["record_type"] == "official"]
    if len(official) != 1:
        return _failed(
            gate,
            "OFFICIAL_EVALUATION_COUNT_INVALID",
            "exactly one admissible official evaluation is required",
        )
    official_record = official[0]
    if candidate_manifest is None:
        return _failed(
            gate,
            "CANDIDATE_MANIFEST_MISSING",
            "official evaluation must be checked against its frozen candidate manifest",
        )
    try:
        candidate = validate_candidate_manifest(candidate_manifest)
    except ProtocolValidationError as exc:
        return _failed(gate, "CANDIDATE_MANIFEST_INVALID", str(exc))
    expected_candidate_sha256 = candidate_sha256 or candidate["candidate_sha256"]
    if candidate["candidate_sha256"] != expected_candidate_sha256:
        return _failed(gate, "CANDIDATE_HASH_MISMATCH", "candidate manifest differs")
    for record in validated:
        if record["selected_partition"] not in candidate["evaluation_partition_catalogue"]:
            return _failed(
                gate,
                "PARTITION_NOT_PRECOMMITTED",
                "evaluation selected a partition outside the frozen catalogue",
            )
        if record["selection_algorithm_version"] != candidate["selection_algorithm_version"]:
            return _failed(
                gate,
                "SELECTION_ALGORITHM_MISMATCH",
                "evaluation selection algorithm differs from precommitment",
            )
        expected_provider = "nist_beacon_v2" if record["fallback_used"] else "drand"
        if record["randomness_provider"] != expected_provider:
            return _failed(
                gate,
                "RANDOMNESS_PROVIDER_POLICY_VIOLATION",
                "evaluation provider does not follow the precommitted fallback policy",
            )
        chain_field = (
            "secondary_chain_or_root" if record["fallback_used"] else "primary_chain_or_root"
        )
        if record["provider_chain_or_root"] != candidate["public_randomness_policy"][chain_field]:
            return _failed(
                gate,
                "RANDOMNESS_CHAIN_MISMATCH",
                "evaluation provider chain/root differs from precommitment",
            )
    if official_record["candidate_sha256"] != expected_candidate_sha256:
        return _failed(gate, "CANDIDATE_HASH_MISMATCH", "official evaluation candidate differs")
    for record in validated:
        if record["candidate_sha256"] != official_record["candidate_sha256"]:
            return _failed(
                gate,
                "CANDIDATE_HASH_MISMATCH",
                "all evaluation records must bind the official candidate",
            )
        if (
            record["record_type"] == "reproduction"
            and record.get("reference_to_original_official_record")
            != official_record["evaluation_record_sha256"]
        ):
            return _failed(
                gate,
                "REPRODUCTION_REFERENCE_INVALID",
                "reproduction records must reference the official record",
            )
    return GateResult._computed(
        gate,
        evidence_hashes=_hashes(official_record["evaluation_record_sha256"]),
    )


def evaluate_scientific_acceptance(record: object) -> GateResult:
    gate = GATE_IDS[4]
    try:
        payload = validate_scientific_acceptance(record)
    except ProtocolValidationError as exc:
        return _failed(gate, "SCIENTIFIC_ACCEPTANCE_INVALID", str(exc))
    required = (
        "test_was_precommitted",
        "evaluation_complete",
        "no_leakage_detected",
        "model_beats_required_baseline",
        "confidence_requirement_passed",
        "calibration_requirement_passed",
        "no_material_asset_group_regression",
    )
    if any(payload[field] is not True for field in required):
        return _failed(
            gate, "SCIENTIFIC_ACCEPTANCE_FAILED", "one or more scientific requirements failed"
        )
    return GateResult._computed(gate, evidence_hashes=())


def evaluate_release_provenance(
    training_record: object,
    release_record: object,
) -> GateResult:
    gate = GATE_IDS[5]
    try:
        training = validate_training_provenance(training_record)
        release = validate_release_provenance(release_record)
    except ProtocolValidationError as exc:
        return _failed(gate, "RELEASE_PROVENANCE_INVALID", str(exc))
    if training["model_output_sha256"] != release["model_output_sha256"]:
        return _failed(
            gate,
            "TRAINING_RELEASE_BINDING_MISMATCH",
            "release manifest is not bound to the training output identity",
        )
    if training["candidate_sha256"] != release["candidate_sha256"]:
        return _failed(
            gate,
            "TRAINING_RELEASE_CANDIDATE_MISMATCH",
            "release candidate differs from the training provenance",
        )
    if training["training_manifest_sha256"] != release["training_manifest_sha256"]:
        return _failed(
            gate,
            "TRAINING_MANIFEST_MISMATCH",
            "release does not bind the exact RTX training manifest",
        )
    for field in ("training_dataset_sha256", "universe_sha256", "feature_schema_sha256"):
        if training[field] != release[field]:
            return _failed(
                gate,
                "TRAINING_INPUT_BINDING_MISMATCH",
                f"release does not bind training {field}",
            )
    return GateResult._computed(
        gate, evidence_hashes=_hashes(release["archive_sha256"], release["manifest_sha256"])
    )


def evaluate_release_eligibility(results: object) -> ReleaseEligibility:
    """Aggregate computed results; caller-supplied eligibility is ignored."""

    if not isinstance(results, (list, tuple)):
        raise TypeError("gate results must be a sequence")
    by_gate: dict[str, GateResult] = {}
    for result in results:
        if not isinstance(result, GateResult):
            raise TypeError("gate results must be computed GateResult instances")
        if result.gate in by_gate:
            raise ValueError(f"duplicate result for gate {result.gate}")
        by_gate[result.gate] = result
    missing = [gate for gate in GATE_IDS if gate not in by_gate]
    if missing:
        raise ValueError(f"missing VMR-V12 gate results: {missing}")
    ordered = tuple(by_gate[gate] for gate in GATE_IDS)
    return ReleaseEligibility(
        ordered,
        all(result.passed for result in ordered),
        _token=_ELIGIBILITY_TOKEN,
    )


def reject_v11_2_evidence(evidence: object) -> None:
    """Reject any previous-generation evidence, including renamed files."""

    try:
        reject_previous_generation(evidence)
    except ProtocolValidationError:
        raise
    if isinstance(evidence, dict) and evidence.get("protocol") != VMR_V12_PROTOCOL:
        raise ProtocolValidationError("evidence must declare protocol VMR-V12")
