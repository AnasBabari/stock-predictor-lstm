"""Load and validate the immutable VMR-V12 protocol metadata."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .canonical import canonical_digest
from .policies import (
    GATE_IDS,
    PUBLIC_RANDOMNESS_POLICY,
    TERMINAL_EVENT_POLICY,
    TERMINAL_EVENT_POLICY_VERSION,
    VMR_V12_BENCHMARK,
    VMR_V12_INTEGRITY_STATEMENT,
    VMR_V12_PRODUCTION_ABSTENTION,
    VMR_V12_PROTOCOL,
    VMR_V12_PROTOCOL_METADATA,
    VMR_V12_SCOPE,
    VMR_V12_UNIVERSE_DESIGN,
)
from .schemas import (
    ProtocolValidationError,
    validate_randomness_policy,
    validate_terminal_policy,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL_PATH = ROOT / "configs" / "vmr_v12_protocol.json"


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolValidationError("VMR-V12 protocol metadata is missing or malformed") from exc
    if not isinstance(value, dict):
        raise ProtocolValidationError("VMR-V12 protocol metadata must be a JSON object")
    return value


def validate_protocol_metadata(metadata: object) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise ProtocolValidationError("VMR-V12 protocol metadata must be an object")
    expected = VMR_V12_PROTOCOL_METADATA
    if set(metadata) != set(expected):
        raise ProtocolValidationError("VMR-V12 protocol metadata fields are not frozen")
    for field in ("schema_version", "protocol", "benchmark", "universe_design", "terminal_policy"):
        if metadata.get(field) != expected[field]:
            raise ProtocolValidationError(f"VMR-V12 protocol field {field} is not frozen correctly")
    previous = metadata.get("previous_generation")
    if not isinstance(previous, dict) or previous != expected["previous_generation"]:
        raise ProtocolValidationError("V11.2 must remain structurally ineligible for VMR-V12")
    production = metadata.get("production")
    if not isinstance(production, dict) or production != expected["production"]:
        raise ProtocolValidationError("VMR-V12 production statuses are not frozen correctly")
    gates = metadata.get("gates")
    if gates != list(GATE_IDS):
        raise ProtocolValidationError("VMR-V12 gate ordering or identifiers changed")
    validate_terminal_policy(
        {
            "version": metadata.get("terminal_policy"),
            "rules": metadata.get("terminal_event_policy_rules"),
        }
    )
    if metadata.get("scope") != VMR_V12_SCOPE:
        raise ProtocolValidationError("VMR-V12 scope has been broadened in metadata")
    if metadata.get("integrity_statement") != VMR_V12_INTEGRITY_STATEMENT:
        raise ProtocolValidationError("VMR-V12 integrity statement changed")
    if metadata.get("not_third_party_certification") is not True:
        raise ProtocolValidationError("VMR-V12 cannot claim third-party certification")
    validate_randomness_policy(metadata.get("randomness"))
    return metadata


def load_protocol_metadata(path: Path = DEFAULT_PROTOCOL_PATH) -> dict[str, Any]:
    return validate_protocol_metadata(_read_object(path))


def protocol_manifest() -> dict[str, Any]:
    """Return the canonical in-code protocol metadata and its digest."""

    metadata = deepcopy(VMR_V12_PROTOCOL_METADATA)
    metadata["terminal_event_policy"] = {
        "version": TERMINAL_EVENT_POLICY["version"],
        "rules": list(TERMINAL_EVENT_POLICY["rules"]),
    }
    metadata["terminal_event_policy_version"] = TERMINAL_EVENT_POLICY_VERSION
    metadata["protocol_sha256"] = canonical_digest(metadata)
    return metadata


__all__ = [
    "DEFAULT_PROTOCOL_PATH",
    "load_protocol_metadata",
    "protocol_manifest",
    "validate_protocol_metadata",
    "VMR_V12_PROTOCOL",
    "VMR_V12_BENCHMARK",
    "VMR_V12_UNIVERSE_DESIGN",
    "TERMINAL_EVENT_POLICY_VERSION",
    "VMR_V12_PRODUCTION_ABSTENTION",
    "PUBLIC_RANDOMNESS_POLICY",
]
