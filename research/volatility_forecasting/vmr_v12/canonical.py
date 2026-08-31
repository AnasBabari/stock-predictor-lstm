"""Deterministic canonical JSON used by VMR-V12 manifests.

Canonicalization is intentionally small and explicit: UTF-8 JSON, sorted
keys, compact separators, no ASCII escaping, and no non-finite numbers.  V12
timestamps are represented as strings and are validated by the schema layer;
this module never silently converts them.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence


class CanonicalizationError(ValueError):
    """Raised when a manifest cannot be represented deterministically."""


def _validate_value(value: object, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalizationError(f"non-finite number at {path}")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str) or not key:
                raise CanonicalizationError(f"object keys must be non-empty strings at {path}")
            _validate_value(child, f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for index, child in enumerate(value):
            _validate_value(child, f"{path}[{index}]")
        return
    raise CanonicalizationError(f"unsupported value type at {path}: {type(value).__name__}")


def canonical_json(value: object) -> str:
    """Return the canonical JSON text for a JSON-compatible value."""

    _validate_value(value)
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CanonicalizationError("value is not canonically serializable") from exc


def canonical_bytes(value: object) -> bytes:
    """Return canonical JSON encoded as UTF-8 bytes."""

    try:
        return canonical_json(value).encode("utf-8")
    except UnicodeError as exc:
        raise CanonicalizationError("value contains invalid UTF-8 text") from exc


def canonical_digest(value: object) -> str:
    """Return the lowercase SHA-256 digest of canonical JSON bytes."""

    return hashlib.sha256(canonical_bytes(value)).hexdigest()
