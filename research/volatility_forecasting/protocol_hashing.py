"""Cryptographically enforced canonical protocol hashing for StockLSTM.

Provides deterministic, byte-exact SHA-256 hashing for experiment protocols.
Ensures sorted keys, compact separators, and UTF-8 encoding.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_protocol_bytes(protocol: dict[str, Any]) -> bytes:
    """Serialize a protocol dictionary into canonical UTF-8 JSON bytes.

    Strips any self-referential digest fields before serialization.
    """
    clean = {k: v for k, v in protocol.items() if not k.endswith("_sha256") and k != "digest"}
    payload = json.dumps(clean, sort_keys=True, separators=(",", ":"))
    return payload.encode("utf-8")


def protocol_sha256(protocol: dict[str, Any]) -> str:
    """Compute canonical SHA-256 digest of a protocol dictionary."""
    return hashlib.sha256(canonical_protocol_bytes(protocol)).hexdigest()
