#!/usr/bin/env python3
"""Create unsigned, content-bound V11.2 attestation request documents.

The generated files are requests for external signatures, not attestations.
This command never creates a key, accepts a private key, or writes a signature.
Its purpose is to give the market-data licensor and independent PIT64 reviewer
the exact subjects, rights, and evidence digests expected by the verifier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "research"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from research.volatility_forecasting.v11_2_attestation import (  # noqa: E402
    ATTESTATION_SCHEMA_VERSION,
    MARKET_DATA_ATTESTATION,
    MARKET_REQUIRED_RIGHTS,
    PIT64_ATTESTATION,
    PIT64_REQUIRED_RIGHTS,
    SIGNATURE_ALGORITHM,
    AttestationError,
    security_master_digest,
    sha256_file,
)
from research.volatility_forecasting.v11_2_protocol import (  # noqa: E402
    V11_2_PROTOCOL_ID,
    canonical_json_digest,
)

REQUEST_SCHEMA_VERSION = 1
_EVIDENCE_NAME = re.compile(r"[A-Za-z0-9_.-]{1,80}")


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AttestationError(f"{label} is missing or malformed: {path}") from exc
    if not isinstance(payload, dict):
        raise AttestationError(f"{label} must contain a JSON object")
    return payload


def _required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AttestationError(f"{label} is required")
    return value.strip()


def _labelled_sha256(value: object, label: str) -> str:
    text = _required_text(value, label)
    if not text.startswith("sha256:"):
        raise AttestationError(f"{label} must use sha256:<digest>")
    digest = text.removeprefix("sha256:")
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise AttestationError(f"{label} is not a lowercase SHA-256 digest")
    return digest


def evidence_args(values: list[str]) -> dict[str, Path]:
    """Parse unique ``NAME=PATH`` evidence arguments."""

    parsed: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        normalized = name.strip()
        if not separator or not _EVIDENCE_NAME.fullmatch(normalized) or not raw_path.strip():
            raise AttestationError("evidence must use NAME=PATH with a safe evidence name")
        if normalized in parsed:
            raise AttestationError(f"duplicate evidence name: {normalized}")
        path = Path(raw_path.strip()).resolve()
        if not path.is_file():
            raise AttestationError(f"evidence file is missing: {path}")
        parsed[normalized] = path
    return parsed


def _evidence_digests(files: dict[str, Path]) -> dict[str, str]:
    if not files:
        raise AttestationError("at least one evidence file is required")
    return {name: f"sha256:{sha256_file(path)}" for name, path in sorted(files.items())}


def _request(
    *,
    attestation_type: str,
    subject_kind: str,
    subject_id: str,
    subject_digest: str,
    required_rights: frozenset[str],
    evidence_files: dict[str, Path],
) -> dict[str, Any]:
    payload = {
        "request_schema_version": REQUEST_SCHEMA_VERSION,
        "receipt_schema_version": ATTESTATION_SCHEMA_VERSION,
        "protocol_id": V11_2_PROTOCOL_ID,
        "attestation_type": attestation_type,
        "signature_algorithm": SIGNATURE_ALGORITHM,
        "subject": {
            "kind": subject_kind,
            "id": subject_id,
            "content_digest": f"sha256:{subject_digest}",
        },
        "required_rights": sorted(required_rights),
        "evidence_files": _evidence_digests(evidence_files),
        "notice": (
            "UNSIGNED REQUEST ONLY. The external issuer must independently verify the "
            "evidence and construct the signed receipt defined by "
            "docs/V11_2_ATTESTATION_SCHEMA.md."
        ),
    }
    payload["request_sha256"] = canonical_json_digest(payload)
    return payload


def create_requests(
    *,
    snapshot_manifest_path: Path,
    universe_manifest_path: Path,
    market_evidence_files: dict[str, Path],
    pit64_evidence_files: dict[str, Path],
    output_dir: Path,
) -> dict[str, Any]:
    """Write immutable unsigned request documents and return their manifest."""

    snapshot = _json_object(snapshot_manifest_path, "snapshot manifest")
    universe = _json_object(universe_manifest_path, "PIT64 universe manifest")
    if (
        snapshot_manifest_path.resolve()
        != market_evidence_files.get("snapshot_manifest", Path()).resolve()
    ):
        raise AttestationError(
            "market evidence snapshot_manifest must be the exact snapshot manifest input"
        )
    if "membership_master" not in pit64_evidence_files:
        raise AttestationError("PIT64 evidence must include membership_master")

    panel_id = _required_text(snapshot.get("panel_id"), "snapshot panel_id")
    pooled_checksum = _labelled_sha256(snapshot.get("pooled_checksum"), "snapshot pooled_checksum")
    universe_version = _required_text(universe.get("universe_version"), "universe_version")
    if universe.get("protocol_id") != V11_2_PROTOCOL_ID:
        raise AttestationError("PIT64 universe protocol does not match V11.2")
    if universe.get("certification_eligible") is not True:
        raise AttestationError("PIT64 universe is explicitly development-only")

    market_request = _request(
        attestation_type=MARKET_DATA_ATTESTATION,
        subject_kind="immutable_ohlcv_snapshot",
        subject_id=panel_id,
        subject_digest=pooled_checksum,
        required_rights=MARKET_REQUIRED_RIGHTS,
        evidence_files=market_evidence_files,
    )
    pit64_request = _request(
        attestation_type=PIT64_ATTESTATION,
        subject_kind="pit64_security_master",
        subject_id=universe_version,
        subject_digest=security_master_digest(universe),
        required_rights=PIT64_REQUIRED_RIGHTS,
        evidence_files=pit64_evidence_files,
    )

    output = output_dir.resolve()
    if output.exists():
        raise AttestationError(f"refusing to overwrite attestation request directory: {output}")
    output.mkdir(parents=True)
    market_path = output / "market_data_attestation_request.json"
    pit64_path = output / "pit64_attestation_request.json"
    market_path.write_text(
        json.dumps(market_request, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pit64_path.write_text(
        json.dumps(pit64_request, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "protocol_id": V11_2_PROTOCOL_ID,
        "status": "unsigned_external_signatures_required",
        "market_request": {
            "path": market_path.name,
            "sha256": hashlib.sha256(market_path.read_bytes()).hexdigest(),
        },
        "pit64_request": {
            "path": pit64_path.name,
            "sha256": hashlib.sha256(pit64_path.read_bytes()).hexdigest(),
        },
        "private_keys_created": False,
        "holdout_accessed": False,
    }
    manifest["manifest_sha256"] = canonical_json_digest(manifest)
    (output / "request_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--universe-manifest", type=Path, required=True)
    parser.add_argument("--market-evidence", action="append", default=[])
    parser.add_argument("--pit64-evidence", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = create_requests(
            snapshot_manifest_path=args.snapshot_manifest.resolve(),
            universe_manifest_path=args.universe_manifest.resolve(),
            market_evidence_files=evidence_args(args.market_evidence),
            pit64_evidence_files=evidence_args(args.pit64_evidence),
            output_dir=args.output_dir,
        )
    except (AttestationError, OSError, ValueError, TypeError) as exc:
        raise SystemExit(f"V11.2 attestation request creation failed: {exc}") from exc
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
