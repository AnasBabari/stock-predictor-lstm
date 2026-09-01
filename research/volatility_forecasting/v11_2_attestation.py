"""Signed provenance receipts for V11.2 certification inputs.

The V11.2 ``certification_eligible`` flag is intentionally not treated as
evidence.  A production run must carry two independently verifiable receipts:

* a market-data licence receipt bound to the immutable OHLCV snapshot; and
* a point-in-time security-master receipt bound to the exact PIT64 identities.

Receipts are detached JSON envelopes signed by an Ed25519 attester.  This
module does not create keys or attest data; it only verifies operator-supplied
evidence before a panel can be sealed or a holdout can be opened.
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .v11_2_protocol import canonical_json_digest

ATTESTATION_SCHEMA_VERSION = 1
MARKET_DATA_ATTESTATION = "v11_2_market_data_license"
PIT64_ATTESTATION = "v11_2_pit64_membership"
SIGNATURE_ALGORITHM = "ed25519"

MARKET_REQUIRED_RIGHTS = frozenset(
    {
        "historical_ohlcv",
        "model_training",
        "derived_model_distribution",
        "production_inference",
    }
)
PIT64_REQUIRED_RIGHTS = frozenset(
    {
        "point_in_time_membership",
        "security_identity",
        "independent_review",
    }
)


class AttestationError(ValueError):
    """Raised when signed provenance evidence is missing or invalid."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the digest of the exact bytes at ``path`` without normalization."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise AttestationError(f"attestation evidence file is unreadable: {path}") from exc
    return digest.hexdigest()


def public_key_fingerprint(path: Path) -> str:
    """Fingerprint an Ed25519 SubjectPublicKeyInfo PEM key."""

    try:
        key = serialization.load_pem_public_key(path.read_bytes())
    except (OSError, ValueError, TypeError) as exc:
        raise AttestationError("attestation public key is missing or malformed") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise AttestationError("attestation public key must be Ed25519")
    der = key.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return _sha256_bytes(der)


def _canonical_unsigned_bytes(payload: Mapping[str, Any]) -> bytes:
    unsigned = {key: value for key, value in payload.items() if key != "signature"}
    return json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def receipt_digest(payload: Mapping[str, Any]) -> str:
    """Digest the exact signed (signature-excluded) receipt payload."""

    return _sha256_bytes(_canonical_unsigned_bytes(payload))


def _require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AttestationError(f"attestation {label} is required")
    return value.strip()


def _require_digest(value: object, label: str, *, labelled: bool = False) -> str:
    text = _require_text(value, label)
    if labelled:
        if not text.startswith("sha256:"):
            raise AttestationError(f"attestation {label} must use sha256:<digest>")
        text = text.removeprefix("sha256:")
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise AttestationError(f"attestation {label} is not a lowercase SHA-256 digest")
    return text


def _require_timestamp(value: object, label: str) -> dt.datetime:
    text = _require_text(value, label)
    try:
        parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AttestationError(f"attestation {label} is not an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise AttestationError(f"attestation {label} must include a timezone")
    return parsed.astimezone(dt.UTC)


def _verify_signature(payload: Mapping[str, Any], public_key_path: Path) -> str:
    signature_text = _require_text(payload.get("signature"), "signature")
    try:
        signature = base64.b64decode(signature_text.encode("ascii"), validate=True)
    except (ValueError, UnicodeEncodeError) as exc:
        raise AttestationError("attestation signature is not valid base64") from exc
    if len(signature) != 64:
        raise AttestationError("attestation signature must be 64 Ed25519 bytes")
    try:
        key = serialization.load_pem_public_key(public_key_path.read_bytes())
    except (OSError, ValueError, TypeError) as exc:
        raise AttestationError("attestation public key is missing or malformed") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise AttestationError("attestation public key must be Ed25519")
    try:
        key.verify(signature, _canonical_unsigned_bytes(payload))
    except InvalidSignature as exc:
        raise AttestationError("attestation signature verification failed") from exc
    return public_key_fingerprint(public_key_path)


def _verify_evidence_files(
    payload: Mapping[str, Any], evidence_files: Mapping[str, Path] | None
) -> None:
    declared = payload.get("evidence_files")
    if not isinstance(declared, dict) or not declared:
        raise AttestationError("attestation evidence_files must be a non-empty object")
    if any(
        not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", name)
        for name in declared
    ):
        raise AttestationError("attestation evidence file names must be non-empty strings")
    if any(
        not isinstance(digest, str)
        or not digest.startswith("sha256:")
        or len(digest.removeprefix("sha256:")) != 64
        or any(char not in "0123456789abcdef" for char in digest.removeprefix("sha256:"))
        for digest in declared.values()
    ):
        raise AttestationError("attestation evidence file digests are malformed")
    if evidence_files is None:
        raise AttestationError("attestation evidence files were not supplied for verification")
    unknown = sorted(set(evidence_files) - set(declared))
    missing = sorted(set(declared) - set(evidence_files))
    if unknown or missing:
        raise AttestationError(
            "attestation evidence mapping differs from the signed receipt"
            + (f"; missing={missing}" if missing else "")
            + (f"; unknown={unknown}" if unknown else "")
        )
    for name, path in evidence_files.items():
        expected = declared[name].removeprefix("sha256:")
        actual = sha256_file(path)
        if actual != expected:
            raise AttestationError(f"attestation evidence checksum mismatch: {name}")


def verify_receipt(
    path: Path,
    public_key_path: Path,
    *,
    attestation_type: str,
    subject_id: str,
    subject_digest: str | None = None,
    required_rights: set[str] | frozenset[str],
    evidence_files: Mapping[str, Path] | None,
    subject_kind: str,
) -> dict[str, Any]:
    """Verify one signed receipt and return its canonical metadata.

    The caller supplies the expected subject and evidence paths.  No receipt
    field is treated as proof until its signature, subject binding, rights,
    and exact source-file checks all pass.
    """

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AttestationError(f"attestation receipt is missing or malformed: {path}") from exc
    if not isinstance(payload, dict):
        raise AttestationError("attestation receipt must contain a JSON object")
    if payload.get("schema_version") != ATTESTATION_SCHEMA_VERSION:
        raise AttestationError("unsupported attestation schema version")
    if payload.get("attestation_type") != attestation_type:
        raise AttestationError("attestation type does not match the required input")
    if payload.get("signature_algorithm") != SIGNATURE_ALGORITHM:
        raise AttestationError("attestation signature algorithm must be Ed25519")
    subject = payload.get("subject")
    if not isinstance(subject, dict):
        raise AttestationError("attestation subject is missing")
    if _require_text(subject.get("kind"), "subject.kind") != subject_kind:
        raise AttestationError("attestation subject kind does not match the required input")
    if _require_text(subject.get("id"), "subject.id") != subject_id:
        raise AttestationError("attestation subject identity does not match the input")
    actual_subject_digest = _require_digest(
        subject.get("content_digest"), "subject.content_digest", labelled=True
    )
    if subject_digest is not None and actual_subject_digest != _require_digest(
        subject_digest, "expected subject digest", labelled=False
    ):
        raise AttestationError("attestation subject content digest does not match the input")
    issuer = payload.get("issuer")
    if not isinstance(issuer, dict):
        raise AttestationError("attestation issuer is missing")
    issuer_name = _require_text(issuer.get("name"), "issuer.name")
    key_id = _require_digest(issuer.get("key_id"), "issuer.key_id", labelled=False)
    _require_timestamp(payload.get("issued_at"), "issued_at")
    rights = payload.get("rights")
    if not isinstance(rights, dict):
        raise AttestationError("attestation rights are missing")
    if any(rights.get(name) is not True for name in required_rights):
        raise AttestationError("attestation does not grant every required usage right")
    independence = payload.get("independent_review")
    if not isinstance(independence, dict) or independence.get("independent") is not True:
        raise AttestationError("attestation lacks an independent review declaration")
    _require_text(independence.get("reviewer"), "independent_review.reviewer")
    _require_text(independence.get("method"), "independent_review.method")
    verified_key_id = _verify_signature(payload, public_key_path)
    if verified_key_id != key_id:
        raise AttestationError("attestation issuer key_id does not match the pinned public key")
    _verify_evidence_files(payload, evidence_files)
    return {
        "attestation_type": attestation_type,
        "receipt_sha256": _sha256_bytes(path.read_bytes()),
        "signed_payload_sha256": receipt_digest(payload),
        "subject_id": subject_id,
        "subject_content_digest": actual_subject_digest,
        "issuer": issuer_name,
        "public_key_sha256": verified_key_id,
        "issued_at": _require_timestamp(payload.get("issued_at"), "issued_at").isoformat(),
    }


def security_master_digest(universe_payload: Mapping[str, Any]) -> str:
    """Compute the pre-manifest digest an attester signs for PIT identities."""

    securities = universe_payload.get("securities")
    if not isinstance(securities, list) or len(securities) != 64:
        raise AttestationError("PIT64 universe must contain exactly 64 securities")
    return canonical_json_digest(securities)


def verify_v11_2_inputs(
    *,
    snapshot_manifest_path: Path,
    universe_manifest_path: Path,
    market_receipt_path: Path,
    market_public_key_path: Path,
    pit64_receipt_path: Path,
    pit64_public_key_path: Path,
    market_evidence_files: Mapping[str, Path],
    pit64_evidence_files: Mapping[str, Path],
) -> dict[str, Any]:
    """Verify the complete signed provenance chain for a V11.2 input pair."""

    try:
        snapshot = json.loads(snapshot_manifest_path.read_text(encoding="utf-8"))
        universe = json.loads(universe_manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AttestationError("V11.2 input manifest is missing or malformed") from exc
    if not isinstance(snapshot, dict) or not isinstance(universe, dict):
        raise AttestationError("V11.2 input manifests must contain JSON objects")
    panel_id = _require_text(snapshot.get("panel_id"), "snapshot.panel_id")
    pooled_checksum = _require_digest(
        snapshot.get("pooled_checksum"), "snapshot.pooled_checksum", labelled=True
    )
    snapshot_license = snapshot.get("license")
    if not isinstance(snapshot_license, dict):
        raise AttestationError("snapshot license metadata is missing")
    if snapshot_license.get("acknowledged") is not True:
        raise AttestationError("snapshot license acknowledgement is not true")
    universe_version = _require_text(universe.get("universe_version"), "universe_version")
    if universe.get("certification_eligible") is not True:
        raise AttestationError("universe is explicitly development-only")
    if "snapshot_manifest" not in market_evidence_files:
        raise AttestationError("market evidence must include snapshot_manifest")
    if "membership_master" not in pit64_evidence_files:
        raise AttestationError("PIT64 evidence must include membership_master")
    market = verify_receipt(
        market_receipt_path,
        market_public_key_path,
        attestation_type=MARKET_DATA_ATTESTATION,
        subject_kind="immutable_ohlcv_snapshot",
        subject_id=panel_id,
        subject_digest=pooled_checksum,
        required_rights=MARKET_REQUIRED_RIGHTS,
        evidence_files=market_evidence_files,
    )
    pit64 = verify_receipt(
        pit64_receipt_path,
        pit64_public_key_path,
        attestation_type=PIT64_ATTESTATION,
        subject_kind="pit64_security_master",
        subject_id=universe_version,
        subject_digest=security_master_digest(universe),
        required_rights=PIT64_REQUIRED_RIGHTS,
        evidence_files=pit64_evidence_files,
    )
    if market["public_key_sha256"] == pit64["public_key_sha256"]:
        raise AttestationError("market and PIT64 receipts must use independent attester keys")
    if market["issuer"] == pit64["issuer"]:
        raise AttestationError("market and PIT64 receipts must name independent issuers")
    return {"market_data": market, "pit64_membership": pit64}


def _dataset_relative_file(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise AttestationError(f"dataset attestation {label} path is missing")
    candidate = Path(value)
    if candidate.is_absolute() or "\x00" in value:
        raise AttestationError(f"dataset attestation {label} path must be relative")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise AttestationError(f"dataset attestation {label} escapes its dataset") from exc
    if not resolved.is_file():
        raise AttestationError(f"dataset attestation {label} file is missing")
    return resolved


def verify_dataset_attestation_record(
    dataset_dir: Path, universe_manifest_path: Path
) -> dict[str, Any]:
    """Verify the copied, immutable receipt chain stored beside a dataset."""

    record_path = dataset_dir / "manifests" / "attestations.json"
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AttestationError("dataset attestation record is missing or malformed") from exc
    if not isinstance(record, dict) or record.get("schema_version") != 1:
        raise AttestationError("dataset attestation record schema is unsupported")
    record_digest = _require_digest(record.pop("record_sha256", None), "record_sha256")
    if canonical_json_digest(record) != record_digest:
        raise AttestationError("dataset attestation record digest does not match its contents")
    market = record.get("market")
    pit64 = record.get("pit64")
    if not isinstance(market, dict) or not isinstance(pit64, dict):
        raise AttestationError("dataset attestation record must contain market and pit64 entries")
    market_evidence_payload = market.get("evidence")
    pit64_evidence_payload = pit64.get("evidence")
    if not isinstance(market_evidence_payload, dict) or not isinstance(
        pit64_evidence_payload, dict
    ):
        raise AttestationError("dataset attestation evidence mappings are missing")
    market_evidence = {
        str(name): _dataset_relative_file(dataset_dir, value, f"market evidence {name}")
        for name, value in market_evidence_payload.items()
    }
    pit64_evidence = {
        str(name): _dataset_relative_file(dataset_dir, value, f"pit64 evidence {name}")
        for name, value in pit64_evidence_payload.items()
    }
    snapshot_manifest = market_evidence.get("snapshot_manifest")
    if snapshot_manifest is None:
        raise AttestationError("market attestation must bind a snapshot_manifest evidence file")
    summary = verify_v11_2_inputs(
        snapshot_manifest_path=snapshot_manifest,
        universe_manifest_path=universe_manifest_path,
        market_receipt_path=_dataset_relative_file(
            dataset_dir, market.get("receipt"), "market receipt"
        ),
        market_public_key_path=_dataset_relative_file(
            dataset_dir, market.get("public_key"), "market public key"
        ),
        pit64_receipt_path=_dataset_relative_file(
            dataset_dir, pit64.get("receipt"), "pit64 receipt"
        ),
        pit64_public_key_path=_dataset_relative_file(
            dataset_dir, pit64.get("public_key"), "pit64 public key"
        ),
        market_evidence_files=market_evidence,
        pit64_evidence_files=pit64_evidence,
    )
    declared = record.get("verification")
    if not isinstance(declared, dict) or canonical_json_digest(declared) != canonical_json_digest(
        summary
    ):
        raise AttestationError("dataset attestation verification summary was changed")
    return {"record_sha256": record_digest, **summary}
