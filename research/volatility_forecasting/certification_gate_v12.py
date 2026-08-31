"""Fail-closed external-evidence gate for the next certification generation.

This module verifies policy structure and invokes the real ``cosign`` binary.
It never generates keys, signatures, holdout secrets, or substitute evidence.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_PATH = ROOT / "configs" / "certification_gates.json"


class CertificationGateError(PermissionError):
    """Raised when certification evidence is missing, forbidden, or unverifiable."""


@dataclass(frozen=True)
class CosignVerification:
    """Arguments for verification performed by the external Cosign executable."""

    artifact_path: Path
    bundle_path: Path
    public_key_path: Path | None = None
    certificate_identity: str | None = None
    certificate_oidc_issuer: str | None = None
    executable: str = "cosign"


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CertificationGateError(f"{label} is missing or malformed") from exc
    if not isinstance(payload, dict):
        raise CertificationGateError(f"{label} must contain a JSON object")
    return payload


def _walk(value: object) -> list[object]:
    if isinstance(value, Mapping):
        values: list[object] = []
        for item in value.values():
            values.extend(_walk(item))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = []
        for item in value:
            values.extend(_walk(item))
    else:
        values = [value]
    return values


def _tokens(value: object) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9_]+", str(value).lower()) if token}


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CertificationGateError(f"certification evidence {label} is required")
    return value


def _require_exact(value: object, expected: object, label: str) -> None:
    if value != expected or type(value) is not type(expected):
        raise CertificationGateError(
            f"certification evidence {label} must be {expected!r}, got {value!r}"
        )


def verify_cosign_bundle(
    request: CosignVerification,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    """Verify one genuine Cosign bundle without accepting a custom receipt format."""

    artifact = request.artifact_path.resolve()
    bundle = request.bundle_path.resolve()
    if not artifact.is_file() or not bundle.is_file():
        raise CertificationGateError("Cosign artifact and bundle must both exist")
    command = [request.executable, "verify-blob", str(artifact), "--bundle", str(bundle)]
    if request.public_key_path is not None:
        public_key = request.public_key_path.resolve()
        if not public_key.is_file():
            raise CertificationGateError("Cosign public key is missing")
        if request.certificate_identity or request.certificate_oidc_issuer:
            raise CertificationGateError("Cosign key and keyless identity modes cannot be mixed")
        command.extend(["--key", str(public_key)])
    else:
        if not request.certificate_identity or not request.certificate_oidc_issuer:
            raise CertificationGateError(
                "keyless Cosign verification requires an identity and OIDC issuer"
            )
        command.extend(
            [
                "--certificate-identity",
                request.certificate_identity,
                "--certificate-oidc-issuer",
                request.certificate_oidc_issuer,
            ]
        )
    try:
        completed = runner(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CertificationGateError("Cosign verification could not be executed") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "verification failed").strip()
        raise CertificationGateError(f"Cosign verification failed: {detail[:500]}")


def verify_certification_evidence(
    evidence_path: Path,
    cosign: CosignVerification,
    *,
    policy_path: Path = DEFAULT_POLICY_PATH,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Validate the next-generation evidence manifest and its real Cosign bundle."""

    policy = _json_object(policy_path, "certification policy")
    evidence = _json_object(evidence_path, "certification evidence")
    forbidden = policy.get("forbidden_markers")
    if not isinstance(forbidden, list) or not all(isinstance(item, str) for item in forbidden):
        raise CertificationGateError("certification policy forbidden markers are malformed")
    evidence_tokens: set[str] = set()
    for value in _walk(evidence):
        evidence_tokens.update(_tokens(value))
    found = sorted(set(forbidden).intersection(evidence_tokens))
    if found:
        raise CertificationGateError(
            f"certification evidence contains permanently forbidden markers: {found}"
        )

    generation = evidence.get("generation")
    if not isinstance(generation, str) or generation.lower() in {"v11.2", "v11_2"}:
        raise CertificationGateError("certification evidence must use a new post-V11.2 generation")
    ohlcv = _require_mapping(evidence.get("ohlcv"), "ohlcv")
    pit64 = _require_mapping(evidence.get("pit64"), "pit64")
    signature = _require_mapping(evidence.get("signature"), "signature")
    holdout = _require_mapping(evidence.get("holdout"), "holdout")
    reserve = _require_mapping(evidence.get("reserve"), "reserve")
    _require_exact(ohlcv.get("permission_status"), "VERIFIED", "ohlcv.permission_status")
    _require_exact(ohlcv.get("source_is_external"), True, "ohlcv.source_is_external")
    _require_exact(ohlcv.get("hash_verified"), True, "ohlcv.hash_verified")
    _require_exact(pit64.get("completeness_verified"), True, "pit64.completeness_verified")
    _require_exact(
        pit64.get("external_reviewer_verified"), True, "pit64.external_reviewer_verified"
    )
    _require_exact(
        signature.get("real_cosign_verification"),
        True,
        "signature.real_cosign_verification",
    )
    _require_exact(signature.get("simulated"), False, "signature.simulated")
    _require_exact(holdout.get("generated_externally"), True, "holdout.generated_externally")
    _require_exact(holdout.get("previously_revealed"), False, "holdout.previously_revealed")
    _require_exact(reserve.get("previously_opened"), False, "reserve.previously_opened")
    _require_exact(evidence.get("code_hash_matches"), True, "code_hash_matches")
    _require_exact(evidence.get("dataset_hash_matches"), True, "dataset_hash_matches")
    if cosign.artifact_path.resolve() != evidence_path.resolve():
        raise CertificationGateError("Cosign must verify the exact evidence manifest")
    verify_cosign_bundle(cosign, runner=runner)
    return evidence
