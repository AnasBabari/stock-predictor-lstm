"""Ed25519-signed manifests for server forecast bundles.

Wraps the existing ``backend/artifacts/signing.py`` signer/verifier so server
bundles inherit the same tested Ed25519 infrastructure.  Verification always
fails closed: there is no dev/no-key mode, an unsigned or tampered bundle can
never be loaded.
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Any

from artifacts.signing import Ed25519ManifestSigner, Ed25519ManifestVerifier

MANIFEST_SCHEMA_VERSION = 1
SIGNATURE_ALGORITHM = "ed25519"
DIGEST_ALGORITHM = "sha256"


class ServerArtifactIntegrityError(RuntimeError):
    """A bundle digest or signature could not be verified."""


def sha256_digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sign_bundle(bundle_json_bytes: bytes, signer: Ed25519ManifestSigner) -> dict[str, Any]:
    """Sign a JSON bundle and return its portable manifest.

    The manifest carries the SHA-256 digest of the exact bytes plus an Ed25519
    signature computed over those same bytes.
    """

    if not isinstance(bundle_json_bytes, bytes | bytearray) or not bundle_json_bytes:
        raise ServerArtifactIntegrityError("Bundle payload must be non-empty bytes.")
    payload = bytes(bundle_json_bytes)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "signature_algorithm": SIGNATURE_ALGORITHM,
        "digest_algorithm": DIGEST_ALGORITHM,
        "sha256": sha256_digest(payload),
        "signature": signer(payload),
    }


def verify_bundle(
    bundle_json_bytes: bytes,
    manifest: dict[str, Any],
    verifier: Ed25519ManifestVerifier,
) -> None:
    """Verify digest and signature before a bundle may be used.

    Raises :class:`ServerArtifactIntegrityError` on any failure; verification
    never degrades to an insecure mode.
    """

    if not isinstance(bundle_json_bytes, bytes | bytearray):
        raise ServerArtifactIntegrityError("Bundle payload must be bytes.")
    if not isinstance(manifest, dict):
        raise ServerArtifactIntegrityError("Bundle manifest is malformed.")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ServerArtifactIntegrityError("Bundle manifest schema is unsupported.")
    if manifest.get("signature_algorithm") != SIGNATURE_ALGORITHM:
        raise ServerArtifactIntegrityError("Bundle signature algorithm is unsupported.")
    expected_digest = manifest.get("sha256")
    signature = manifest.get("signature")
    if not isinstance(expected_digest, str) or not isinstance(signature, str) or not signature:
        raise ServerArtifactIntegrityError("Bundle manifest is missing digest or signature.")

    payload = bytes(bundle_json_bytes)
    actual_digest = sha256_digest(payload)
    if not hmac.compare_digest(actual_digest, expected_digest):
        raise ServerArtifactIntegrityError("Bundle digest mismatch: payload was modified.")
    if not verifier(payload, signature):
        raise ServerArtifactIntegrityError("Bundle signature is invalid.")
