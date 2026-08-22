"""Signed, versioned global-model release bundles (slice 11).

A release directory contains:
- model artifacts (any files the runtime needs, e.g. TF.js topology+shards)
- catalog.json: pinned download metadata for the deployment build
- manifest.json: schema/target versions, blend weights per horizon,
  universe description, time boundaries, metrics+CIs, git SHA, snapshot
  digests, per-file sha256 checksums
- manifest.sig: Ed25519 signature over the exact manifest bytes

Fail-closed contract: verification requires the public key, an intact
signature over the exact manifest bytes, and every file's sha256 to match.
Any missing piece raises.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

RELEASE_SCHEMA_VERSION = 1


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_release(
    out_dir: Path,
    model_files: dict[str, bytes],
    metadata: dict,
    *,
    private_key_path: Path,
) -> Path:
    """Assemble + sign a release bundle. Overwrites are refused."""
    from artifacts.signing import Ed25519ManifestSigner

    if not model_files:
        raise ValueError("release requires at least one model file")
    out_dir.mkdir(parents=True, exist_ok=False)

    file_checksums: dict[str, str] = {}
    for name in sorted(model_files):
        data = model_files[name]
        target_path = out_dir / name
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(data)
        file_checksums[name] = _sha256_bytes(data)

    manifest = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "metadata": metadata,
        "files": file_checksums,
    }
    canonical_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    signer = Ed25519ManifestSigner.from_pem_file(private_key_path)
    signature = signer(canonical_bytes)

    # Write detached signature and embedded manifest
    (out_dir / "manifest.sig").write_text(signature, encoding="utf-8")
    manifest["signature"] = signature
    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    (out_dir / "manifest.json").write_bytes(manifest_bytes)
    return out_dir


def build_catalog(
    artifacts: list[dict],
    *,
    signature: str,
    recorded_sha: str,
    schema_version: int = 1,
) -> dict:
    """Build deployment catalog.json metadata."""
    if not artifacts:
        raise ValueError("catalog requires at least one artifact entry")
    if not signature or len(signature) < 64:
        raise ValueError("catalog requires a valid 64+ char signature")
    if not recorded_sha or len(recorded_sha) < 7:
        raise ValueError("catalog requires a valid git SHA of at least 7 chars")
    return {
        "schema_version": schema_version,
        "recorded_sha": recorded_sha,
        "signature": signature,
        "artifacts": artifacts,
    }


def verify_release(release_dir: Path, *, public_key_path: Path) -> dict:
    """Verify signature + all file checksums; returns the manifest.

    Raises on: missing key/manifest/signature, invalid signature, any file
    checksum mismatch, unsupported schema, or any listed file absent from disk.
    """
    import base64

    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import serialization

    manifest_path = release_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest file at {manifest_path}")

    raw = manifest_path.read_bytes()
    try:
        manifest = json.loads(raw)
    except Exception as exc:
        raise ValueError("Failed to parse manifest.json") from exc

    if manifest.get("schema_version") != RELEASE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported release schema version: {manifest.get('schema_version')}")

    signature_b64 = manifest.pop("signature", None)
    if not signature_b64:
        raise ValueError("manifest.json has no signature.")

    if not public_key_path.exists():
        raise FileNotFoundError(f"Missing public key at {public_key_path}")

    key = serialization.load_pem_public_key(public_key_path.read_bytes())
    canonical = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    try:
        key.verify(base64.b64decode(signature_b64), canonical)
    except InvalidSignature as exc:
        raise ValueError("release signature verification failed.") from exc

    for name, expected in manifest["files"].items():
        path = release_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Listed release file absent from disk: {name}")
        actual = _sha256_bytes(path.read_bytes())
        if actual != expected:
            raise ValueError(f"checksum mismatch for {name}: tampered or truncated.")

    manifest["signature"] = signature_b64
    return manifest
