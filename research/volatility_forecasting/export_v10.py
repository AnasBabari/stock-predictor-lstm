"""ONNX export verification and release bundle assembly for StockLSTM V10."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from research.volatility_forecasting.signing_v10 import (
    ReleaseSignatureError,
    verify_detached_signature,
)


@dataclass(frozen=True)
class ReleaseBundleManifestV10:
    bundle_id: str
    protocol_version: str
    certified_horizons: list[int]
    model_family_by_horizon: dict[str, str]
    feature_schema_sha256: str
    universe_sha256: str
    created_at_utc: str
    checksums: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assemble_release_bundle(
    output_dir: Path,
    bundle_id: str,
    protocol_version: str,
    certified_horizons: list[int],
    model_family_by_horizon: dict[int, str],
    feature_schema_sha256: str,
    universe_sha256: str,
    files_to_include: dict[str, bytes],
) -> Path:
    """Assemble an immutable release bundle directory with manifest and checksums."""
    target_dir = Path(output_dir) / bundle_id
    target_dir.mkdir(parents=True, exist_ok=True)

    checksums = {}
    for rel_path, content in files_to_include.items():
        file_path = target_dir / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(content)
        checksums[rel_path] = hashlib.sha256(content).hexdigest()

    manifest = ReleaseBundleManifestV10(
        bundle_id=bundle_id,
        protocol_version=protocol_version,
        certified_horizons=certified_horizons,
        model_family_by_horizon={str(k): v for k, v in model_family_by_horizon.items()},
        feature_schema_sha256=feature_schema_sha256,
        universe_sha256=universe_sha256,
        created_at_utc="2026-08-29T21:00:00Z",
        checksums=checksums,
    )

    manifest_content = json.dumps(manifest.to_dict(), indent=2).encode("utf-8")
    (target_dir / "manifest.json").write_bytes(manifest_content)
    (target_dir / "checksums.json").write_text(json.dumps(checksums, indent=2), encoding="utf-8")

    return target_dir


def verify_release_bundle_integrity(
    bundle_dir: Path,
    public_key_pem: bytes | None = None,
) -> bool:
    """Verify all file checksums and optional detached signature in a release bundle."""
    target = Path(bundle_dir)
    manifest_file = target / "manifest.json"
    checksums_file = target / "checksums.json"

    if not manifest_file.exists() or not checksums_file.exists():
        raise ValueError(f"Invalid bundle: missing manifest or checksums in {target}")

    checksums = json.loads(checksums_file.read_text(encoding="utf-8"))
    for rel_path, expected_sha in checksums.items():
        p = target / rel_path
        if not p.exists():
            raise ValueError(f"Bundle missing file: {rel_path}")
        actual_sha = hashlib.sha256(p.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            raise ValueError(f"Checksum mismatch for {rel_path}: expected {expected_sha}, got {actual_sha}")

    sig_file = target / "signature.ed25519"
    if public_key_pem is not None:
        if not sig_file.exists():
            raise ReleaseSignatureError(f"Missing required signature.ed25519 in {target}")
        sig_bytes = sig_file.read_bytes()
        manifest_bytes = manifest_file.read_bytes()
        verify_detached_signature(manifest_bytes, sig_bytes, public_key_pem)

    return True
