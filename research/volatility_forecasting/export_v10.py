"""ONNX export verification and immutable release bundle assembly for StockLSTM V10."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from research.volatility_forecasting.certification_v10 import CertificationReportV10
from research.volatility_forecasting.signing_v10 import (
    ReleaseSignatureError,
    verify_detached_signature,
)


class ReleasePathTraversalError(ValueError):
    """Raised when release bundle contains unsafe relative or path traversal filenames."""


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
    certification_report: CertificationReportV10 | dict[str, Any],
    protocol_version: str,
    model_family_by_horizon: dict[int, str],
    feature_schema_sha256: str,
    universe_sha256: str,
    files_to_include: dict[str, bytes],
) -> Path:
    """Assemble an immutable release bundle directory with manifest and checksums."""
    target_dir = Path(output_dir) / bundle_id
    if target_dir.exists():
        raise ValueError(
            f"Release bundle {bundle_id} already exists at {target_dir}. Overwriting forbidden."
        )

    # Validate certification report
    if isinstance(certification_report, CertificationReportV10):
        cert_horizons = list(certification_report.certified_horizons)
    else:
        cert_horizons = list(certification_report.get("certified_horizons", []))

    if not cert_horizons:
        raise ValueError(
            "Cannot assemble release bundle: certification report has 0 certified horizons."
        )

    # Validate filenames against path traversal
    for rel_path in files_to_include:
        if ".." in rel_path or rel_path.startswith("/") or rel_path.startswith("\\"):
            raise ReleasePathTraversalError(f"Unsafe path detected in release file: {rel_path}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="release_atomic_", dir=output_dir))
    try:
        checksums = {}
        for rel_path, content in files_to_include.items():
            file_path = tmp_dir / rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(content)
            checksums[rel_path] = hashlib.sha256(content).hexdigest()

        created_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        manifest = ReleaseBundleManifestV10(
            bundle_id=bundle_id,
            protocol_version=protocol_version,
            certified_horizons=cert_horizons,
            model_family_by_horizon={
                str(k): v for k, v in model_family_by_horizon.items() if int(k) in cert_horizons
            },
            feature_schema_sha256=feature_schema_sha256,
            universe_sha256=universe_sha256,
            created_at_utc=created_at,
            checksums=checksums,
        )

        manifest_content = json.dumps(manifest.to_dict(), indent=2).encode("utf-8")
        (tmp_dir / "manifest.json").write_bytes(manifest_content)
        (tmp_dir / "checksums.json").write_text(json.dumps(checksums, indent=2), encoding="utf-8")

        tmp_dir.rename(target_dir)
        return target_dir
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def verify_release_bundle_integrity(
    bundle_dir: Path,
    public_key_pem: bytes,
) -> bool:
    """Verify all file checksums and required detached signature in a release bundle."""
    target = Path(bundle_dir)
    manifest_file = target / "manifest.json"
    checksums_file = target / "checksums.json"
    sig_file = target / "signature.ed25519"

    if not manifest_file.exists() or not checksums_file.exists():
        raise ValueError(f"Invalid bundle: missing manifest or checksums in {target}")

    if not sig_file.exists():
        raise ReleaseSignatureError(f"Missing required signature.ed25519 in {target}")

    checksums = json.loads(checksums_file.read_text(encoding="utf-8"))
    for rel_path, expected_sha in checksums.items():
        p = target / rel_path
        if not p.exists():
            raise ValueError(f"Bundle missing file: {rel_path}")
        actual_sha = hashlib.sha256(p.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            raise ValueError(
                f"Checksum mismatch for {rel_path}: expected {expected_sha}, got {actual_sha}"
            )

    sig_bytes = sig_file.read_bytes()
    manifest_bytes = manifest_file.read_bytes()
    verify_detached_signature(manifest_bytes, sig_bytes, public_key_pem)

    return True
