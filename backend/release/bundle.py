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
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

RELEASE_SCHEMA_VERSION = 1
# Metadata schema consumed by the certified volatility ONNX serving runtime.
RUNTIME_SCHEMA_VERSION = "volatility-runtime-v1"
MAX_RELEASE_FILES = 32
MAX_RELEASE_FILE_BYTES = 64 * 1024 * 1024
MAX_RELEASE_TOTAL_BYTES = 128 * 1024 * 1024
MAX_RELEASE_MANIFEST_BYTES = 1024 * 1024


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _artifact_path(root: Path, name: object) -> Path:
    """Resolve one portable relative artifact path inside ``root``."""
    if not isinstance(name, str) or not name or len(name) > 240 or "\\" in name or ":" in name:
        raise ValueError("release artifact path is invalid")
    relative = PurePosixPath(name)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("release artifact path must stay inside the release directory")
    target = root.joinpath(*relative.parts).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("release artifact path escapes the release directory") from error
    return target


def build_release(
    out_dir: Path,
    model_files: dict[str, bytes],
    metadata: dict,
    *,
    private_key_path: Path,
    created_at_utc: datetime | None = None,
) -> Path:
    """Assemble + sign a release bundle. Overwrites are refused."""
    try:
        from artifacts.signing import Ed25519ManifestSigner
    except ImportError:  # pragma: no cover - repository-root contexts import the package path
        from backend.artifacts.signing import Ed25519ManifestSigner  # type: ignore[no-redef]

    if not model_files:
        raise ValueError("release requires at least one model file")
    if len(model_files) > MAX_RELEASE_FILES:
        raise ValueError("release contains too many artifact files")
    total_bytes = sum(len(data) for data in model_files.values())
    if total_bytes > MAX_RELEASE_TOTAL_BYTES or any(
        not isinstance(data, bytes) or not data or len(data) > MAX_RELEASE_FILE_BYTES
        for data in model_files.values()
    ):
        raise ValueError("release artifact files exceed the bounded size contract")
    out_dir.mkdir(parents=True, exist_ok=False)

    file_checksums: dict[str, str] = {}
    for name in sorted(model_files):
        data = model_files[name]
        target_path = _artifact_path(out_dir, name)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(data)
        file_checksums[name] = _sha256_bytes(data)

    created_at = created_at_utc or datetime.now(UTC)
    if created_at.tzinfo is None:
        raise ValueError("release creation timestamp must be timezone-aware")
    manifest = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "created_at_utc": created_at.astimezone(UTC).isoformat(),
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
    if len(raw) > MAX_RELEASE_MANIFEST_BYTES:
        raise ValueError("release manifest exceeds the bounded size contract")
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

    key: Any = serialization.load_pem_public_key(public_key_path.read_bytes())
    canonical = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    try:
        key.verify(base64.b64decode(signature_b64), canonical)
    except InvalidSignature as exc:
        raise ValueError("release signature verification failed.") from exc

    files = manifest.get("files")
    if not isinstance(files, dict) or not files or len(files) > MAX_RELEASE_FILES:
        raise ValueError("release file manifest is invalid")
    total_bytes = 0
    for name, expected in files.items():
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("release file checksum is invalid")
        path = _artifact_path(release_dir, name)
        if not path.exists():
            raise FileNotFoundError(f"Listed release file absent from disk: {name}")
        size = path.stat().st_size
        total_bytes += size
        if size <= 0 or size > MAX_RELEASE_FILE_BYTES or total_bytes > MAX_RELEASE_TOTAL_BYTES:
            raise ValueError("release artifact files exceed the bounded size contract")
        actual = _sha256_bytes(path.read_bytes())
        if actual != expected:
            raise ValueError(f"checksum mismatch for {name}: tampered or truncated.")

    manifest["signature"] = signature_b64
    return manifest


def validate_certification_manifest(cert_manifest: dict) -> bool:
    """Validates certification manifest according to declared protocol version.

    Fail-closed:
    - V1: Requires temporal_relative_rmse <= 1.0 and temporal_relative_mae <= 1.0
    - V2: Independently re-evaluates all mandatory gates declared in gate_config against
      the serialized metrics, asserts agreement between top-level and decision-level metadata,
      and verifies coverage requirements.
    - Missing required fields, unknown protocol, or any violated gate raises ValueError or returns False.
    """
    if not isinstance(cert_manifest, dict):
        raise ValueError("Certification manifest must be a dictionary.")

    status = cert_manifest.get("status")
    if status != "holdout_opened":
        raise ValueError(f"Holdout status must be 'holdout_opened', got '{status}'")

    protocol = cert_manifest.get("certification_protocol_version", "global-cert-v1")
    decisions = cert_manifest.get("decisions", {})
    if not isinstance(decisions, dict) or not decisions:
        raise ValueError("Certification manifest contains no horizon decisions.")

    if protocol == "global-cert-v1":
        for _h, dec in decisions.items():
            if not isinstance(dec, dict):
                return False
            if dec.get("decision") != "pass":
                return False
            if "temporal_relative_rmse" not in dec or "temporal_relative_mae" not in dec:
                return False
            if float(dec["temporal_relative_rmse"]) > 1.000001:
                return False
            if float(dec["temporal_relative_mae"]) > 1.000001:
                return False
        return True

    elif protocol == "global-cert-v2":
        gate_config = cert_manifest.get("gate_config")
        if not isinstance(gate_config, dict):
            raise ValueError("V2 certification manifest missing required 'gate_config' dict.")

        for _h, dec in decisions.items():
            if not isinstance(dec, dict):
                return False
            if dec.get("decision") != "pass":
                return False
            if dec.get("failed_gates"):
                return False

            # Verify consistency between top-level and decision-level metadata
            dec_gate_cfg = dec.get("gate_config")
            if dec_gate_cfg is not None and dec_gate_cfg != gate_config:
                return False
            if (
                dec.get("certification_protocol_version")
                and dec.get("certification_protocol_version") != protocol
            ):
                return False

            # Independently evaluate every declared mandatory gate:
            # 1. Temporal Relative RMSE
            if gate_config.get("require_temporal_relative_rmse", True):
                if "temporal_relative_rmse" not in dec:
                    raise ValueError("Decision missing required 'temporal_relative_rmse' metric.")
                max_rmse = float(gate_config.get("max_temporal_relative_rmse", 1.00))
                if float(dec["temporal_relative_rmse"]) > max_rmse + 1e-9:
                    return False

            # 2. Temporal Relative MAE
            if gate_config.get("require_temporal_relative_mae", True):
                if "temporal_relative_mae" not in dec:
                    raise ValueError("Decision missing required 'temporal_relative_mae' metric.")
                max_mae = float(gate_config.get("max_temporal_relative_mae", 1.00))
                if float(dec["temporal_relative_mae"]) > max_mae + 1e-9:
                    return False

            # 3. Asset-Transfer Relative RMSE
            if gate_config.get("require_transfer_relative_rmse", False):
                if "transfer_relative_rmse" not in dec:
                    raise ValueError("Decision missing required 'transfer_relative_rmse' metric.")
                max_trans_rmse = float(gate_config.get("max_transfer_relative_rmse", 1.00))
                if float(dec["transfer_relative_rmse"]) > max_trans_rmse + 1e-9:
                    return False

            # 4. Asset-Transfer Relative MAE
            if gate_config.get("require_transfer_relative_mae", False):
                if "transfer_relative_mae" not in dec:
                    raise ValueError("Decision missing required 'transfer_relative_mae' metric.")
                max_trans_mae = float(gate_config.get("max_transfer_relative_mae", 1.00))
                if float(dec["transfer_relative_mae"]) > max_trans_mae + 1e-9:
                    return False

            # 5. Direction skill vs majority on non-neutral subset
            if gate_config.get("require_direction_skill", False):
                if "direction_accuracy_delta_vs_majority" not in dec:
                    raise ValueError(
                        "Decision missing required 'direction_accuracy_delta_vs_majority' metric."
                    )
                min_delta = float(gate_config.get("min_direction_accuracy_delta_vs_majority", 0.00))
                if float(dec["direction_accuracy_delta_vs_majority"]) < min_delta - 1e-9:
                    return False

            # 6. Probabilistic direction
            if gate_config.get("require_probabilistic_direction", False):
                prob_status = dec.get("direction_probability_status")
                if prob_status != "evaluated":
                    return False
                if "temporal_brier" not in dec or dec["temporal_brier"] is None:
                    raise ValueError("Decision missing required 'temporal_brier' metric.")
                max_brier = gate_config.get("max_direction_brier")
                if max_brier is not None and float(dec["temporal_brier"]) > float(max_brier) + 1e-9:
                    return False
                prob_cov = float(dec.get("direction_probability_coverage", 0.0))
                if prob_cov < 0.999:
                    return False

        return True

    elif protocol == "global-cert-v3":
        gate_config = cert_manifest.get("gate_config")
        if not isinstance(gate_config, dict):
            raise ValueError("V3 certification manifest missing required 'gate_config' dict.")

        family_alpha = float(gate_config.get("family_alpha", 0.05))

        for _h, dec in decisions.items():
            if not isinstance(dec, dict):
                return False
            if dec.get("decision") != "pass":
                return False
            if dec.get("failed_gates"):
                return False

            # Metadata consistency check
            dec_gate_cfg = dec.get("gate_config")
            if dec_gate_cfg is not None and dec_gate_cfg != gate_config:
                return False

            # Temporal rank IC checks
            if gate_config.get("require_temporal_mean_ic_positive", True):
                if "temporal_mean_ic" not in dec or dec["temporal_mean_ic"] is None:
                    raise ValueError("Decision missing required 'temporal_mean_ic' metric.")
                if float(dec["temporal_mean_ic"]) <= 0.0:
                    return False

            if gate_config.get("require_temporal_bootstrap_lower_bound_positive", True):
                if (
                    "temporal_mean_ic_ci_lower_95" not in dec
                    or dec["temporal_mean_ic_ci_lower_95"] is None
                ):
                    raise ValueError(
                        "Decision missing required 'temporal_mean_ic_ci_lower_95' metric."
                    )
                if float(dec["temporal_mean_ic_ci_lower_95"]) <= 0.0:
                    return False

            if gate_config.get("require_temporal_holm_hac_significance", True):
                if "temporal_holm_hac_p" not in dec or dec["temporal_holm_hac_p"] is None:
                    raise ValueError("Decision missing required 'temporal_holm_hac_p' metric.")
                if float(dec["temporal_holm_hac_p"]) > family_alpha + 1e-9:
                    return False

            min_temp_cov = float(gate_config.get("min_temporal_prediction_coverage", 0.90))
            if float(dec.get("temporal_prediction_coverage", 0.0)) < min_temp_cov - 1e-9:
                return False

            min_temp_sess = float(gate_config.get("min_temporal_ic_session_coverage", 0.90))
            if float(dec.get("temporal_session_coverage", 0.0)) < min_temp_sess - 1e-9:
                return False

            min_temp_br = int(gate_config.get("min_temporal_daily_breadth", 30))
            if float(dec.get("temporal_median_breadth", 0.0)) < min_temp_br:
                return False

            # Transfer rank IC checks
            if gate_config.get("require_transfer_mean_ic_positive", True):
                if "transfer_mean_ic" not in dec or dec["transfer_mean_ic"] is None:
                    raise ValueError("Decision missing required 'transfer_mean_ic' metric.")
                if float(dec["transfer_mean_ic"]) <= 0.0:
                    return False

            if gate_config.get("require_transfer_bootstrap_lower_bound_positive", True):
                if (
                    "transfer_mean_ic_ci_lower_95" not in dec
                    or dec["transfer_mean_ic_ci_lower_95"] is None
                ):
                    raise ValueError(
                        "Decision missing required 'transfer_mean_ic_ci_lower_95' metric."
                    )
                if float(dec["transfer_mean_ic_ci_lower_95"]) <= 0.0:
                    return False

            if gate_config.get("require_transfer_holm_hac_significance", True):
                if "transfer_holm_hac_p" not in dec or dec["transfer_holm_hac_p"] is None:
                    raise ValueError("Decision missing required 'transfer_holm_hac_p' metric.")
                if float(dec["transfer_holm_hac_p"]) > family_alpha + 1e-9:
                    return False

            min_trans_cov = float(gate_config.get("min_transfer_prediction_coverage", 0.90))
            if float(dec.get("transfer_prediction_coverage", 0.0)) < min_trans_cov - 1e-9:
                return False

            min_trans_sess = float(gate_config.get("min_transfer_ic_session_coverage", 0.90))
            if float(dec.get("transfer_session_coverage", 0.0)) < min_trans_sess - 1e-9:
                return False

            min_trans_br = int(gate_config.get("min_transfer_daily_breadth", 30))
            if float(dec.get("transfer_median_breadth", 0.0)) < min_trans_br:
                return False

        return True

    else:
        raise ValueError(f"Unknown or unsupported certification protocol: '{protocol}'")
