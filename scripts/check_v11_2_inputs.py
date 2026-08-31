#!/usr/bin/env python3
"""Fail-closed preflight for the real V11.2 PIT64 training inputs.

This command performs no feature construction, training, encryption, or
holdout access.  It only proves that an operator supplied an audited PIT64
panel, signed market/PIT64 provenance receipts, and an external 32-byte key
before the preparation command is run.
The repository's secondary ``data/ndx100/cache`` is explicitly rejected.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.panel.features import DEPLOYABLE_FEATURE_COLUMNS_V5  # noqa: E402
from research.volatility_forecasting.v11_2_attestation import (  # noqa: E402
    AttestationError,
    verify_v11_2_inputs,
)
from research.volatility_forecasting.v11_2_protocol import (  # noqa: E402
    V11_2_HORIZONS,
    V11_2_PROTOCOL_ID,
    V112Protocol,
    feature_schema_digest,
)
from research.volatility_forecasting.v11_2_universe import (  # noqa: E402
    V112UniverseManifest,
    load_universe_manifest,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is missing or malformed") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _validate_panel(panel_path: Path, universe: V112UniverseManifest) -> dict[str, Any]:
    required = {"dates", "security_ids", "features", "returns", "rv", "feature_names", "horizons"}
    with np.load(panel_path, allow_pickle=False) as panel:
        missing = required.difference(panel.files)
        if missing:
            raise ValueError(f"panel is missing arrays: {sorted(missing)}")
        dates = [str(value) for value in panel["dates"].tolist()]
        security_ids = [str(value) for value in panel["security_ids"].tolist()]
        features = np.asarray(panel["features"], dtype=np.float32)
        returns = np.asarray(panel["returns"], dtype=np.float32)
        realized_variance = np.asarray(panel["rv"], dtype=np.float32)
        feature_names = [str(value) for value in panel["feature_names"].tolist()]
        horizons = [int(value) for value in panel["horizons"].tolist()]

    protocol = V112Protocol()
    if feature_names != list(DEPLOYABLE_FEATURE_COLUMNS_V5):
        raise ValueError("panel feature ordering does not match deployable_v5")
    if horizons != list(V11_2_HORIZONS):
        raise ValueError("panel horizons do not match the frozen V11.2 contract")
    if features.ndim != 3 or features.shape[1:] != (
        protocol.window_size,
        len(protocol.feature_names),
    ):
        raise ValueError("panel features must have shape [rows, 60, 26]")
    if returns.ndim != 2 or returns.shape[1] != len(V11_2_HORIZONS):
        raise ValueError("panel returns must have four horizon columns")
    if realized_variance.shape != returns.shape:
        raise ValueError("panel realized variance must match the returns shape")
    if not (len(dates) == len(security_ids) == len(features) == len(returns)):
        raise ValueError("panel identity and target arrays must have equal row counts")
    if not dates or any(not value.strip() for value in dates):
        raise ValueError("panel dates must be non-empty")
    if any(not value.strip() for value in security_ids):
        raise ValueError("panel security IDs must be non-empty")
    if not all(np.isfinite(values).all() for values in (features, returns, realized_variance)):
        raise ValueError("panel arrays must contain only finite values")
    if len(set(security_ids)) != universe.universe_size:
        raise ValueError("panel must contain every accepted PIT64 security")
    universe_ids = {security.security_id for security in universe.securities}
    if set(security_ids) != universe_ids:
        raise ValueError("panel security IDs do not exactly match the audited universe")
    if len(set(zip(security_ids, dates, strict=True))) != len(dates):
        raise ValueError("panel contains duplicate security/session observations")
    for value in dates:
        try:
            parsed = dt.date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("panel dates must use canonical ISO calendar form") from exc
        if parsed.isoformat() != value:
            raise ValueError("panel dates must use canonical ISO calendar form")
    return {
        "stock_origin_observations": len(dates),
        "unique_sessions": len(set(dates)),
        "security_count": len(set(security_ids)),
        "feature_count": features.shape[-1],
        "horizons": horizons,
    }


def check_inputs(
    *,
    panel_path: Path,
    universe_path: Path,
    key_path: Path,
    repository_root: Path,
    snapshot_manifest_path: Path | None = None,
    market_attestation_path: Path | None = None,
    market_public_key_path: Path | None = None,
    pit64_attestation_path: Path | None = None,
    pit64_public_key_path: Path | None = None,
    market_evidence_files: dict[str, Path] | None = None,
    pit64_evidence_files: dict[str, Path] | None = None,
    require_signed_attestations: bool = True,
) -> dict[str, Any]:
    """Return a readiness report without touching or decrypting holdout data.

    Signed input provenance is required by default.  Callers may explicitly
    pass ``require_signed_attestations=False`` only for a development-only
    structural diagnostic; that report must never be used to prepare or
    certify a production dataset.
    """
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    panel = panel_path.resolve()
    universe_file = universe_path.resolve()
    key = key_path.resolve()
    repo = repository_root.resolve()
    if _under(panel, repo / "data" / "ndx100" / "cache"):
        check("secondary_ndx100_cache_rejected", False, "V11.2 requires an audited PIT64 panel")
    else:
        check("secondary_ndx100_cache_rejected", True)
    check("panel_exists", panel.is_file(), str(panel))
    check("universe_exists", universe_file.is_file(), str(universe_file))
    check("external_key_path", not _under(key, repo), str(key))
    check("key_exists", key.is_file(), str(key))
    if key.is_file():
        check("key_length", key.stat().st_size == 32, f"bytes={key.stat().st_size}")
    else:
        check("key_length", False, "key is unavailable")

    universe: V112UniverseManifest | None = None
    if universe_file.is_file():
        try:
            universe = load_universe_manifest(universe_file)
            check("universe_manifest", universe.protocol_id == V11_2_PROTOCOL_ID)
            check("universe_size", universe.universe_size == V112Protocol().universe_size)
            check(
                "universe_certification_eligible",
                universe.certification_eligible,
                f"certification_eligible={universe.certification_eligible}",
            )
        except (OSError, ValueError, TypeError) as exc:
            check("universe_manifest", False, str(exc))
    else:
        check("universe_manifest", False, "universe manifest is unavailable")

    if panel.is_file() and universe is not None:
        try:
            panel_summary = _validate_panel(panel, universe)
            check("panel_schema_and_identity", True, json.dumps(panel_summary, sort_keys=True))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            check("panel_schema_and_identity", False, str(exc))
            panel_summary = {}
    else:
        check("panel_schema_and_identity", False, "panel or universe is unavailable")
        panel_summary = {}

    sidecar = panel.with_suffix(panel.suffix + ".manifest.json")
    if sidecar.is_file() and universe is not None:
        try:
            payload = _json_object(sidecar, "panel sidecar")
            check("panel_sidecar_protocol", payload.get("protocol_id") == V11_2_PROTOCOL_ID)
            check("panel_sidecar_schema", payload.get("schema_sha256") == feature_schema_digest())
            check(
                "panel_sidecar_universe",
                payload.get("universe_manifest_sha256") == universe.manifest_sha256,
            )
            check(
                "panel_sidecar_certification_eligible",
                payload.get("certification_eligible") is True,
                f"certification_eligible={payload.get('certification_eligible')!r}",
            )
            sidecar_rows = payload.get("stock_origin_observations")
            sidecar_sessions = payload.get("unique_sessions")
            check(
                "panel_sidecar_counts",
                sidecar_rows == panel_summary.get("stock_origin_observations")
                and sidecar_sessions == panel_summary.get("unique_sessions"),
            )
            snapshot_digest = payload.get("snapshot_manifest_sha256")
            check(
                "panel_sidecar_snapshot_digest",
                isinstance(snapshot_digest, str)
                and len(snapshot_digest) == 64
                and all(value in "0123456789abcdef" for value in snapshot_digest),
            )
            panel_digest = payload.get("panel_sha256")
            expected_panel_digest = _sha256(panel)
            panel_digest_valid = panel_digest == expected_panel_digest
            check(
                "panel_sidecar_panel_digest",
                panel_digest_valid if require_signed_attestations else (panel_digest is None or panel_digest_valid),
                "panel bytes do not match the sidecar digest"
                if panel_digest is not None and not panel_digest_valid
                else "panel_sha256 is required for signed certification inputs"
                if require_signed_attestations and panel_digest is None
                else "",
            )
        except (OSError, ValueError, TypeError) as exc:
            check("panel_sidecar", False, str(exc))
    else:
        check("panel_sidecar", False, f"missing sidecar: {sidecar}")

    attestation_summary: dict[str, Any] = {}
    if require_signed_attestations:
        attestation_paths = (
            snapshot_manifest_path,
            market_attestation_path,
            market_public_key_path,
            pit64_attestation_path,
            pit64_public_key_path,
        )
        if any(path is None for path in attestation_paths):
            check(
                "signed_input_attestations",
                False,
                "snapshot manifest, two signed receipts, and two pinned public keys are required",
            )
        else:
            try:
                attestation_summary = verify_v11_2_inputs(
                    snapshot_manifest_path=snapshot_manifest_path,  # type: ignore[arg-type]
                    universe_manifest_path=universe_file,
                    market_receipt_path=market_attestation_path,  # type: ignore[arg-type]
                    market_public_key_path=market_public_key_path,  # type: ignore[arg-type]
                    pit64_receipt_path=pit64_attestation_path,  # type: ignore[arg-type]
                    pit64_public_key_path=pit64_public_key_path,  # type: ignore[arg-type]
                    market_evidence_files=market_evidence_files or {},
                    pit64_evidence_files=pit64_evidence_files or {},
                )
                snapshot_digest = hashlib.sha256(
                    snapshot_manifest_path.read_bytes()  # type: ignore[union-attr]
                ).hexdigest()
                sidecar_payload = _json_object(sidecar, "panel sidecar")
                if sidecar_payload.get("snapshot_manifest_sha256") != snapshot_digest:
                    raise AttestationError("panel sidecar is not bound to the attested snapshot manifest")
                if sidecar_payload.get("attestation_summary") != attestation_summary:
                    raise AttestationError("panel sidecar attestation summary does not match receipts")
                check("signed_input_attestations", True)
            except (AttestationError, OSError, ValueError, TypeError) as exc:
                attestation_summary = {}
                check("signed_input_attestations", False, str(exc))
    ready = all(item["passed"] for item in checks)
    return {
        "protocol_id": V11_2_PROTOCOL_ID,
        "protocol_sha256": V112Protocol().digest(),
        "panel": str(panel),
        "universe_manifest": str(universe_file),
        "key_path": str(key),
        "ready": ready,
        "checks": checks,
        "panel_summary": panel_summary,
        "attestation_summary": attestation_summary,
    }


def _evidence_args(values: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name.strip() or not path.strip():
            raise SystemExit("evidence must use NAME=PATH")
        if name.strip() in parsed:
            raise SystemExit(f"duplicate evidence name: {name.strip()}")
        parsed[name.strip()] = Path(path).resolve()
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--universe-manifest", type=Path, required=True)
    parser.add_argument("--key-path", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--snapshot-manifest", type=Path)
    parser.add_argument("--market-attestation", type=Path)
    parser.add_argument("--market-public-key", type=Path)
    parser.add_argument("--pit64-attestation", type=Path)
    parser.add_argument("--pit64-public-key", type=Path)
    parser.add_argument("--market-evidence", action="append", default=[])
    parser.add_argument("--pit64-evidence", action="append", default=[])
    args = parser.parse_args()
    report = check_inputs(
        panel_path=args.panel,
        universe_path=args.universe_manifest,
        key_path=args.key_path,
        repository_root=args.repository_root,
        snapshot_manifest_path=args.snapshot_manifest,
        market_attestation_path=args.market_attestation,
        market_public_key_path=args.market_public_key,
        pit64_attestation_path=args.pit64_attestation,
        pit64_public_key_path=args.pit64_public_key,
        market_evidence_files=_evidence_args(args.market_evidence),
        pit64_evidence_files=_evidence_args(args.pit64_evidence),
        require_signed_attestations=True,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
