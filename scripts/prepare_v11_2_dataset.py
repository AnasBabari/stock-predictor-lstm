"""Retired V11.2 preparation entry point; every invocation fails closed.

V11.2 is permanently INVALIDATED_OPENED. The historical sealing implementation
is retained for auditability, but no new V11.2 dataset or reserve can be made.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "research"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from research.volatility_forecasting.v11_2_attestation import (  # noqa: E402
    AttestationError,
    verify_v11_2_inputs,
)
from research.volatility_forecasting.v11_2_protocol import (  # noqa: E402
    V112Protocol,
    assert_v11_2_certification_active,
    canonical_json_digest,
    feature_schema_digest,
    protocol_manifest,
)
from research.volatility_forecasting.v11_2_sealed_store import seal_v112_dataset  # noqa: E402
from research.volatility_forecasting.v11_2_split import (  # noqa: E402
    create_v112_split,
    save_split_manifest,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _evidence_args(values: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name.strip() or not path.strip():
            raise SystemExit("evidence must use NAME=PATH")
        normalized = name.strip()
        if normalized in parsed:
            raise SystemExit(f"duplicate evidence name: {normalized}")
        parsed[normalized] = Path(path).resolve()
    return parsed


def _copy_attestation_inputs(
    *,
    output_dir: Path,
    market_receipt: Path,
    market_public_key: Path,
    market_evidence: dict[str, Path],
    pit64_receipt: Path,
    pit64_public_key: Path,
    pit64_evidence: dict[str, Path],
    summary: dict[str, object],
) -> None:
    """Copy the verified evidence into the immutable dataset namespace."""

    record_path = output_dir / "manifests" / "attestations.json"
    if record_path.exists():
        raise SystemExit("dataset attestation record already exists and is immutable")
    root = output_dir / "manifests" / "attestations"
    evidence_root = root / "evidence"
    evidence_root.mkdir(parents=True, exist_ok=True)

    def copy_one(source: Path, target: Path) -> str:
        if not source.is_file():
            raise SystemExit(f"attestation input is missing: {source}")
        shutil.copyfile(source, target)
        return target.relative_to(output_dir).as_posix()

    market_evidence_paths: dict[str, str] = {}
    for name, source in sorted(market_evidence.items()):
        market_evidence_paths[name] = copy_one(source, evidence_root / f"market-{name}")
    pit64_evidence_paths: dict[str, str] = {}
    for name, source in sorted(pit64_evidence.items()):
        pit64_evidence_paths[name] = copy_one(source, evidence_root / f"pit64-{name}")
    body = {
        "schema_version": 1,
        "market": {
            "receipt": copy_one(market_receipt, root / "market_receipt.json"),
            "public_key": copy_one(market_public_key, root / "market_public_key.pem"),
            "evidence": market_evidence_paths,
        },
        "pit64": {
            "receipt": copy_one(pit64_receipt, root / "pit64_receipt.json"),
            "public_key": copy_one(pit64_public_key, root / "pit64_public_key.pem"),
            "evidence": pit64_evidence_paths,
        },
        "verification": summary,
    }
    body["record_sha256"] = canonical_json_digest(body)
    (output_dir / "manifests" / "attestations.json").write_text(
        json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _verify_panel_binding(
    *,
    panel_path: Path,
    universe_payload: dict[str, object],
    snapshot_manifest_path: Path,
    attestation_summary: dict[str, object],
) -> None:
    """Require the panel sidecar to bind bytes to the verified input pair."""

    sidecar_path = panel_path.with_suffix(panel_path.suffix + ".manifest.json")
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit("certification-eligible panels require a valid panel sidecar") from exc
    if not isinstance(payload, dict):
        raise SystemExit("panel sidecar must contain a JSON object")
    expected_panel_sha = _sha256_file(panel_path)
    if payload.get("panel_sha256") != expected_panel_sha:
        raise SystemExit("panel sidecar is not bound to the panel bytes")
    expected_snapshot_sha = _sha256_file(snapshot_manifest_path)
    if payload.get("snapshot_manifest_sha256") != expected_snapshot_sha:
        raise SystemExit("panel sidecar is not bound to the attested snapshot manifest")
    if payload.get("universe_manifest_sha256") != universe_payload.get("manifest_sha256"):
        raise SystemExit("panel sidecar is not bound to the audited universe manifest")
    if payload.get("certification_eligible") is not True:
        raise SystemExit("panel sidecar is not certification-eligible")
    if payload.get("attestation_summary") != attestation_summary:
        raise SystemExit("panel sidecar attestation summary does not match the receipts")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--universe-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--key-path", type=Path, required=True)
    parser.add_argument("--schema-sha256", required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--snapshot-manifest", type=Path)
    parser.add_argument("--market-attestation", type=Path)
    parser.add_argument("--market-public-key", type=Path)
    parser.add_argument("--pit64-attestation", type=Path)
    parser.add_argument("--pit64-public-key", type=Path)
    parser.add_argument("--market-evidence", action="append", default=[])
    parser.add_argument("--pit64-evidence", action="append", default=[])
    args = parser.parse_args()
    assert_v11_2_certification_active()

    if _under(args.panel, ROOT / "data" / "ndx100" / "cache"):
        raise SystemExit("the secondary data/ndx100/cache cannot be prepared for V11.2")

    protocol = V112Protocol()
    schema_sha = str(args.schema_sha256)
    if len(schema_sha) != 64 or any(value not in "0123456789abcdef" for value in schema_sha):
        raise SystemExit("--schema-sha256 must be a lowercase SHA-256 digest")
    if schema_sha != feature_schema_digest(protocol):
        raise SystemExit("--schema-sha256 does not match the frozen V11.2 feature contract")
    universe_payload = json.loads(args.universe_manifest.read_text(encoding="utf-8"))
    if universe_payload.get("protocol_id") != protocol.protocol_id:
        raise SystemExit("universe manifest protocol does not match V11.2")
    if int(universe_payload.get("universe_size", 0)) != protocol.universe_size:
        raise SystemExit("universe manifest must contain exactly 64 accepted securities")
    universe_sha = str(universe_payload.get("manifest_sha256", ""))
    if len(universe_sha) != 64 or any(value not in "0123456789abcdef" for value in universe_sha):
        raise SystemExit("universe manifest must contain a SHA-256 manifest digest")
    digest_payload = {
        key: universe_payload[key]
        for key in (
            "protocol_id",
            "universe_version",
            "selection_method",
            "membership_sources",
            "certification_eligible",
            "securities",
        )
        if key in universe_payload
    }
    if canonical_json_digest(digest_payload) != universe_sha:
        raise SystemExit("universe manifest content does not match its manifest digest")
    manifest_ids = [
        str(item.get("security_id", "")) for item in universe_payload.get("securities", [])
    ]
    if len(manifest_ids) != protocol.universe_size or not all(manifest_ids):
        raise SystemExit("universe manifest must list 64 non-empty permanent security IDs")
    if len(set(manifest_ids)) != len(manifest_ids):
        raise SystemExit("universe manifest contains duplicate permanent security IDs")

    attestation_summary: dict[str, object] = {}
    if universe_payload.get("certification_eligible") is True:
        required = (
            args.snapshot_manifest,
            args.market_attestation,
            args.market_public_key,
            args.pit64_attestation,
            args.pit64_public_key,
        )
        if any(path is None for path in required):
            raise SystemExit(
                "certification-eligible datasets require a snapshot manifest, two signed "
                "attestations, and two pinned public keys"
            )
        try:
            attestation_summary = verify_v11_2_inputs(
                snapshot_manifest_path=args.snapshot_manifest,  # type: ignore[arg-type]
                universe_manifest_path=args.universe_manifest.resolve(),
                market_receipt_path=args.market_attestation,  # type: ignore[arg-type]
                market_public_key_path=args.market_public_key,  # type: ignore[arg-type]
                pit64_receipt_path=args.pit64_attestation,  # type: ignore[arg-type]
                pit64_public_key_path=args.pit64_public_key,  # type: ignore[arg-type]
                market_evidence_files=_evidence_args(args.market_evidence),
                pit64_evidence_files=_evidence_args(args.pit64_evidence),
            )
        except (AttestationError, OSError, ValueError, TypeError) as exc:
            raise SystemExit(f"signed V11.2 input attestation failed: {exc}") from exc

    with np.load(args.panel, allow_pickle=False) as panel:
        required = {
            "dates",
            "security_ids",
            "features",
            "returns",
            "rv",
            "feature_names",
            "horizons",
        }
        missing = required.difference(panel.files)
        if missing:
            raise SystemExit(f"panel is missing arrays: {sorted(missing)}")
        dates = [str(value) for value in panel["dates"].tolist()]
        security_ids = [str(value) for value in panel["security_ids"].tolist()]
        features = np.asarray(panel["features"], dtype=np.float32)
        returns = np.asarray(panel["returns"], dtype=np.float32)
        rv = np.asarray(panel["rv"], dtype=np.float32)
        feature_names = [str(value) for value in panel["feature_names"].tolist()]
        horizons = [int(value) for value in panel["horizons"].tolist()]

    if feature_names != list(protocol.feature_names):
        raise SystemExit("panel feature_names do not match the frozen deployable_v5 ordering")
    if horizons != list(protocol.horizons):
        raise SystemExit("panel horizons do not match the frozen V11.2 horizon contract")
    if features.ndim != 3 or features.shape[1:] != (
        protocol.window_size,
        len(protocol.feature_names),
    ):
        raise SystemExit("panel features must have shape [rows, 60, 26]")
    if returns.ndim != 2 or returns.shape[1] != len(protocol.horizons):
        raise SystemExit("panel returns must have one column per V11.2 horizon")
    if rv.ndim != 2 or rv.shape != returns.shape:
        raise SystemExit("panel realized variance must match the returns shape")
    if not all(np.isfinite(values).all() for values in (features, returns, rv)):
        raise SystemExit("panel arrays must contain only finite numeric values")
    if not (len(dates) == len(security_ids) == len(features) == len(returns) == len(rv)):
        raise SystemExit("panel identity and target arrays must have equal row counts")

    if len(set(security_ids)) != protocol.universe_size:
        raise SystemExit("panel must contain all 64 accepted securities")
    if set(security_ids) != set(manifest_ids):
        raise SystemExit("panel security IDs do not exactly match the audited universe manifest")
    if attestation_summary:
        _verify_panel_binding(
            panel_path=args.panel.resolve(),
            universe_payload=universe_payload,
            snapshot_manifest_path=args.snapshot_manifest,  # type: ignore[arg-type]
            attestation_summary=attestation_summary,
        )
    split = create_v112_split(dates, security_ids)
    panel_sha = _sha256_file(args.panel)
    metadata = seal_v112_dataset(
        dates=dates,
        security_ids=security_ids,
        features=features,
        returns=returns,
        rv=rv,
        split=split,
        output_dir=args.output_dir,
        panel_sha256=panel_sha,
        schema_sha256=schema_sha,
        key_path=args.key_path,
        repository_root=args.repository_root,
    )
    if attestation_summary:
        _copy_attestation_inputs(
            output_dir=args.output_dir,
            market_receipt=args.market_attestation,  # type: ignore[arg-type]
            market_public_key=args.market_public_key,  # type: ignore[arg-type]
            market_evidence=_evidence_args(args.market_evidence),
            pit64_receipt=args.pit64_attestation,  # type: ignore[arg-type]
            pit64_public_key=args.pit64_public_key,  # type: ignore[arg-type]
            pit64_evidence=_evidence_args(args.pit64_evidence),
            summary=attestation_summary,
        )
    (args.output_dir / "manifests" / "protocol.json").write_text(
        json.dumps(protocol_manifest(protocol), indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.output_dir / "manifests" / "universe.json").write_text(
        json.dumps(universe_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    save_split_manifest(split, args.output_dir / "manifests" / "split.json")
    print(
        json.dumps(
            {
                "protocol_sha256": protocol.digest(),
                "universe_sha256": universe_sha,
                "panel_sha256": panel_sha,
                "split_sha256": split.split_sha256,
                "sealed_ciphertext_sha256": metadata.ciphertext_sha256,
                "test_stock_origin_observations": metadata.test_stock_origin_observations,
                "test_unique_sessions": metadata.test_unique_sessions,
                "sealed_test_status": "LOCKED_UNOPENED",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
