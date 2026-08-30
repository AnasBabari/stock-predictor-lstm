"""Fail-closed audit that a V11.2 candidate is ready before holdout opening."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from research.volatility_forecasting.v11_2_protocol import V112Protocol, canonical_json_digest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = V112Protocol()
    dataset = args.dataset_dir
    results = args.results_dir
    metadata_path = dataset / "sealed" / "sealed_metadata.json"
    payload_path = dataset / "sealed" / "test_payload.aesgcm"
    lock_path = dataset / "sealed" / "SEALED_TEST_OPENED.json"
    bundle_path = results / "v11_2_routing_bundle.json"
    bundle_sha_path = results / "v11_2_routing_bundle.sha256"
    dev_manifest_path = dataset / "manifests" / "development_manifest.json"
    required = [metadata_path, payload_path, bundle_path, bundle_sha_path, dev_manifest_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"missing pre-unseal artifact(s): {missing}")
    if lock_path.exists():
        raise SystemExit("sealed test is already open; candidate cannot be certified")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest = json.loads(bundle_path.read_text(encoding="utf-8"))
    development = json.loads(dev_manifest_path.read_text(encoding="utf-8"))
    ciphertext_sha = _sha256(payload_path)
    if ciphertext_sha != metadata.get("ciphertext_sha256"):
        raise SystemExit("sealed ciphertext digest does not match metadata")
    if development.get("sealed_test_status") != "LOCKED_UNOPENED":
        raise SystemExit("development manifest does not prove an unopened holdout")
    if manifest.get("sealed_test_status") != "LOCKED_UNOPENED":
        raise SystemExit("routing bundle does not prove an unopened holdout")
    if manifest.get("protocol", {}).get("protocol_id") != protocol.protocol_id:
        raise SystemExit("routing bundle protocol mismatch")
    routes = manifest.get("routes", [])
    if sorted(route.get("horizon") for route in routes) != list(protocol.horizons):
        raise SystemExit("routing bundle does not contain exactly one route per horizon")
    expected_bundle_sha = bundle_sha_path.read_text(encoding="ascii").strip()
    actual_bundle_sha = canonical_json_digest(
        {key: value for key, value in manifest.items() if key != "master_freeze_sha256"}
    )
    if expected_bundle_sha != actual_bundle_sha:
        raise SystemExit("routing bundle digest does not match its canonical manifest")

    audit = {
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.digest(),
        "routing_bundle_sha256": expected_bundle_sha,
        "sealed_ciphertext_sha256": ciphertext_sha,
        "test_stock_origin_observations": metadata.get("test_stock_origin_observations"),
        "test_unique_sessions": metadata.get("test_unique_sessions"),
        "test_sessions": metadata.get("test_sessions"),
        "sealed_test_status": "LOCKED_UNOPENED",
        "audited_at": datetime.now(UTC).isoformat(),
        "decryption_performed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
