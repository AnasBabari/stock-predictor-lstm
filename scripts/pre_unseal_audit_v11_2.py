"""Fail-closed audit that a V11.2 candidate is ready before holdout opening."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "research"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from research.volatility_forecasting.v11_2_protocol import (  # noqa: E402
    V112Protocol,
    canonical_json_digest,
    feature_schema_digest,
    protocol_manifest,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, label: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise SystemExit(f"{label} is not a lowercase SHA-256 digest")
    return digest


def _json_object(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"{label} is malformed") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} must contain a JSON object")
    return payload


def _required_file(root: Path, relative: object, label: str) -> Path:
    if isinstance(relative, Path):
        candidate = relative
        name = relative.as_posix()
    elif isinstance(relative, str):
        name = relative
        candidate = Path(name)
    else:
        raise SystemExit(f"{label} must be a relative artifact path")
    if not name or "\x00" in name or candidate.is_absolute():
        raise SystemExit(f"{label} must be a relative artifact path")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise SystemExit(f"{label} escapes the results directory") from exc
    if not resolved.is_file():
        raise SystemExit(f"{label} is missing: {name}")
    return resolved


def _verify_self_digest(path: Path, digest_field: str, label: str) -> tuple[str, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} must contain a JSON object")
    declared = _require_sha256(payload.pop(digest_field, None), f"{label} {digest_field}")
    if canonical_json_digest(payload) != declared:
        raise SystemExit(f"{label} digest does not match its canonical contents")
    return declared, payload


def audit_pre_unseal(dataset: Path, results: Path) -> dict[str, object]:
    """Verify every immutable development artifact without decrypting the test."""
    protocol = V112Protocol()
    metadata_path = dataset / "sealed" / "sealed_metadata.json"
    payload_path = dataset / "sealed" / "test_payload.aesgcm"
    lock_path = dataset / "sealed" / "SEALED_TEST_OPENED.json"
    bundle_path = results / "v11_2_routing_bundle.json"
    bundle_sha_path = results / "v11_2_routing_bundle.sha256"
    dev_manifest_path = dataset / "manifests" / "development_manifest.json"
    protocol_path = dataset / "manifests" / "protocol.json"
    universe_path = dataset / "manifests" / "universe.json"
    split_path = dataset / "manifests" / "split.json"
    train_path = dataset / "development" / "train.npz"
    validation_path = dataset / "development" / "validation.npz"
    required = [
        metadata_path,
        payload_path,
        bundle_path,
        bundle_sha_path,
        dev_manifest_path,
        protocol_path,
        universe_path,
        split_path,
        train_path,
        validation_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"missing pre-unseal artifact(s): {missing}")
    if lock_path.exists():
        raise SystemExit("sealed test is already open; candidate cannot be certified")

    metadata = _json_object(metadata_path, "sealed metadata")
    manifest = _json_object(bundle_path, "routing bundle")
    development = _json_object(dev_manifest_path, "development manifest")
    stored_protocol = _json_object(protocol_path, "protocol manifest")
    universe = _json_object(universe_path, "universe manifest")
    split = _json_object(split_path, "split manifest")
    if canonical_json_digest(stored_protocol) != canonical_json_digest(protocol_manifest(protocol)):
        raise SystemExit("stored protocol manifest does not match V11.2")
    if development.get("protocol_id") != protocol.protocol_id:
        raise SystemExit("development manifest protocol does not match V11.2")
    if metadata.get("protocol_id") != protocol.protocol_id:
        raise SystemExit("sealed metadata protocol does not match V11.2")
    if metadata.get("sealed_test_status") != "LOCKED_UNOPENED":
        raise SystemExit("sealed metadata does not prove an unopened holdout")
    metadata_panel = _require_sha256(metadata.get("panel_sha256"), "sealed metadata panel digest")
    metadata_schema = _require_sha256(
        metadata.get("schema_sha256"), "sealed metadata schema digest"
    )
    metadata_split = _require_sha256(metadata.get("split_sha256"), "sealed metadata split digest")
    if metadata_schema != feature_schema_digest(protocol):
        raise SystemExit("sealed metadata schema digest does not match V11.2")
    if metadata_panel != development.get("panel_sha256"):
        raise SystemExit("sealed metadata panel digest differs from development evidence")
    if metadata_split != development.get("split_sha256"):
        raise SystemExit("sealed metadata split digest differs from development evidence")
    universe_digest = _require_sha256(universe.get("manifest_sha256"), "universe manifest digest")
    universe_body = {
        key: universe[key]
        for key in (
            "protocol_id",
            "universe_version",
            "selection_method",
            "membership_sources",
            "securities",
        )
        if key in universe
    }
    if canonical_json_digest(universe_body) != universe_digest:
        raise SystemExit("universe manifest digest does not match its canonical contents")
    if (
        universe.get("protocol_id") != protocol.protocol_id
        or universe.get("universe_size") != protocol.universe_size
    ):
        raise SystemExit("universe manifest does not contain the frozen 64-security universe")
    securities = universe.get("securities")
    if not isinstance(securities, list) or len(securities) != protocol.universe_size:
        raise SystemExit("universe manifest must list exactly 64 securities")
    security_ids = [item.get("security_id") for item in securities if isinstance(item, dict)]
    if (
        len(security_ids) != protocol.universe_size
        or any(not isinstance(value, str) or not value.strip() for value in security_ids)
        or len(set(security_ids)) != protocol.universe_size
    ):
        raise SystemExit("universe manifest security IDs are invalid")
    if universe_digest != manifest.get("universe_sha256"):
        raise SystemExit("routing bundle universe digest differs from the audited manifest")
    if split.get("protocol") != protocol.protocol_id or split.get("nominal_split") != "70/15/15":
        raise SystemExit("split manifest does not match the frozen V11.2 protocol")
    split_digest = _require_sha256(split.get("split_sha256"), "split manifest digest")
    if split_digest != metadata_split or split_digest != manifest.get("split_sha256"):
        raise SystemExit("split manifest digest differs from the sealed evidence")
    ciphertext_sha = _sha256(payload_path)
    metadata_ciphertext = _require_sha256(
        metadata.get("ciphertext_sha256"), "sealed metadata ciphertext digest"
    )
    if ciphertext_sha != metadata_ciphertext:
        raise SystemExit("sealed ciphertext digest does not match metadata")
    if ciphertext_sha != manifest.get("sealed_ciphertext_sha256"):
        raise SystemExit("sealed ciphertext digest differs from the routing bundle")
    if development.get("sealed_test_status") != "LOCKED_UNOPENED":
        raise SystemExit("development manifest does not prove an unopened holdout")
    if development.get("schema_sha256") != feature_schema_digest(protocol):
        raise SystemExit("development manifest schema digest does not match V11.2")
    if development.get("feature_names") != list(protocol.feature_names):
        raise SystemExit("development manifest feature ordering does not match V11.2")
    if development.get("window_size") != protocol.window_size or development.get(
        "horizons"
    ) != list(protocol.horizons):
        raise SystemExit("development manifest target geometry does not match V11.2")
    if _sha256(train_path) != _require_sha256(development.get("train_sha256"), "train digest"):
        raise SystemExit("development train bytes do not match their manifest digest")
    if _sha256(validation_path) != _require_sha256(
        development.get("validation_sha256"), "validation digest"
    ):
        raise SystemExit("development validation bytes do not match their manifest digest")
    if manifest.get("sealed_test_status") != "LOCKED_UNOPENED":
        raise SystemExit("routing bundle does not prove an unopened holdout")
    protocol_payload = manifest.get("protocol")
    if not isinstance(protocol_payload, dict) or canonical_json_digest(protocol_payload) != (
        canonical_json_digest(protocol_manifest(protocol))
    ):
        raise SystemExit("routing bundle protocol mismatch")
    routes = manifest.get("routes", [])
    if not isinstance(routes, list) or any(not isinstance(route, dict) for route in routes):
        raise SystemExit("routing bundle routes must be a list of objects")
    route_horizons = [route.get("horizon") for route in routes]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in route_horizons):
        raise SystemExit("routing bundle route horizons must be integers")
    if sorted(route_horizons) != list(protocol.horizons):
        raise SystemExit("routing bundle does not contain exactly one route per horizon")
    if len({route.get("horizon") for route in routes}) != len(routes):
        raise SystemExit("routing bundle contains duplicate horizon routes")
    expected_bundle_sha = bundle_sha_path.read_text(encoding="ascii").strip()
    _require_sha256(expected_bundle_sha, "routing bundle digest")
    actual_bundle_sha = canonical_json_digest(
        {key: value for key, value in manifest.items() if key != "master_freeze_sha256"}
    )
    if expected_bundle_sha != actual_bundle_sha:
        raise SystemExit("routing bundle digest does not match its canonical manifest")
    if manifest.get("master_freeze_sha256") != expected_bundle_sha:
        raise SystemExit("routing bundle master digest does not match its digest file")

    for label in ("universe_sha256", "panel_sha256", "schema_sha256", "split_sha256"):
        _require_sha256(manifest.get(label), f"routing bundle {label}")
    if manifest.get("schema_sha256") != feature_schema_digest(protocol):
        raise SystemExit("routing bundle schema digest does not match the frozen V11.2 contract")
    if manifest.get("panel_sha256") != development.get("panel_sha256"):
        raise SystemExit("routing bundle panel digest differs from development evidence")
    if manifest.get("split_sha256") != development.get("split_sha256"):
        raise SystemExit("routing bundle split digest differs from development evidence")

    results_root = results.resolve()
    scaler_path = _required_file(results_root, "numeric_scaler.json", "numeric scaler")
    scaler_digest = _sha256(scaler_path)
    if any(route.get("scaler_digest") != scaler_digest for route in routes):
        raise SystemExit("routing bundle scaler digest does not match numeric_scaler.json")
    for route in routes:
        horizon = int(route["horizon"])
        family = str(route.get("family", ""))
        if family not in set(protocol.candidate_families):
            raise SystemExit(f"route {horizon} uses a family outside the protocol")
        baseline = family in {
            "ZERO_RETURN_CONST_VAR",
            "ZERO_RETURN_PERSISTENCE_VOL",
            "M0_HAR_BASELINE",
        }
        if bool(route.get("learned_promotion")) != (not baseline):
            raise SystemExit(f"route {horizon} has inconsistent learned_promotion")
        artifact_path = _required_file(
            results_root, route.get("artifact_path"), f"route {horizon} artifact"
        )
        if _sha256(artifact_path) != route.get("model_digest"):
            raise SystemExit(f"route {horizon} model digest does not match its artifact")
        selection_path = _required_file(
            results_root, f"selection_horizon_{horizon}.json", f"route {horizon} selection record"
        )
        if _sha256(selection_path) != route.get("selection_record_digest"):
            raise SystemExit(f"route {horizon} selection digest does not match its record")

    comparison_path = _required_file(
        results_root, "v11_2_development_model_comparison.json", "development comparison"
    )
    comparison_payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    if not isinstance(comparison_payload, dict):
        raise SystemExit("development comparison must contain a JSON object")
    comparison_digest = _require_sha256(
        comparison_payload.get("report_sha256"), "development comparison report_sha256"
    )
    comparison_body = dict(comparison_payload)
    comparison_body.pop("report_sha256", None)
    if canonical_json_digest(comparison_body) != comparison_digest:
        raise SystemExit("development comparison digest does not match its canonical contents")
    if comparison_digest != manifest.get("development_evidence_sha256"):
        raise SystemExit("routing bundle development evidence digest does not match report")

    seed_values = manifest.get("seed_evidence_sha256", [])
    if not isinstance(seed_values, list):
        raise SystemExit("routing bundle seed evidence must be a list")
    expected_seed_digests = {
        _require_sha256(value, "seed evidence digest") for value in seed_values
    }
    if len(expected_seed_digests) != len(protocol.horizons) * len(protocol.seeds):
        raise SystemExit(
            "routing bundle does not contain one unique evidence digest per seed/horizon"
        )
    for horizon in protocol.horizons:
        for seed in protocol.seeds:
            evidence_path = _required_file(
                results_root,
                Path("seed_evidence") / f"horizon_{horizon}" / f"seed_{seed}.json",
                f"h{horizon} seed {seed} evidence",
            )
            evidence_digest, evidence_body = _verify_self_digest(
                evidence_path, "evidence_sha256", f"h{horizon} seed {seed} evidence"
            )
            if evidence_digest not in expected_seed_digests:
                raise SystemExit(f"h{horizon} seed {seed} evidence is not referenced by the bundle")
            if evidence_body.get("horizon") != horizon or evidence_body.get("seed") != seed:
                raise SystemExit(f"h{horizon} seed {seed} evidence identity mismatch")

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
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audit = audit_pre_unseal(args.dataset_dir.resolve(), args.results_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
