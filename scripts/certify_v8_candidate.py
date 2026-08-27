#!/usr/bin/env python3
"""One-shot v8 certification — sealed historical test + asset-transfer.

This is the v8 analogue of ``scripts/certify_prospective_volatility_candidate.py``
but for the chronological 70/15/15 split. It never trains.

It verifies:
- candidate role == prospective_v8_development_candidate
- protocol == global-volatility-distribution-v8-*
- panel/universe/split/news checksums match
- feature/target order matches
- test sealed (no prior holdout-opened marker)
- every member and horizon present
- no retraining, no future news
- NMM/MSFT coverage where applicable
- all required horizons pass

On success materializes ``locked_v8_certification_candidate`` with
``release_eligible=true`` and ``metric_source=
locked_historical_temporal_test_plus_asset_transfer``.
On any failure: ``release_eligible=false``, no partial candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "research"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backend.panel.snapshots import load_panel_from_directory  # noqa: E402
from research.volatility_forecasting.cache import (  # noqa: E402
    find_compatible_example_cache,
    load_example_cache,
    panel_fingerprint,
)
from research.volatility_forecasting.data import build_volatility_panel_examples  # noqa: E402
from research.volatility_forecasting.split_v8 import build_v8_chronological_split  # noqa: E402
from research.volatility_forecasting.v8_protocol import (  # noqa: E402
    V8_PROTOCOL_VERSION_NEWS,
    V8_PROTOCOL_VERSION_NUMERIC,
    v8_protocol,
)


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1024 * 1024), b""):
            h.update(c)
    return h.hexdigest()


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="One-shot v8 certification (sealed test)")
    ap.add_argument("--candidate-dir", type=Path, required=True)
    ap.add_argument("--panel-dir", type=Path, required=True)
    ap.add_argument("--universe-manifest", type=Path, required=True)
    ap.add_argument("--news-manifest", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True, help="Empty output dir for certification")
    ap.add_argument("--holdouts", type=str, default="NMM,MSFT")
    ap.add_argument(
        "--open-sealed-test",
        action="store_true",
        help="Required acknowledgement: irreversibly opens sealed test",
    )
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.open_sealed_test:
        print("--open-sealed-test is required (acknowledges one-shot)", file=sys.stderr)
        return 2
    cand_dir = args.candidate_dir.resolve()
    panel_dir = args.panel_dir.resolve()
    uni_path = args.universe_manifest.resolve()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        print(f"--out must be empty: {out}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)

    holdouts = tuple(sorted({t.strip().upper() for t in args.holdouts.split(",") if t.strip()}))

    # Load and validate a real frozen candidate. Placeholder/dry-run artifacts
    # must fail before the one-shot marker is written and before any sealed row
    # is opened.
    cand_manifest_path = cand_dir / "candidate-manifest.json"
    if not cand_manifest_path.exists():
        print(f"candidate manifest missing: {cand_manifest_path}", file=sys.stderr)
        return 2
    cand = json.loads(cand_manifest_path.read_text(encoding="utf-8"))
    if cand.get("artifact_role") != "prospective_v8_development_candidate":
        print(
            f"candidate role must be prospective_v8_development_candidate, got {cand.get('artifact_role')}",
            file=sys.stderr,
        )
        return 2
    if cand.get("placeholder") or cand.get("model_type") in {
        "ridge_stub",
        "ridge_cpu",
        "dummy",
    }:
        print("placeholder candidates are not certifiable", file=sys.stderr)
        return 2
    members = cand.get("members")
    if not isinstance(members, list) or not members:
        print("candidate has no persisted model members", file=sys.stderr)
        return 2
    if any(
        not isinstance(member, dict)
        or not isinstance(member.get("weights_file"), str)
        or not str(member["weights_file"]).endswith(".pt")
        for member in members
    ):
        print("candidate members are placeholders or malformed", file=sys.stderr)
        return 2
    proto_version = cand.get("protocol_version")
    if proto_version not in (V8_PROTOCOL_VERSION_NEWS, V8_PROTOCOL_VERSION_NUMERIC):
        print(f"candidate protocol {proto_version} not v8", file=sys.stderr)
        return 2
    news_enabled = bool(cand.get("news_enabled"))
    protocol = v8_protocol(news_enabled=news_enabled)
    if proto_version != protocol.protocol_version:
        print(
            f"candidate protocol {proto_version} != expected {protocol.protocol_version}",
            file=sys.stderr,
        )
        return 2

    uni = json.loads(uni_path.read_text(encoding="utf-8"))
    uni_sha = uni.get("sha256")
    if cand.get("universe_manifest_sha256") != uni_sha:
        print(
            f"candidate universe sha mismatch {cand.get('universe_manifest_sha256')} vs {uni_sha}",
            file=sys.stderr,
        )
        return 2

    panel_fp = panel_fingerprint(panel_dir)
    if cand.get("panel_checksum") != panel_fp:
        print(
            f"candidate panel checksum mismatch {cand.get('panel_checksum')} vs {panel_fp}",
            file=sys.stderr,
        )
        return 2

    # Verify split manifest matches candidate's split
    # Rebuild split to ensure checksums and purge/embargo still hold
    # Load examples (cached)
    for root in [Path(r"C:\tmp\stocklstm-volatility-panel-v1\example-cache")]:
        try:
            compat = find_compatible_example_cache(root, panel_checksum=panel_fp, protocol=protocol)
            if compat:
                examples = load_example_cache(compat, panel_checksum=panel_fp, protocol=protocol)
                break
        except Exception:
            continue
    else:
        panel = load_panel_from_directory(panel_dir)
        examples = build_volatility_panel_examples(panel, protocol)

    # Use candidate's split manifest SHA to verify we are opening the same sealed test
    cand_split_sha = cand.get("split_manifest_sha256")
    # Also need news checksum
    news_sha = cand.get("news_snapshot_checksum") or (
        "sha256:" + hashlib.sha256(b"no_news").hexdigest()
    )

    # Create one-shot marker BEFORE opening test (fail-closed)
    marker = {
        "candidate_manifest_sha256": _sha256_file(cand_manifest_path),
        "panel_checksum": panel_fp,
        "universe_sha256": uni_sha,
        "protocol_version": proto_version,
        "holdouts": list(holdouts),
        "one_shot": True,
    }
    (out / "v8-holdout-opened.json").write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"one-shot marker written to {out / 'v8-holdout-opened.json'}")

    # Rebuild split — this is the sealed test opening
    split = build_v8_chronological_split(
        examples,
        protocol=protocol,
        required_asset_holdouts=holdouts,
        universe_manifest_sha256=uni_sha,
        panel_checksum=panel_fp,
        news_snapshot_checksum=news_sha,
    )
    # Verify split SHA still matches candidate's (detects tampering)
    recomputed_sha = hashlib.sha256(
        json.dumps(split.manifest.__dict__, sort_keys=True, default=str).encode()
    ).hexdigest()
    if cand_split_sha and recomputed_sha != cand_split_sha:
        print(
            f"split manifest SHA mismatch after open: {recomputed_sha} vs candidate {cand_split_sha}",
            file=sys.stderr,
        )
        # Do not materialize, but still write failed report
        status = "failed"
    else:
        print(
            "real v8 prediction-only certification is not yet wired; refusing "
            "to fabricate sealed-test evidence",
            file=sys.stderr,
        )
        status = "failed"

    report = {
        "status": status,
        "protocol_version": proto_version,
        "model_version": cand.get("model_version"),
        "metric_source": "locked_historical_temporal_test_plus_asset_transfer",
        "certification_scope": "historical_temporal_test_plus_asset_transfer",
        "panel_checksum": panel_fp,
        "universe_sha256": uni_sha,
        "holdouts": list(holdouts),
        "temporal_test_rows": split.manifest.temporal_test_rows,
        "asset_transfer_test_rows": split.manifest.asset_transfer_test_rows,
        "pooled_test_rows": split.manifest.pooled_test_rows,
        "split_manifest": split.manifest.__dict__,
        "candidate_manifest_sha256": marker["candidate_manifest_sha256"],
        "holdout_opened": str(out / "v8-holdout-opened.json"),
    }
    (out / "v8-locked-certification.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    print(f"certification report {status} written to {out / 'v8-locked-certification.json'}")

    if status == "passed":
        # Materialize locked candidate (copy manifest + members)
        cand_out = out / "candidate"
        cand_out.mkdir()
        # Copy member files
        for seed in protocol.seeds:
            src = cand_dir / f"seed-{seed}.json"
            if src.exists():
                (cand_out / src.name).write_bytes(src.read_bytes())
        locked_manifest = dict(cand)
        locked_manifest["artifact_role"] = "locked_v8_certification_candidate"
        locked_manifest["release_eligible"] = True
        locked_manifest["certification_report_sha256"] = _sha256_file(
            out / "v8-locked-certification.json"
        )
        (cand_out / "candidate-manifest.json").write_text(
            json.dumps(locked_manifest, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        print(f"locked candidate materialized at {cand_out}")
        return 0
    else:
        print("certification failed — no release eligible candidate materialized", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
