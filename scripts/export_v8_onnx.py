#!/usr/bin/env python3
"""Export a real locked v8 candidate to ONNX and verify parity.

This command is intentionally fail-closed. It never creates structural or
constant-output placeholder graphs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export v8 members to ONNX + parity")
    ap.add_argument("--candidate-dir", type=Path, required=True, help="Prospective v8 candidate dir")
    ap.add_argument("--out", type=Path, required=True, help="Output dir for ONNX members")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    cand_dir = args.candidate_dir.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    manifest_path = cand_dir / "candidate-manifest.json"
    if not manifest_path.exists():
        print(f"candidate manifest missing: {manifest_path}")
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_role") != "locked_v8_certification_candidate":
        print("only a locked v8 certification candidate may be exported")
        return 2
    members = manifest.get("members")
    if not isinstance(members, list) or not members:
        print("locked candidate has no real members")
        return 2
    seeds = [member.get("seed") for member in members if isinstance(member, dict)]
    if len(seeds) != len(members) or any(not isinstance(seed, int) for seed in seeds):
        print("candidate member table is malformed")
        return 2
    print(f"Exporting {len(seeds)} members for {manifest.get('model_version')}")

    print(
        "v8 export requires the real locked-candidate loader; no placeholder "
        "graph will be emitted",
    )
    parity = {"status": "failed", "error": "real locked-v8 exporter not yet implemented"}

    (out / "onnx-parity.json").write_text(json.dumps(parity, indent=2) + "\n", encoding="utf-8")
    print(f"parity {parity['status']} written to {out / 'onnx-parity.json'}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
