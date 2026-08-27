#!/usr/bin/env python3
"""Export a real locked v8 candidate to ONNX and verify parity.

This command is intentionally fail-closed. It never creates structural or
constant-output placeholder graphs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "research"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from research.volatility_forecasting.candidate_v8 import v8_ensemble_identity  # noqa: E402
from research.volatility_forecasting.export import (  # noqa: E402
    export_candidate_onnx,
    load_locked_v8_candidate_member,
    load_locked_v8_certification,
    verify_onnx_parity,
)
from research.volatility_forecasting.v8_protocol import v8_manifest  # noqa: E402


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_verified_certification(candidate_dir: Path, manifest: dict) -> tuple[dict, str]:
    return load_locked_v8_certification(candidate_dir, manifest)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export v8 members to ONNX + parity")
    ap.add_argument("--candidate-dir", type=Path, required=True, help="Prospective v8 candidate dir")
    ap.add_argument("--out", type=Path, required=True, help="Output dir for ONNX members")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--parity-rows", type=int, default=7)
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    cand_dir = args.candidate_dir.resolve()
    out = args.out.resolve()
    if out.exists():
        print(f"--out must not exist: {out}")
        return 2
    if args.opset < 17 or args.parity_rows < 2:
        print("opset must be >=17 and parity rows must be >=2")
        return 2

    manifest_path = cand_dir / "candidate-manifest.json"
    if not manifest_path.exists():
        print(f"candidate manifest missing: {manifest_path}")
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_role") != "locked_v8_certification_candidate":
        print("only a locked v8 certification candidate may be exported")
        return 2
    try:
        _certification, expected_certification_sha = _load_verified_certification(
            cand_dir, manifest
        )
    except ValueError as error:
        print(str(error))
        return 2
    members = manifest.get("members")
    if not isinstance(members, list) or not members:
        print("locked candidate has no real members")
        return 2
    seeds = [member.get("seed") for member in members if isinstance(member, dict)]
    if len(seeds) != len(members) or any(not isinstance(seed, int) for seed in seeds):
        print("candidate member table is malformed")
        return 2
    protocol_payload = manifest.get("protocol")
    news_enabled = bool(
        protocol_payload.get("news_enabled") if isinstance(protocol_payload, dict) else False
    )
    protocol = v8_manifest(news_enabled=news_enabled)
    expected_seeds = tuple(int(value) for value in protocol["seeds"])
    if tuple(sorted(seeds)) != expected_seeds:
        print(f"candidate seeds {tuple(sorted(seeds))} differ from protocol {expected_seeds}")
        return 2
    if manifest.get("protocol") != protocol:
        print("candidate protocol payload differs from the frozen v8 manifest")
        return 2
    try:
        candidates = tuple(
            load_locked_v8_candidate_member(cand_dir, seed) for seed in expected_seeds
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"locked candidate verification failed: {error}")
        return 2
    identity = v8_ensemble_identity(candidates)
    if identity != manifest.get("model_identity"):
        print("locked ensemble identity differs from its member content")
        return 2

    print(f"Exporting {len(candidates)} members for {manifest.get('model_version')}")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{out.name}-", dir=out.parent))
    try:
        rows: list[dict[str, object]] = []
        for candidate in candidates:
            filename = f"seed-{candidate.seed}.onnx"
            onnx_path = export_candidate_onnx(
                candidate,
                temporary / filename,
                opset_version=args.opset,
            )
            maximum_errors = verify_onnx_parity(
                candidate,
                onnx_path,
                rows=args.parity_rows,
            )
            rows.append(
                {
                    "seed": candidate.seed,
                    "model_identity": candidate.model_identity,
                    "onnx_file": filename,
                    "onnx_sha256": _sha256_file(onnx_path),
                    "maximum_absolute_errors": maximum_errors,
                }
            )
        parity = {
            "status": "passed",
            "artifact_role": "locked_v8_onnx_parity",
            "protocol_version": manifest.get("protocol_version"),
            "model_version": manifest.get("model_version"),
            "model_identity": identity,
            "certification_report_sha256": expected_certification_sha,
            "opset_version": args.opset,
            "parity_rows": args.parity_rows,
            "members": rows,
        }
        (temporary / "onnx-parity.json").write_text(
            json.dumps(parity, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(out)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(f"parity passed written to {out / 'onnx-parity.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
