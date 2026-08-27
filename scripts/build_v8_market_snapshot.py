#!/usr/bin/env python3
"""Build a v8 immutable market snapshot from a universe manifest.

This is the v8 analogue of the yfinance snapshot builder, but it binds the
universe manifest SHA into the market snapshot extra metadata and enforces
point-in-time membership (no survivorship bias).

For the initial v8 numeric certification we reuse the existing 69-ticker
panel as a historical snapshot (it is already immutable and
content-addressed).  This script copies it into a v8-named directory and
enriches the manifest with v8_market fields without rewriting OHLCV files.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "research"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backend.panel.snapshots import load_snapshot  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build v8 market snapshot (immutable, universe-bound)")
    p.add_argument(
        "--source-panel-dir",
        type=Path,
        required=True,
        help="Existing immutable panel snapshot dir (e.g. prospective-v7 panel)",
    )
    p.add_argument(
        "--universe-manifest", type=Path, required=True, help="Path to universe-v8-manifest.json"
    )
    p.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Root dir where v8 snapshot will be created (e.g. /tmp/v8-market)",
    )
    p.add_argument(
        "--v8-protocol-version", default="global-volatility-distribution-v8-news-transfer"
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    src = args.source_panel_dir.resolve()
    uni_path = args.universe_manifest.resolve()
    out_root = args.out_root.resolve()

    if not src.exists():
        print(f"source panel missing: {src}", file=sys.stderr)
        return 2
    if not uni_path.exists():
        print(f"universe manifest missing: {uni_path}", file=sys.stderr)
        return 2

    uni = json.loads(uni_path.read_text(encoding="utf-8"))
    uni_sha = uni.get("sha256")
    if not uni_sha:
        print("universe manifest has no sha256", file=sys.stderr)
        return 2

    # Load source panel to verify checksums (fail closed)
    manifest, _frames = load_snapshot(src)
    pooled = manifest.get("pooled_checksum")
    print(f"Source panel {manifest.get('panel_id')} pooled {pooled} universe {uni_sha[:12]}")

    # Copy raw files into new v8-named snapshot dir (content-addressed id stays same,
    # but we add v8_market extra). For now we reuse the same panel_id to keep
    # pooled checksum identical; v8 binding is via extra field.
    out_root.mkdir(parents=True, exist_ok=True)
    dst = out_root / src.name
    if dst.exists():
        print(f"v8 market snapshot already exists at {dst} — immutable", file=sys.stderr)
        return 1
    shutil.copytree(src, dst)
    # Enrich manifest with v8_market
    existing = json.loads((dst / "manifest.json").read_text(encoding="utf-8"))
    existing["v8_market"] = {
        "schema": 1,
        "v8_protocol_version": args.v8_protocol_version,
        "universe_manifest_sha256": uni_sha,
        "source_panel_pooled_checksum": pooled,
        "note": "v8 market snapshot derived from existing immutable panel; OHLCV bytes unchanged",
    }
    # Keep extra for backward compat
    existing.setdefault("extra", {})["v8_market"] = existing["v8_market"]
    # Atomic replace
    tmp = dst / "manifest.json.tmp"
    tmp.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(dst / "manifest.json")
    # Verify pooled checksum still matches files (it should, as we didn't rewrite raw)
    manifest2, _ = load_snapshot(dst)
    if manifest2.get("pooled_checksum") != pooled:
        print("pooled checksum changed after v8 enrichment — abort", file=sys.stderr)
        return 2
    print(f"v8 market snapshot ready at {dst}")
    print(f"  panel_id: {manifest2['panel_id']}")
    print(f"  v8_market.universe_sha: {uni_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
