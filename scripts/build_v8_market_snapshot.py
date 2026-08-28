#!/usr/bin/env python3
"""Acquire or convert an immutable universe-bound v8 market snapshot."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "research"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from volatility_forecasting.market_snapshot_v8 import (  # noqa: E402
    fetch_v8_market_frames,
    verify_v8_market_snapshot,
    write_v8_market_snapshot,
)
from volatility_forecasting.universe_v8 import verify_universe_manifest  # noqa: E402
from volatility_forecasting.v8_protocol import V8_PROTOCOL_VERSION_NUMERIC  # noqa: E402

from backend.panel.snapshots import load_snapshot  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an immutable v8 market snapshot")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--download-from-universe",
        action="store_true",
        help="Download raw and adjusted histories for the complete universe",
    )
    mode.add_argument(
        "--source-panel-dir",
        type=Path,
        help="Convert a legacy adjusted-only panel into a diagnostic v8 snapshot",
    )
    parser.add_argument("--universe-manifest", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--years", type=int, default=10)
    parser.add_argument("--provider", default="yfinance")
    parser.add_argument("--provider-snapshot-id", required=True)
    parser.add_argument("--provider-license-id", required=True)
    parser.add_argument("--license-acknowledged", action="store_true")
    parser.add_argument("--v8-protocol-version", default=V8_PROTOCOL_VERSION_NUMERIC)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.license_acknowledged:
        print("--license-acknowledged is required after reviewing provider terms", file=sys.stderr)
        return 2
    try:
        universe = verify_universe_manifest(
            json.loads(args.universe_manifest.resolve().read_text(encoding="utf-8"))
        )
        if args.download_from_universe:
            os.environ["PANEL_LICENSE_ACKNOWLEDGED"] = "true"
            frames = fetch_v8_market_frames(universe, years=args.years)
            allow_diagnostic = False
            derived_checksum = None
        else:
            legacy_manifest, frames = load_snapshot(args.source_panel_dir.resolve())
            allow_diagnostic = True
            derived_checksum = str(legacy_manifest.get("pooled_checksum"))
        output = write_v8_market_snapshot(
            args.out_root.resolve(),
            frames,
            universe_manifest=universe,
            provider=args.provider,
            provider_snapshot_id=args.provider_snapshot_id,
            provider_license_id=args.provider_license_id,
            license_acknowledged=True,
            v8_protocol_version=args.v8_protocol_version,
            allow_incomplete_diagnostic=allow_diagnostic,
            derived_from_panel_checksum=derived_checksum,
        )
        manifest, _ = verify_v8_market_snapshot(
            output,
            universe_manifest=universe,
            require_certifiable=args.download_from_universe,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"v8 market build failed: {error}", file=sys.stderr)
        return 2

    status = manifest["v8_market"]
    print(f"v8 market snapshot: {output}")
    print(f"pooled_checksum: {manifest['pooled_checksum']}")
    print(f"acquired: {status['acquired_ticker_count']}/{status['requested_ticker_count']}")
    print(f"coverage_certifiable: {status['coverage_certifiable']}")
    if status["coverage_reasons"]:
        print("coverage_reasons: " + ", ".join(status["coverage_reasons"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
