#!/usr/bin/env python3
"""Build and validate content-addressed global panel snapshots from real or fixture data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from panel.snapshots import (  # noqa: E402
    build_snapshot,
    fetch_panel_universe,
    write_snapshot,
)
from scripts.run_global_pipeline import generate_synthetic_universe  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and validate global panel snapshots")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data/panel_snapshot"), help="Output directory"
    )
    parser.add_argument(
        "--universe-file", type=Path, default=None, help="Text file containing universe tickers (one per line)"
    )
    parser.add_argument("--tickers", type=str, default=None, help="Comma-separated ticker list")
    parser.add_argument("--years", type=int, default=8, help="Years of daily history to fetch")
    parser.add_argument(
        "--license-acknowledged",
        action="store_true",
        help="Explicitly acknowledge provider terms for panel download and training",
    )
    parser.add_argument(
        "--synthetic", action="store_true", help="Generate synthetic fixture panel for testing"
    )
    parser.add_argument("--n-tickers", type=int, default=10, help="Number of synthetic tickers")
    parser.add_argument("--n-sessions", type=int, default=500, help="Number of synthetic sessions")
    args = parser.parse_args()

    if args.synthetic:
        tickers = [f"TICK{i:02d}" for i in range(args.n_tickers)]
        print(f"Generating synthetic fixture panel ({args.n_tickers} tickers, {args.n_sessions} sessions)...")
        universe = generate_synthetic_universe(tickers, n_sessions=args.n_sessions)
        out_path = write_snapshot(args.out_dir, universe, license_acknowledged=True)
        manifest = build_snapshot(universe, license_acknowledged=True)
        print(f"Synthetic panel snapshot created successfully at {out_path} (ID: {manifest['panel_id']})")
        return 0

    ticker_list: list[str] = []
    if args.universe_file and args.universe_file.exists():
        lines = args.universe_file.read_text(encoding="utf-8").splitlines()
        ticker_list = [line.strip().upper() for line in lines if line.strip() and not line.startswith("#")]
    elif args.tickers:
        ticker_list = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    if not ticker_list:
        print("Error: Specify --universe-file, --tickers, or --synthetic", file=sys.stderr)
        return 1

    if not args.license_acknowledged:
        print(
            "Error: --license-acknowledged is required before downloading real market data. "
            "Review provider terms for local caching and model training.",
            file=sys.stderr,
        )
        return 1

    print(f"Fetching OHLCV for {len(ticker_list)} tickers ({args.years} years)...")
    universe = fetch_panel_universe(ticker_list, years=args.years)
    out_path = write_snapshot(args.out_dir, universe, license_acknowledged=True)
    manifest = build_snapshot(universe, license_acknowledged=True)
    print(f"Immutable panel snapshot created at {out_path} (ID: {manifest['panel_id']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
