#!/usr/bin/env python3
"""Build and validate content-addressed global panel snapshots and features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from scripts.run_global_pipeline import (  # noqa: E402
    PipelineConfig,
    generate_synthetic_universe,
    run_pipeline,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and validate global panel snapshots")
    parser.add_argument(
        "--run-dir", type=Path, default=Path("runs/panel_snapshot"), help="Output directory"
    )
    parser.add_argument("--n-tickers", type=int, default=10, help="Number of universe tickers")
    parser.add_argument("--n-sessions", type=int, default=450, help="Number of master sessions")
    args = parser.parse_args()

    tickers = [f"TICK{i:02d}" for i in range(args.n_tickers)]
    universe = generate_synthetic_universe(tickers, n_sessions=args.n_sessions)
    cfg = PipelineConfig(run_id="panel_build_run")
    print(
        f"Building panel snapshot with {len(tickers)} tickers across {args.n_sessions} sessions..."
    )
    results = run_pipeline(config=cfg, run_dir=args.run_dir, universe_data=universe)
    print(f"Panel snapshot build completed. Panel ID: {results['stages']['snapshot']['panel_id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
