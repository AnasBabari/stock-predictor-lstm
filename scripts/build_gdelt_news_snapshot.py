#!/usr/bin/env python3
"""Build a resumable immutable historical GDELT news-event snapshot."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "research"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from volatility_forecasting.gdelt import iter_gdelt_v1_daily_archives  # noqa: E402
from volatility_forecasting.gdelt_snapshot import (  # noqa: E402
    build_gdelt_daily_snapshot,
    load_ticker_aliases,
)


def _date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from error


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stream GDELT daily archives into a bounded point-in-time snapshot"
    )
    parser.add_argument("--start", type=_date, required=True, help="First archive date, inclusive")
    parser.add_argument("--end", type=_date, required=True, help="Last archive date, exclusive")
    parser.add_argument("--ticker-aliases", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--acknowledge-gdelt-terms",
        action="store_true",
        help="Required acknowledgement that the official provider terms were reviewed",
    )
    args = parser.parse_args()
    if not args.acknowledge_gdelt_terms:
        parser.error("--acknowledge-gdelt-terms is required")

    archives = iter_gdelt_v1_daily_archives(args.start, args.end)
    aliases = load_ticker_aliases(args.ticker_aliases)
    print(
        f"Preparing {len(archives):,} complete daily archives for {len(aliases)} tickers. "
        "Verified daily parts will be reused after interruption.",
        flush=True,
    )
    manifest = build_gdelt_daily_snapshot(
        archives,
        output_dir=args.output_dir.resolve(),
        work_dir=args.work_dir.resolve(),
        ticker_aliases=aliases,
        license_acknowledged=True,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
