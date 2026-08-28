#!/usr/bin/env python3
"""Bind an immutable historical event lake to the v8 market/universe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "research"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from volatility_forecasting.news_snapshot_v8 import (  # noqa: E402
    verify_v8_news_manifest,
    write_v8_news_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a v8 historical-news binding")
    parser.add_argument("--news-snapshot-dir", type=Path, required=True)
    parser.add_argument("--universe-manifest", type=Path, required=True)
    parser.add_argument("--market-manifest", type=Path, required=True)
    parser.add_argument("--ticker-aliases", type=Path, required=True)
    parser.add_argument("--provider-license-id", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-provider-gaps",
        action="store_true",
        help="Write an explicitly incomplete, non-certifiable research binding",
    )
    args = parser.parse_args()
    try:
        universe = json.loads(args.universe_manifest.read_text(encoding="utf-8"))
        market = json.loads(args.market_manifest.read_text(encoding="utf-8"))
        output = write_v8_news_manifest(
            args.out_dir.resolve(),
            news_snapshot_dir=args.news_snapshot_dir.resolve(),
            universe_manifest=universe,
            market_manifest=market,
            ticker_aliases_path=args.ticker_aliases.resolve(),
            provider_license_id=args.provider_license_id,
            allow_provider_gaps=args.allow_provider_gaps,
        )
        manifest = json.loads(output.read_text(encoding="utf-8"))
        verify_v8_news_manifest(
            manifest,
            news_snapshot_dir=args.news_snapshot_dir.resolve(),
            universe_manifest=universe,
            market_manifest=market,
            ticker_aliases_path=args.ticker_aliases.resolve(),
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"v8 news preparation failed: {error}", file=sys.stderr)
        return 2
    print(f"v8 news manifest: {output}")
    print(f"snapshot_id: {manifest['snapshot_id']}")
    print(f"events: {manifest['eligible_event_count']}/{manifest['article_count']} eligible")
    print(f"coverage_complete: {manifest['coverage_complete']}")
    print("news_status: snapshot_ready_uncertified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
