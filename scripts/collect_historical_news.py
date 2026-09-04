"""Collect timestamped headlines and material filings for the five-ticker research universe."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from datetime import UTC, datetime
except ImportError:
    from datetime import datetime, timezone

    UTC = timezone.utc  # noqa: UP017

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.price_forecasting import DEFAULT_TICKERS  # noqa: E402
from research.price_forecasting.news_archive import (  # noqa: E402
    collect_alpaca_news,
    collect_public_recent_news,
    collect_sec_edgar_filings,
    collect_yahoo_news,
    merge_news_archive,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=datetime.now(UTC).date().isoformat())
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data" / "news" / "alpaca")
    parser.add_argument(
        "--provider",
        choices=["auto", "alpaca", "yahoo", "edgar", "all"],
        default="auto",
        help="Data provider: auto (alpaca if keyed, plus yahoo and edgar), alpaca, yahoo, edgar, all",
    )
    parser.add_argument(
        "--public-api-base",
        default="",
        help="Collect the recent public context feed instead of authenticated history.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    key_id = os.getenv("ALPACA_API_KEY_ID") or os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("ALPACA_API_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")

    if args.provider == "alpaca" and not (key_id and secret_key):
        raise RuntimeError(
            "Set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY locally; credentials are never written to the archive."
        )

    tickers = [value.strip().upper() for value in args.tickers.split(",") if value.strip()]
    summaries = []

    for ticker in tickers:
        records = []
        if args.public_api_base:
            print(f"collecting recent public headlines ticker={ticker}", flush=True)
            records.extend(collect_public_recent_news(ticker, api_base=args.public_api_base))

        if (args.provider in ("alpaca", "all") or args.provider == "auto") and (
            key_id and secret_key
        ):
            print(
                f"collecting alpaca news ticker={ticker} start={args.start} end={args.end}",
                flush=True,
            )
            records.extend(
                collect_alpaca_news(
                    ticker,
                    key_id=str(key_id),
                    secret_key=str(secret_key),
                    start=args.start,
                    end=args.end,
                )
            )

        if args.provider in ("yahoo", "all", "auto"):
            print(f"collecting yahoo news ticker={ticker}", flush=True)
            try:
                records.extend(collect_yahoo_news(ticker))
            except Exception as error:
                print(f"warning: yahoo collection error for {ticker}: {error}", flush=True)

        if args.provider in ("edgar", "all", "auto"):
            print(
                f"collecting SEC EDGAR material filings ticker={ticker} start={args.start} end={args.end}",
                flush=True,
            )
            try:
                records.extend(
                    collect_sec_edgar_filings(
                        ticker,
                        start=args.start,
                        end=args.end,
                    )
                )
            except Exception as error:
                print(f"warning: SEC EDGAR collection error for {ticker}: {error}", flush=True)

        manifest = merge_news_archive(args.output_dir / f"{ticker}.jsonl", records)
        summaries.append({"ticker": ticker, **manifest})
        print(
            f"ticker={ticker} articles={manifest['article_count']} sha256={manifest['sha256']}",
            flush=True,
        )

    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
