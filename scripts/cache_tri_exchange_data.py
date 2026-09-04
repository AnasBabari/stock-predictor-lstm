"""Download, validate, and cache OHLCV parquets for the 30-ticker flagship tri-exchange universe.

Universe:
- NASDAQ (10): AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA, AMD, COST, QCOM
- NYSE (10):   JPM, XOM, WMT, JNJ, CAT, KO, NEE, DIS, BAC, GE
- LSE (10):    SHEL.L, AZN.L, HSBA.L, BP.L, ULVR.L, GSK.L, RIO.L, BATS.L, BARC.L, DGE.L
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from backend.data_pipeline import _download_ohlcv  # noqa: E402
from research.price_forecasting.gpu_pipeline import (  # noqa: E402
    TRI_EXCHANGE_TICKERS,
    _normalise_ohlcv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "tri_exchange" / "cache",
        help="Directory to save validated parquet files",
    )
    parser.add_argument(
        "--ndx-cache-dir",
        type=Path,
        default=REPO_ROOT / "data" / "ndx100" / "cache",
        help="Existing cache directory to check first for NDX constituents",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Caching tri-exchange universe ({len(TRI_EXCHANGE_TICKERS)} tickers) into {args.output_dir}..."
    )

    succeeded = 0
    failed: list[tuple[str, str]] = []

    for index, ticker in enumerate(TRI_EXCHANGE_TICKERS, start=1):
        target_parquet = args.output_dir / f"{ticker}.parquet"
        t0 = time.perf_counter()
        try:
            # Check existing NDX cache first if available
            existing_ndx = args.ndx_cache_dir / f"{ticker}.parquet"
            if existing_ndx.is_file():
                df = pd.read_parquet(existing_ndx)
            else:
                df = _download_ohlcv(ticker)

            norm = _normalise_ohlcv(df)
            norm.to_parquet(target_parquet)
            elapsed = time.perf_counter() - t0
            succeeded += 1
            print(
                f"[{index:02d}/{len(TRI_EXCHANGE_TICKERS):02d}] ✓ {ticker:<8} "
                f"bars={len(norm)} ({norm.index[0].date()} -> {norm.index[-1].date()}) in {elapsed:.2f}s"
            )
        except Exception as exc:
            failed.append((ticker, str(exc)))
            print(f"[{index:02d}/{len(TRI_EXCHANGE_TICKERS):02d}] ✗ {ticker:<8} FAILED: {exc}")

    print(f"\nCompleted: {succeeded}/{len(TRI_EXCHANGE_TICKERS)} tickers successfully cached.")
    if failed:
        print(f"Failed tickers: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
