"""Download, validate, and cache OHLCV parquets for the 300-ticker flagship tri-exchange universe.

Universe:
- NASDAQ-100 (~100): AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA, AMD, COST, QCOM, ...
- NYSE Leaders (~100): JPM, XOM, WMT, JNJ, CAT, KO, NEE, DIS, BAC, GE, ...
- FTSE 100 (~100): SHEL.L, AZN.L, HSBA.L, BP.L, ULVR.L, GSK.L, RIO.L, ...
"""

from __future__ import annotations

import argparse
import json
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

NASDAQ_100 = (
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AMD", "COST", "QCOM",
    "AVGO", "NFLX", "ADBE", "TXN", "INTC", "CMCSA", "PEP", "CSCO", "INTU", "AMAT",
    "PYPL", "BKNG", "ISRG", "MDLZ", "GILD", "REGN", "ADI", "VRTX", "LRCX", "PANW",
    "SNPS", "KLAC", "CDNS", "CHTR", "MAR", "ORLY", "NXPI", "FTNT", "CTAS", "PCAR",
    "KDP", "PAYX", "MNST", "MCHP", "ROST", "AEP", "KHC", "ODFL", "FAST", "IDXX",
    "EXC", "LULU", "VRSK", "CSX", "GEHC", "BIIB", "DXCM", "XEL", "EA", "FANG",
    "TEAM", "MRVL", "ON", "ANSS", "BKR", "WBD", "DLTR", "ILMN", "ALGN", "CEG",
    "CPRT", "SBUX", "HON", "ASML", "AZN", "PDD", "MELI", "CRWD", "DDOG", "ZS",
    "WDAY", "ABNB", "MRNA", "DASH", "TTD", "MDB", "SMCI", "ARM", "APP", "PLTR",
    "ROP", "TTWO", "CDW", "GFS", "ANET", "CCEP", "AXON", "MSTR", "LIN", "CTSH",
)

NYSE_100 = (
    "JPM", "XOM", "WMT", "JNJ", "CAT", "KO", "NEE", "DIS", "BAC", "GE",
    "UNH", "PG", "V", "HD", "CVX", "MA", "PFE", "MRK", "ABT", "ORCL",
    "LLY", "MCD", "TMO", "NKE", "IBM", "GS", "MS", "RTX", "UNP", "BMY",
    "COP", "PM", "LOW", "UPS", "MSI", "DE", "SCHW", "C", "AXP", "BLK",
    "PGR", "CB", "MMC", "AON", "MET", "PRU", "AIG", "TRV", "ALL", "EOG",
    "SLB", "MPC", "PSX", "VLO", "OXY", "DVN", "HES", "KMI", "WMB", "ET",
    "MDT", "SYK", "BSX", "EW", "BAX", "BDX", "ZTS", "CI", "ELV", "HUM",
    "CVS", "FDX", "NSC", "WM", "RSG", "EMR", "ETN", "ITW", "PH", "CMI",
    "TT", "JCI", "CARR", "OTIS", "GD", "NOC", "LHX", "LMT", "BA", "HMC",
    "TM", "SONY", "NVO", "BABA", "BBL", "NVS", "SAP", "TSM", "DEO", "BAM",
)

LSE_100 = (
    "SHEL.L", "AZN.L", "HSBA.L", "BP.L", "ULVR.L", "GSK.L", "RIO.L", "BATS.L", "BARC.L", "DGE.L",
    "REL.L", "LSEG.L", "LLOY.L", "NG.L", "EXPN.L", "VOD.L", "GLEN.L", "AAL.L", "PRU.L", "NWG.L",
    "STAN.L", "BA.L", "CPG.L", "IMB.L", "AV.L", "SSE.L", "RR.L", "TSCO.L", "ABF.L", "ADM.L",
    "ANTO.L", "INF.L", "MNDI.L", "WPP.L", "HLN.L", "SDR.L", "LAND.L", "BLND.L", "UU.L", "SVT.L",
    "WTB.L", "AUTO.L", "ENT.L", "JD.L", "KGF.L", "MKS.L", "NXT.L", "BME.L", "OCDO.L", "SMIN.L",
    "SMT.L", "FDM.L", "BDEV.L", "PSN.L", "TW.L", "CRDA.L", "JMAT.L", "HLMA.L", "SPX.L", "WEIR.L",
    "IMI.L", "SN.L", "SGE.L", "RS1.L", "BEZ.L", "HIK.L", "IHG.L", "PHNX.L", "LGEN.L", "STJ.L",
    "UTG.L", "ITRK.L", "BKG.L", "SGRO.L", "ICP.L", "PSON.L", "DCC.L", "BVI.L", "SBRY.L", "FRAS.L",
    "RMV.L", "DARK.L", "WISE.L", "EMG.L", "MONY.L", "SMDS.L", "HL.L", "EDV.L", "GAW.L", "IGG.L",
    "VTY.L", "WIZZ.L", "GNS.L", "BOY.L", "ITV.L", "GFTU.L", "MAN.L", "TCG.L", "TRN.L", "EZJ.L",
)

UNIVERSES: dict[str, tuple[str, ...]] = {
    "tri_exchange": TRI_EXCHANGE_TICKERS,
    "broad_300": tuple(dict.fromkeys(NASDAQ_100 + NYSE_100 + LSE_100)),
    "all": tuple(dict.fromkeys(NASDAQ_100 + NYSE_100 + LSE_100)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--universe",
        choices=["tri_exchange", "broad_300", "all"],
        default="broad_300",
        help="Target universe to cache",
    )
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
    parser.add_argument(
        "--min-rows",
        type=int,
        default=500,
        help="Minimum valid completed bars required per ticker",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tickers = UNIVERSES[args.universe]
    print(f"Caching universe '{args.universe}' ({len(tickers)} symbols) into {args.output_dir}...")

    succeeded: list[str] = []
    failed: list[tuple[str, str]] = []
    start = time.perf_counter()

    for index, ticker in enumerate(tickers, start=1):
        target_parquet = args.output_dir / f"{ticker}.parquet"
        t0 = time.perf_counter()
        try:
            # 1. If target already cached and valid, keep
            if target_parquet.is_file():
                df = pd.read_parquet(target_parquet)
                if len(df) >= args.min_rows:
                    succeeded.append(ticker)
                    print(
                        f"[{index:03d}/{len(tickers):03d}] ✓ {ticker:<8} cached ({len(df)} bars)",
                        flush=True,
                    )
                    continue

            # 2. If available in existing NDX cache, load and normalize
            existing_ndx = args.ndx_cache_dir / f"{ticker}.parquet"
            if existing_ndx.is_file():
                raw = pd.read_parquet(existing_ndx)
                norm = _normalise_ohlcv(raw)
            else:
                raw = _download_ohlcv(ticker)
                norm = _normalise_ohlcv(raw)

            if len(norm) < args.min_rows:
                failed.append((ticker, f"Insufficient rows ({len(norm)} < {args.min_rows})"))
                print(
                    f"[{index:03d}/{len(tickers):03d}] ✗ {ticker:<8} FAILED (< {args.min_rows} bars)",
                    flush=True,
                )
                continue

            norm.to_parquet(target_parquet)
            elapsed = time.perf_counter() - t0
            succeeded.append(ticker)
            print(
                f"[{index:03d}/{len(tickers):03d}] ✓ {ticker:<8} saved ({len(norm)} bars, "
                f"{norm.index[0].date()} -> {norm.index[-1].date()}) in {elapsed:.2f}s",
                flush=True,
            )
        except Exception as exc:
            failed.append((ticker, str(exc)))
            print(f"[{index:03d}/{len(tickers):03d}] ✗ {ticker:<8} ERROR: {exc}", flush=True)

    elapsed_total = time.perf_counter() - start
    manifest = {
        "universe": args.universe,
        "requested_count": len(tickers),
        "success_count": len(succeeded),
        "failed_count": len(failed),
        "valid_tickers": succeeded,
        "failed_tickers": failed,
        "elapsed_seconds": elapsed_total,
        "output_dir": str(args.output_dir),
    }
    manifest_path = args.output_dir.parent / f"manifest_{args.universe}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n--- Summary ---")
    print(
        f"Success: {len(succeeded)}/{len(tickers)} ({len(succeeded)/len(tickers)*100:.1f}%) "
        f"in {elapsed_total:.1f}s"
    )
    if failed:
        print(f"Failed symbols: {[sym for sym, _ in failed]}")
    print(f"Manifest saved to: {manifest_path}")
    return 0 if len(succeeded) >= 200 else 1


if __name__ == "__main__":
    raise SystemExit(main())
