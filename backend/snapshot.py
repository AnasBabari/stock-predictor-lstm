"""Immutable, content-addressed market snapshot creation and validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf  # type: ignore[import-untyped]

SNAPSHOT_SCHEMA_VERSION = 1
REQUIRED_OHLCV = ("Open", "High", "Low", "Close", "Volume")
DEFAULT_BENCHMARK_UNIVERSE = (
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "JPM",
    "XOM",
    "JNJ",
    "WMT",
    "SPY",
    "IWM",
    "GLD",
    "BTC-USD",
)
DEFAULT_CONTEXT_TICKERS = ("QQQ", "^VIX", "^TNX")
TICKER_PATTERN = re.compile(r"(?:\^[A-Z0-9]{1,10}|[A-Z0-9.\-]{1,12})")


def normalise_ticker(value: str) -> str:
    ticker = str(value).strip().upper()
    if not TICKER_PATTERN.fullmatch(ticker):
        raise ValueError("Snapshot ticker contains an unsupported identity.")
    return ticker


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def validate_market_frame(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Validate a provider response without silently repairing invalid observations."""

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"No market data is available for {ticker}.")
    result = frame.copy()
    if isinstance(result.columns, pd.MultiIndex):
        result.columns = result.columns.get_level_values(0)
    if not set(REQUIRED_OHLCV).issubset(result.columns):
        raise ValueError(f"Market data for {ticker} is missing required OHLCV columns.")
    if result.index.has_duplicates:
        raise ValueError(f"Market data for {ticker} contains duplicate sessions.")
    result.index = pd.DatetimeIndex(result.index)
    if not result.index.is_monotonic_increasing:
        raise ValueError(f"Market data for {ticker} is not ordered by session.")
    ohlcv = result.loc[:, list(REQUIRED_OHLCV)].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(ohlcv.to_numpy(dtype=float)).all():
        raise ValueError(f"Market data for {ticker} contains non-finite OHLCV values.")
    if (ohlcv[["Open", "High", "Low", "Close"]] <= 0).any().any() or (ohlcv["Volume"] < 0).any():
        raise ValueError(f"Market data for {ticker} contains invalid OHLCV values.")
    return result.loc[:, list(REQUIRED_OHLCV)].copy()


def create_market_snapshot(
    tickers: tuple[str, ...] | list[str],
    *,
    start: str,
    end: str,
    output: Path,
    downloader: Callable[..., pd.DataFrame] = yf.download,
    benchmark_universe: tuple[str, ...] | list[str] | None = None,
) -> dict:
    """Download, validate and write immutable Parquet market data plus its manifest."""

    selected = tuple(dict.fromkeys(normalise_ticker(ticker) for ticker in tickers))
    targets = tuple(
        dict.fromkeys(
            normalise_ticker(ticker)
            for ticker in (benchmark_universe if benchmark_universe is not None else selected)
        )
    )
    if not selected:
        raise ValueError("At least one ticker is required.")
    if pd.Timestamp(start) >= pd.Timestamp(end):
        raise ValueError("Snapshot start must precede end.")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Snapshot output already exists and is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    market_dir = output / "market"
    market_dir.mkdir()
    assets: list[dict] = []
    for ticker in selected:
        raw = downloader(ticker, start=start, end=end, progress=False, auto_adjust=True, timeout=30)
        frame = validate_market_frame(raw, ticker)
        relative_path = Path("market") / f"{ticker}.parquet"
        destination = output / relative_path
        frame.to_parquet(destination, index=True)
        assets.append(
            {
                "ticker": ticker,
                "path": relative_path.as_posix(),
                "rows": int(len(frame)),
                "start": str(frame.index[0].date()),
                "end": str(frame.index[-1].date()),
                "sha256": _sha256_file(destination),
            }
        )
    content_hash = hashlib.sha256(_canonical_json({"assets": assets})).hexdigest()
    manifest = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "requested": {"tickers": list(selected), "start": start, "end": end},
        "benchmark_universe": list(targets),
        "provider": {
            "name": "yfinance",
            "version": getattr(yf, "__version__", "unknown"),
            "auto_adjust": True,
        },
        "environment": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "assets": assets,
        "content_sha256": content_hash,
    }
    import os
    import uuid

    manifest["manifest_sha256"] = hashlib.sha256(_canonical_json(manifest)).hexdigest()
    temp_path = output / f".manifest-{uuid.uuid4().hex}.json"
    temp_path.write_bytes(_canonical_json(manifest) + b"\n")
    os.replace(temp_path, output / "manifest.json")
    return manifest


def load_market_snapshot(manifest_path: Path) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Verify a market snapshot before exposing its frames to a benchmark."""

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    claimed_manifest_hash = manifest.pop("manifest_sha256", None)
    if claimed_manifest_hash != hashlib.sha256(_canonical_json(manifest)).hexdigest():
        raise ValueError("Market snapshot manifest hash does not match.")
    asset_digest = hashlib.sha256(_canonical_json({"assets": manifest["assets"]})).hexdigest()
    if asset_digest != manifest.get("content_sha256"):
        raise ValueError("Market snapshot content hash does not match.")
    frames: dict[str, pd.DataFrame] = {}
    for asset in manifest["assets"]:
        ticker = normalise_ticker(asset["ticker"])
        expected_path = Path("market") / f"{ticker}.parquet"
        if Path(asset["path"]) != expected_path:
            raise ValueError(f"Market snapshot asset path is invalid: {ticker}")
        path = manifest_path.parent / expected_path
        if not path.is_file() or _sha256_file(path) != asset["sha256"]:
            raise ValueError(f"Market snapshot asset hash does not match: {ticker}")
        frames[ticker] = validate_market_frame(pd.read_parquet(path), ticker)
    manifest["manifest_sha256"] = claimed_manifest_hash
    return manifest, frames


def _read_universe(path: Path) -> tuple[str, ...]:
    data = json.loads(path.read_text(encoding="utf-8"))
    tickers = data.get("tickers", data) if isinstance(data, dict) else data
    if not isinstance(tickers, list):
        raise ValueError("Universe JSON must be an array or an object with a 'tickers' array.")
    return tuple(str(ticker) for ticker in tickers)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("market", nargs="?")
    parser.add_argument("--universe", type=Path)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    universe = _read_universe(args.universe) if args.universe else DEFAULT_BENCHMARK_UNIVERSE
    tickers = tuple(dict.fromkeys((*universe, *DEFAULT_CONTEXT_TICKERS)))
    manifest = create_market_snapshot(
        tickers,
        start=args.start,
        end=args.end,
        output=args.output,
        benchmark_universe=universe,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
