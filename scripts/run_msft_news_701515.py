"""Fixed GPU models, MSFT chronological 70/15/15 exploratory evaluation."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from run_historical_news_volatility import (
    ROOT,
    _normalise_ohlcv,
    build_causal_news_features,
    build_price_features,
    canonicalize,
    fit_and_score,
    write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ticker", default="MSFT")
    args = parser.parse_args()
    ticker = args.ticker.upper()
    if not ticker.isalpha():
        raise ValueError("Expected a US alphabetic ticker")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    boundaries = list(pd.date_range("2019-01-01", "2026-09-01", freq="QS")) + [
        pd.Timestamp("2026-09-05")
    ]
    hashes = {}
    for start, end in zip(boundaries, boundaries[1:], strict=False):
        chunk = ROOT / f"data/news/historical_v1/{ticker}_{start.date()}.jsonl"
        manifest = json.loads(chunk.with_suffix(".manifest.json").read_text())
        digest = hashlib.sha256(chunk.read_bytes()).hexdigest()
        if (
            digest != manifest["sha256"]
            or manifest.get("pagination_complete") is not True
            or manifest.get("requested_start") != str(start)
            or manifest.get("requested_end_exclusive") != str(end)
        ):
            raise ValueError(f"Invalid news chunk: {chunk.name}")
        hashes[chunk.name] = digest
        records.extend(json.loads(line) for line in chunk.read_text().splitlines() if line)
    records, diagnostics = canonicalize(records)
    price_path = ROOT / f"data/tri_exchange/cache/{ticker}.parquet"
    prices = _normalise_ohlcv(pd.read_parquet(price_path)).loc[:"2026-09-04"]
    features = build_price_features(prices)
    features = features.join(build_causal_news_features(features.index, ticker, records))
    # Common eligibility uses the longest target horizon for identical boundaries.
    eligible = features.dropna().index
    eligible = eligible[(eligible >= "2019-01-01") & (eligible <= prices.index[-21])]
    n = len(eligible)
    train_end, test_start = eligible[int(n * 0.70)], eligible[int(n * 0.85)]
    end = eligible[-1] + pd.Timedelta(days=1)
    protocol = {
        "ticker": ticker,
        "ratios_before_purge": [70, 15, 15],
        "origins": n,
        "start": str(eligible[0]),
        "train_end_exclusive": str(train_end),
        "test_start": str(test_start),
        "end_exclusive": str(end),
        "news_hashes": hashes,
        "ohlcv_sha256": hashlib.sha256(price_path.read_bytes()).hexdigest(),
        "diagnostics": diagnostics,
        "horizons": [5, 10, 20],
        "model": "XGBoost CUDA: 200 trees, depth 3, learning rate .03, lambda 10, min child weight 20, seed 42",
        "selection": "Fixed settings, no validation tuning; both fits use training partition only",
        "purge": "Label end strictly before each partition boundary; ratios change after purge",
        "caveat": "Exploratory reused historical data, not an independently untouched certification test",
    }
    write_json(args.output_dir / "protocol.json", protocol)
    print(
        json.dumps({k: v for k, v in protocol.items() if k not in ("news_hashes", "diagnostics")}),
        flush=True,
    )
    for partition, begin, finish in (
        ("validation", train_end, test_start),
        ("test", test_start, end),
    ):
        output = args.output_dir / partition
        output.mkdir()
        fit_and_score(
            {ticker: (prices, features)},
            output,
            "xgboost_cuda",
            split={
                "start": str(eligible[0]),
                "train_end": str(train_end),
                "score_start": str(begin),
                "score_end": str(finish),
                "partition": partition,
            },
        )


if __name__ == "__main__":
    main()
