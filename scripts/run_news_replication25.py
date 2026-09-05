"""Run the fixed replication roster sequentially without tuning or replacement."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TICKERS = [
    "ADBE",
    "AMD",
    "CSCO",
    "IBM",
    "JPM",
    "BAC",
    "GS",
    "AXP",
    "JNJ",
    "MRK",
    "ABT",
    "GILD",
    "CAT",
    "DE",
    "HON",
    "LMT",
    "CVX",
    "COP",
    "EOG",
    "KO",
    "COST",
    "MCD",
    "HD",
    "DIS",
    "AEP",
]


def main():
    output = ROOT / "artifacts/news_replication25_v1"
    output.mkdir(exist_ok=False)
    guarded = [
        "docs/NEWS_REPLICATION_25.md",
        "scripts/run_news_replication25.py",
        "scripts/run_msft_news_701515.py",
        "scripts/run_historical_news_volatility.py",
        "research/price_forecasting/news_archive.py",
        "research/price_forecasting/gpu_pipeline.py",
    ]
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in guarded}
    (output / "lock.json").write_text(json.dumps({"tickers": TICKERS, "hashes": hashes}, indent=2))
    results = []
    for ticker in TICKERS:
        if any(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != h for p, h in hashes.items()):
            raise RuntimeError("Locked implementation changed; stopping replication")
        print(f"START {ticker}", flush=True)
        with (output / f"{ticker}.log").open("w") as log:
            run = subprocess.run(
                [
                    sys.executable,
                    "-u",
                    str(ROOT / "scripts/run_msft_news_701515.py"),
                    "--ticker",
                    ticker,
                    "--output-dir",
                    str(output / ticker),
                ],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        results.append({"ticker": ticker, "exit_code": run.returncode})
        (output / "progress.json").write_text(json.dumps(results, indent=2))
        print(f"FINISH {ticker} exit={run.returncode}", flush=True)
    (output / "completion.json").write_text(
        json.dumps(
            {
                "attempted": len(results),
                "succeeded": sum(r["exit_code"] == 0 for r in results),
                "analysis_complete": False,
                "deployment": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
