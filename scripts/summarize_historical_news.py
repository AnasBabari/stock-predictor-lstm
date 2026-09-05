"""Build inspectable corpus provenance and yearly counts; never fit a model."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from run_historical_news_volatility import TICKERS, canonicalize, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    inventory, yearly, sources, diagnostics = [], [], [], {}
    for ticker in TICKERS:
        raw = []
        files = sorted(args.corpus_dir.glob(f"{ticker}_*.jsonl"))
        if len(files) != 31:
            raise ValueError(f"Incomplete quarterly download for {ticker}: {len(files)}/31")
        for file in files:
            manifest = json.loads(file.with_suffix(".manifest.json").read_text())
            digest = hashlib.sha256(file.read_bytes()).hexdigest()
            if digest != manifest["sha256"] or not manifest.get("pagination_complete"):
                raise ValueError(f"Invalid completed chunk {file.name}")
            inventory.append({"file": file.name, **manifest})
            raw.extend(json.loads(line) for line in file.read_text().splitlines() if line)
        retained, diagnostics[ticker] = canonicalize(raw)
        output = args.output_dir / f"{ticker}.jsonl"
        output.write_bytes(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in retained).encode()
        )
        diagnostics[ticker]["canonical_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
        for year in range(2019, 2027):
            yearly.append(
                {
                    "ticker": ticker,
                    "year": year,
                    "raw_publication_records": sum(
                        r["published_at"].startswith(str(year)) for r in raw
                    ),
                    "retained_available_records": sum(
                        r["t_available"].startswith(str(year)) for r in retained
                    ),
                }
            )
        sources.extend(
            {"ticker": ticker, "source": source, "raw_count": count}
            for source, count in Counter(r.get("source", "unknown") for r in raw).items()
        )
    pd.DataFrame(yearly).to_csv(args.output_dir / "ticker_year_counts.csv", index=False)
    pd.DataFrame(sources).to_csv(args.output_dir / "source_mix.csv", index=False)
    write_json(
        args.output_dir / "corpus.manifest.json",
        {
            "chunks": inventory,
            "diagnostics": diagnostics,
            "schema_version": "news-event-v2",
            "eligibility": "exploratory provider-update-gated history; not independently verified versions",
            "publication_window": "2019-01-01 <= publication < 2026-09-05",
            "provider": "alpaca",
        },
    )
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
