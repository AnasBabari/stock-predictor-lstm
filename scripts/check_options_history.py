"""Offline options inventory gate; no fitting, target scoring, or network access."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.options.providers.orats import OratsStrikesProvider  # noqa: E402
from data.options.surface import smile_term_structure  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(
        (ROOT / "artifacts/price_validation_comparison_20260905_003731/protocol.json").read_text()
    )
    universe = sorted(protocol["file_hashes"])
    # This is the previous 286-stock basket for diagnostic coverage, not a new study universe.
    files = sorted((ROOT / "data/options").rglob("*.zip"))
    inventory, coverage = [], []
    for path in files:
        if not path.name.startswith("ORATS_SMV_Strikes_"):
            continue
        raw = OratsStrikesProvider().read(path, tickers=universe)
        inventory.append(
            {
                "file": str(path.relative_to(ROOT)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "rows_in_diagnostic_universe": len(raw),
                "dates": sorted(raw.trade_date.dt.strftime("%Y-%m-%d").unique().tolist()),
                "tickers": int(raw.ticker.nunique()),
                "duplicate_strike_rows": int(
                    raw.duplicated(["ticker", "trade_date", "expir_date", "strike"]).sum()
                ),
            }
        )
        for (ticker, date), group in raw.groupby(["ticker", "trade_date"]):
            try:
                term = smile_term_structure(group, ticker, date)
                error = None
            except ValueError as exc:
                term, error = {}, str(exc)
            for feature in ("atm_30d", "atm_60d", "atm_90d", "put25_30d", "call25_30d"):
                coverage.append(
                    {
                        "ticker": ticker,
                        "year": int(date.year),
                        "date": str(date.date()),
                        "feature": feature,
                        "finite": bool(np.isfinite(term.get(feature, np.nan))),
                        "error": error,
                    }
                )
    detail = pd.DataFrame(coverage)
    detail.to_csv(args.output_dir / "observed_feature_coverage.csv", index=False)
    present = set(detail.ticker) if not detail.empty else set()
    summary = {
        "decision": "NO_GO",
        "status": "blocked_historical_data_coverage",
        "models_fit": 0,
        "diagnostic_universe_size": len(universe),
        "study_universe_and_period": "not explicitly fixed in preregistration",
        "source_files": inventory,
        "absent_tickers": sorted(set(universe) - present),
        "missingness_policy": "No forward fill; finite coverage describes observed sample dates only, not continuous history",
        "timestamp_verification": "UNVERIFIED: adapter synthesizes 15:46 ET; no per-record availability timestamp checked",
        "target_window_verification": "NOT_PERFORMED: no target windows constructed; no training authorized",
        "blockers": [
            "Only a single-day sample is available locally",
            "No fixed study universe/period or quantitative continuity criteria",
            "Historical entitlement and source availability timing not verified",
        ],
        "adapter_findings": [
            "Missing volume/OI is converted to zero by normalize_orats, hiding raw missingness",
            "Source allow-list alone does not prove historical coverage or temporal eligibility",
            "IV percentile requires history and is not produced by the current feature row",
        ],
    }
    (args.output_dir / "gate.json").write_text(json.dumps(summary, indent=2))
    if not detail.empty:
        detail.groupby(["ticker", "year", "feature"]).finite.agg(["count", "sum", "mean"]).to_csv(
            args.output_dir / "ticker_year_feature.csv"
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
