#!/usr/bin/env python3
"""Run the CSCO 2026-07-20..24 software-regression benchmark.

The default path is offline and delegates to the hash-verified golden-fixture
engine. A mutable provider download is available only through ``--live`` and
is labelled non-golden in the output.

Educational research only -- not financial advice.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.ndx100.csco_fixture import (  # noqa: E402
    EXPECTED_EVIDENCE_ROLE,
    TARGET_DAYS,
    TRAIN_END,
    load_csco_golden_history,
    run_csco_benchmark,
)

TICKER = "CSCO"


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line contract without performing any I/O."""
    parser = argparse.ArgumentParser(
        prog="backtest_csco_2026_07",
        description=(
            "CSCO five-session retrospective benchmark for 2026-07-20..24. "
            "Runs offline against the hash-verified golden fixture by default."
        ),
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "Download mutable provider history instead of using the golden "
            "fixture. Live results are diagnostic and are not golden evidence."
        ),
    )
    return parser


def load_live_history() -> pd.DataFrame:
    """Download a mutable CSCO history for an explicitly requested live run."""
    import yfinance as yf

    raw = yf.download(
        TICKER,
        start="2015-01-01",
        end="2026-07-25",
        auto_adjust=True,
        progress=False,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    required = {"Close", "Volume"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise RuntimeError(f"live CSCO history is missing columns: {missing}")
    frame = raw[["Close", "Volume"]].copy()
    frame.columns = ["close", "volume"]
    if frame.isna().any().any():
        frame = frame.dropna()
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
    frame = frame.sort_index()
    if frame.empty:
        raise RuntimeError("no history downloaded for CSCO")
    if not frame.index.is_unique:
        raise RuntimeError("live CSCO history contains duplicate sessions")
    return frame


def print_results(
    history: pd.DataFrame,
    table: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    *,
    evidence_role: str,
) -> None:
    """Print the shared benchmark result without reimplementing model logic."""
    actuals = history.loc[list(TARGET_DAYS), "close"]
    train_history = history.loc[history.index <= TRAIN_END]
    # A synthetic benchmark must be impossible to misread as market evidence,
    # so the banner is printed unconditionally before any number.
    print(
        "SYNTHETIC SOFTWARE REGRESSION — NOT MARKET PERFORMANCE\n"
        "This run uses generated data. These metrics say nothing about\n"
        "forecasting skill on real markets and must never be cited as such."
    )
    print()
    print(f"evidence role: {evidence_role}")
    print(f"target week: {TARGET_DAYS[0].date()}..{TARGET_DAYS[-1].date()}")
    print(f"last training close ({TRAIN_END.date()}): {float(train_history['close'].iloc[-1]):.2f}")

    print("\nPer-day predictions:")
    header = "model".ljust(24) + "".join(day.strftime("%Y-%m-%d").rjust(12) for day in TARGET_DAYS)
    print(header)
    print("actual".ljust(24) + "".join(f"{float(value):12.2f}" for value in actuals))
    for model_name, values in predictions.items():
        print(model_name.ljust(24) + "".join(f"{float(value):12.2f}" for value in values))

    print("\nResults ordered by percentage error (MAPE):")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.live:
        history = load_live_history()
        evidence_role = "mutable_live_diagnostic"
    else:
        history = load_csco_golden_history()
        evidence_role = EXPECTED_EVIDENCE_ROLE

    # The golden contract is CPU-bound. Keeping the live diagnostic on CPU as
    # well makes differences attributable to data rather than device kernels.
    table, predictions = run_csco_benchmark(history=history, device="cpu")
    print_results(history, table, predictions, evidence_role=evidence_role)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
