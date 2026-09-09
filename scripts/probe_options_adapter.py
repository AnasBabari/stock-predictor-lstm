"""Gate 0 proof: ORATS sample -> canonical row + yfinance probe quarantine.

1. Reads the free ORATS near-EOD sample for one ticker/date, builds the
   term structure, assembles the canonical feature row (RV from the local
   SPY cache), validates the contract, and asserts study eligibility.
2. Runs the yfinance plumbing probe for the same ticker, assembles its row,
   and asserts the eligibility gate REJECTS it.
No study dataset is written; this proves parser + features + joins only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.options.features import canonical_feature_row, realized_vol_annualized  # noqa: E402
from data.options.normalize import check_normalized  # noqa: E402
from data.options.providers.orats import OratsStrikesProvider  # noqa: E402
from data.options.providers.yfinance_probe import YFinanceProbeProvider  # noqa: E402
from data.options.schemas import STUDY_ELIGIBLE_SOURCES  # noqa: E402
from data.options.surface import smile_term_structure  # noqa: E402
from data.options.validation import assert_study_eligible, check_feature_row  # noqa: E402

SAMPLE = REPO_ROOT / "data/options/samples/ORATS_SMV_Strikes_20240103.zip"
TICKER = "SPY"


def main() -> int:
    # --- Leg 1: ORATS sample determinism ---------------------------------
    frame = OratsStrikesProvider().read(SAMPLE, tickers=[TICKER])
    violations = check_normalized(frame)
    assert not violations, violations
    trade_date = frame["trade_date"].iloc[0]
    term = smile_term_structure(frame, TICKER, trade_date)
    spy_hist = pd.read_parquet(REPO_ROOT / "data/macro/market_dailies.parquet")["SPY"]
    spy_hist.index = pd.to_datetime(spy_hist.index).tz_localize(None)
    rv = realized_vol_annualized(spy_hist.loc[: trade_date.normalize()].iloc[:-1])
    day = frame[(frame.expir_date > trade_date)]
    agg_vol = float(day["call_volume"].sum() + day["put_volume"].sum())
    agg_oi = float(day["call_oi"].sum() + day["put_oi"].sum())
    row = canonical_feature_row(
        date=trade_date,
        ticker=TICKER,
        term=term,
        rv_20d_annualized=rv,
        option_volume=agg_vol,
        open_interest=agg_oi,
        source=frame["source"].iloc[0],
        source_timestamp=frame["source_timestamp"].iloc[0],
    )
    errors = check_feature_row(row)
    assert not errors, errors
    assert_study_eligible(row)
    print("ORATS leg: contract clean, study eligible")
    for key in (
        "atm_iv_30d",
        "atm_iv_60d",
        "iv_term_60_30",
        "iv_term_90_30",
        "put_iv_25d",
        "call_iv_25d",
        "skew_25d",
        "iv_rv_spread_20d",
        "option_volume",
        "open_interest",
        "source",
    ):
        print(f"  {key} = {row[key]}")

    # --- Leg 2: yfinance quarantine ---------------------------------------
    probe = YFinanceProbeProvider().read(TICKER)
    assert (probe["source"] == "YFINANCE_EPHEMERAL_PLUMBING_ONLY").all()
    assert probe["source"].iloc[0] not in STUDY_ELIGIBLE_SOURCES
    pterm = smile_term_structure(probe, TICKER, probe["trade_date"].iloc[0])
    prow = canonical_feature_row(
        date=probe["trade_date"].iloc[0],
        ticker=TICKER,
        term=pterm,
        rv_20d_annualized=float("nan"),
        option_volume=float(probe["call_volume"].sum() + probe["put_volume"].sum()),
        open_interest=float(probe["call_oi"].sum() + probe["put_oi"].sum()),
        source=probe["source"].iloc[0],
        source_timestamp=probe["source_timestamp"].iloc[0],
    )
    try:
        assert_study_eligible(prow)
    except ValueError:
        print("yfinance leg: eligibility gate correctly REJECTED the probe row")
    else:
        raise AssertionError("quarantine breach: probe row accepted as study eligible")
    print("GATE 0 PROOF COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
