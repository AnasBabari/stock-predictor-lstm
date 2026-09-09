"""Canonical daily options feature row (study contract)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schemas import FEATURE_ROW_FIELDS


def canonical_feature_row(
    *,
    date: pd.Timestamp,
    ticker: str,
    term: dict,
    rv_20d_annualized: float,
    option_volume: float,
    open_interest: float,
    source: str,
    source_timestamp: str,
) -> dict:
    """Assemble FEATURE_ROW_FIELDS. Conventions (documented, fixed):

    - iv_term_60_30 / iv_term_90_30: calendar spreads (far minus near).
    - put/call 25-delta legs: 30-day constant-maturity smile points.
    - skew_25d = put25 - call25 (positive = downside bid).
    - iv_rv_spread_20d = atm_30d minus annualized 20-day realized vol;
      rv_20d_annualized is supplied by study code (price-source agnostic).
    """
    row = {
        "date": pd.Timestamp(date).tz_localize(None),
        "ticker": ticker.strip().upper(),
        "atm_iv_30d": float(term["atm_30d"]),
        "atm_iv_60d": float(term["atm_60d"]),
        "iv_term_60_30": float(term["atm_60d"] - term["atm_30d"]),
        "iv_term_90_30": float(term["atm_90d"] - term["atm_30d"]),
        "put_iv_25d": float(term["put25_30d"]),
        "call_iv_25d": float(term["call25_30d"]),
        "skew_25d": float(term["put25_30d"] - term["call25_30d"]),
        "iv_rv_spread_20d": float(term["atm_30d"] - rv_20d_annualized),
        "option_volume": float(option_volume),
        "open_interest": float(open_interest),
        "source": source,
        "source_timestamp": source_timestamp,
    }
    missing = [f for f in FEATURE_ROW_FIELDS if f not in row]
    if missing:  # pragma: no cover - contract guard
        raise ValueError(f"Feature row missing fields: {missing}")
    return row


def realized_vol_annualized(closes: pd.Series, window: int = 20) -> float:
    """Annualized trailing realized vol from session closes (sqrt-252)."""
    rets = np.log(closes.astype(float) / closes.astype(float).shift(1)).iloc[-window:]
    if len(rets.dropna()) < window:
        return float("nan")
    return float(rets.std(ddof=1) * np.sqrt(252.0))
