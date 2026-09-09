"""yfinance current-chain probe. PLUMBING ONLY — never study eligible.

Exercises chain -> normalize -> surface -> feature-row end to end on live
data. Approximations (BS delta with ^IRX rate, current dividend yield) are
documented here and quarantined by the source label; validation.py rejects
this source for any study dataset.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from ..schemas import NORMALIZED_COLUMNS, SOURCE_YFINANCE_PLUMBING
from .base import OptionsProvider


def _bs_call_delta(forward: float, strike: float, t_years: float, vol: float, rate: float) -> float:
    from math import erf, log, sqrt

    if (
        not (np.isfinite([forward, strike, t_years, vol]).all())
        or forward <= 0
        or strike <= 0
        or t_years <= 0
        or vol <= 0
    ):
        return float("nan")

    def _normal_cdf(x: float) -> float:
        return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))

    d1 = (log(forward / strike) + (rate + 0.5 * vol * vol) * t_years) / (vol * sqrt(t_years))
    return float(_normal_cdf(d1))


class YFinanceProbeProvider(OptionsProvider):
    """Current chain for one ticker; output flagged ineligible for study."""

    name = "yfinance_probe"

    def read(self, ticker: str, max_expiries: int | None = None) -> pd.DataFrame:
        import yfinance as yf

        symbol = ticker.strip().upper()
        tk = yf.Ticker(symbol)
        expiries = list(tk.options or [])
        if not expiries:
            raise ValueError(f"No yfinance option expiries for {symbol}")
        if max_expiries:
            expiries = expiries[:max_expiries]
        try:
            rate = (
                float(
                    yf.download("^IRX", period="5d", progress=False, auto_adjust=False)[
                        "Close"
                    ].iloc[-1, 0]
                )
                / 100.0
            )
        except Exception:
            rate = 0.04
        try:
            div_yield = float(tk.info.get("dividendYield") or 0.0)
        except Exception:
            div_yield = 0.0
        today = datetime.now(UTC).date().isoformat()
        under = float(tk.history(period="1d")["Close"].iloc[-1])
        rows = []
        for expiry in expiries:
            chain = tk.option_chain(expiry)
            exp_date = pd.Timestamp(expiry).tz_localize(None)
            dte = max((exp_date.date() - datetime.now(UTC).date()).days, 0)
            t_years = max(dte, 1) / 365.0
            calls = chain.calls.set_index("strike")
            puts = chain.puts.set_index("strike")
            for strike in sorted(set(calls.index) & set(puts.index)):
                c, p = calls.loc[strike], puts.loc[strike]
                fwd = under * np.exp((rate - div_yield) * t_years)
                cvol = float(c.get("impliedVolatility", float("nan")))
                rows.append(
                    {
                        "ticker": symbol,
                        "trade_date": pd.Timestamp.now().tz_localize(None).normalize(),
                        "expir_date": exp_date,
                        "dte_days": float(dte),
                        "strike": float(strike),
                        "underlying_px": under,
                        "call_mid_iv": cvol,
                        "put_mid_iv": float(p.get("impliedVolatility", float("nan"))),
                        "call_bid": float(c.get("bid", 0.0)),
                        "put_bid": float(p.get("bid", 0.0)),
                        "call_volume": float(c.get("volume", 0.0) or 0.0),
                        "put_volume": float(p.get("volume", 0.0) or 0.0),
                        "call_oi": float(c.get("openInterest", 0.0) or 0.0),
                        "put_oi": float(p.get("openInterest", 0.0) or 0.0),
                        "call_delta": _bs_call_delta(fwd, float(strike), t_years, cvol, rate),
                        "source": SOURCE_YFINANCE_PLUMBING,
                        "source_timestamp": f"live-probe {today}",
                    }
                )
        frame = pd.DataFrame(rows, columns=list(NORMALIZED_COLUMNS))
        if frame.empty:
            raise ValueError(f"Empty yfinance chain frame for {symbol}")
        return frame
