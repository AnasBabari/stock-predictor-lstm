"""Daily-OHLC range variance estimators for the volatility-structure study.

All outputs are per-session daily VARIANCES (not annualized vol): Parkinson,
Garman-Klass, Rogers-Satchell in closed form plus a trailing-window
Yang-Zhang. Inputs are native-currency OHLC series; invalid rows (non-positive
prices, high < low) yield NaN, never an exception, so long research panels do
not die on one bad print. Empty input raises. Note Garman-Klass is unbiased
but can print negative single-day values on strong trend days; floor it in
the study harness if a feature requires non-negativity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_FOUR_LOG_2 = 4.0 * np.log(2.0)
_TWO_LOG_2_MINUS_1 = 2.0 * np.log(2.0) - 1.0


def _as_float(series: pd.Series, name: str) -> pd.Series:
    values = pd.Series(series, dtype=float)
    if values.empty:
        raise ValueError("OHLC input is empty")
    return values


def range_variances(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    yang_zhang_window: int = 22,
) -> pd.DataFrame:
    """Return per-session Parkinson / Garman-Klass / Rogers-Satchell / Yang-Zhang variances."""
    if yang_zhang_window < 2:
        raise ValueError("yang_zhang_window must be at least 2")
    o = _as_float(open_, "open")
    h = _as_float(high, "high")
    lo = _as_float(low, "low")
    c = _as_float(close, "close")
    index = o.index
    for series in (h, lo, c):
        if not series.index.equals(index):
            raise ValueError("OHLC indexes are misaligned")
    valid = (
        np.isfinite(o.to_numpy())
        & np.isfinite(h.to_numpy())
        & np.isfinite(lo.to_numpy())
        & np.isfinite(c.to_numpy())
        & (o.to_numpy() > 0)
        & (h.to_numpy() > 0)
        & (lo.to_numpy() > 0)
        & (c.to_numpy() > 0)
        & (h.to_numpy() >= lo.to_numpy())
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        log_hl = np.log(h.to_numpy() / lo.to_numpy())
        log_co = np.log(c.to_numpy() / o.to_numpy())
        log_hc = np.log(h.to_numpy() / c.to_numpy())
        log_ho = np.log(h.to_numpy() / o.to_numpy())
        log_lc = np.log(lo.to_numpy() / c.to_numpy())
        log_lo = np.log(lo.to_numpy() / o.to_numpy())
        parkinson = log_hl**2 / _FOUR_LOG_2
        garman_klass = 0.5 * log_hl**2 - _TWO_LOG_2_MINUS_1 * log_co**2
        rogers_satchell = log_hc * log_ho + log_lc * log_lo
    frame = pd.DataFrame(
        {
            "parkinson_var": np.where(valid, parkinson, np.nan),
            "garman_klass_var": np.where(valid, garman_klass, np.nan),
            "rogers_satchell_var": np.where(valid, rogers_satchell, np.nan),
        },
        index=index,
    )
    frame["yang_zhang_var"] = _yang_zhang(o, h, lo, c, valid, yang_zhang_window)
    return frame


def _yang_zhang(
    o: pd.Series,
    h: pd.Series,
    lo: pd.Series,
    c: pd.Series,
    valid: np.ndarray,
    window: int,
) -> pd.Series:
    """Trailing-window Yang-Zhang variance.

    The overnight leg needs one prior close, so output is NaN for the first
    ``window`` sessions (indices 0..window-1) and finite from ``window`` on.
    """
    o_arr, c_arr = o.to_numpy(), c.to_numpy()
    prev_c = np.roll(c_arr, 1)
    prev_c[0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        overnight = np.log(o_arr / prev_c)
        open_close = np.log(c_arr / o_arr)
        log_hc = np.log(h.to_numpy() / c_arr)
        log_ho = np.log(h.to_numpy() / o_arr)
        log_lc = np.log(lo.to_numpy() / c_arr)
        log_lo = np.log(lo.to_numpy() / o_arr)
        rs_day = log_hc * log_ho + log_lc * log_lo
    overnight = pd.Series(np.where(valid, overnight, np.nan), index=o.index)
    open_close = pd.Series(np.where(valid, open_close, np.nan), index=o.index)
    rs_day = pd.Series(np.where(valid, rs_day, np.nan), index=o.index)
    var_o = overnight.rolling(window, min_periods=window).var(ddof=1)
    var_c = open_close.rolling(window, min_periods=window).var(ddof=1)
    mean_rs = rs_day.rolling(window, min_periods=window).mean()
    n = float(window)
    k = 0.34 / (1.34 + (n + 1.0) / (n - 1.0))
    return var_o + k * var_c + (1.0 - k) * mean_rs
