"""Smile and term-structure interpolation on normalized strike frames.

Conventions: linear interpolation only, never extrapolation. Constant
maturity via total-variance (IV^2 * T) interpolation across the two
bracketing expiries. Liquidity-aware: smile legs prefer bracketing strikes
with a positive bid on the interpolated side, falling back to nearest
quoted strikes when the book is one-sided.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _interp_bracketed(x: np.ndarray, y: np.ndarray, target: float) -> float:
    order = np.argsort(x)
    x, y = x[order], y[order]
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2 or not (x.min() <= target <= x.max()):
        return float("nan")
    return float(np.interp(target, x, y))


def expiry_smile(slice_df: pd.DataFrame) -> dict[str, float]:
    """ATM + 25-delta legs for one expiry slice (call-delta in [0,1])."""
    out = {"atm_iv": float("nan"), "put25_iv": float("nan"), "call25_iv": float("nan")}
    s = slice_df.dropna(subset=["strike", "call_delta"]).sort_values("strike")
    if s.empty:
        return out
    stk = float(s["underlying_px"].iloc[0])
    atm_row = s.iloc[(s["strike"] - stk).abs().argsort()[:1]]
    c_iv = float(atm_row["call_mid_iv"].iloc[0])
    p_iv = float(atm_row["put_mid_iv"].iloc[0])
    if np.isfinite(c_iv) and np.isfinite(p_iv):
        out["atm_iv"] = (c_iv + p_iv) / 2.0
    # 25-delta put <=> call-delta 0.75; 25-delta call <=> call-delta 0.25.
    for key, col, bid, target in (
        ("put25_iv", "put_mid_iv", "put_bid", 0.75),
        ("call25_iv", "call_mid_iv", "call_bid", 0.25),
    ):
        liquid = s[pd.to_numeric(s[bid], errors="coerce").fillna(0) > 0]
        leg = liquid if len(liquid) >= 2 else s
        out[key] = _interp_bracketed(
            leg["call_delta"].to_numpy(float), leg[col].to_numpy(float), target
        )
    return out


def constant_maturity(points: list[tuple[float, float]], target_days: float) -> float:
    """Variance-time interpolation of IV to a fixed maturity (no extrapolation)."""
    clean = [(d, v) for d, v in points if np.isfinite(d) and np.isfinite(v) and d > 0 and v > 0]
    if not clean:
        return float("nan")
    for d, v in clean:
        if abs(d - target_days) < 1e-9:
            return float(v)
    ds = sorted(d for d, _ in clean)
    if not (ds[0] <= target_days <= ds[-1]):
        return float("nan")
    t_star = target_days / 365.0
    var_points = sorted((d / 365.0, (v**2) * (d / 365.0)) for d, v in clean)
    xs = [p[0] for p in var_points]
    w_star = float(np.interp(t_star, xs, [p[1] for p in var_points]))
    return float(np.sqrt(max(w_star, 0.0) / t_star))


def smile_term_structure(frame: pd.DataFrame, ticker: str, trade_date: pd.Timestamp) -> dict:
    """Per-expiry smiles + 30/60/90d constant-maturity ATM and 25-delta legs."""
    sub = frame[(frame.ticker == ticker) & (frame.trade_date == trade_date)]
    if sub.empty:
        raise ValueError(f"No normalized rows for {ticker} on {trade_date}")
    stk = float(sub["underlying_px"].iloc[0])
    per_expiry = []
    for expiry, sl in sub.groupby("expir_date", sort=True):
        dte = float(sl["dte_days"].iloc[0])
        if not np.isfinite(dte) or dte <= 0:
            continue
        smile = expiry_smile(sl)
        smile.update({"dte": dte, "expiry": expiry})
        per_expiry.append(smile)
    if not per_expiry:
        raise ValueError(f"No positive-DTE expiries for {ticker} on {trade_date}")
    atm_points = [(e["dte"], e["atm_iv"]) for e in per_expiry]
    put_points = [(e["dte"], e["put25_iv"]) for e in per_expiry]
    call_points = [(e["dte"], e["call25_iv"]) for e in per_expiry]
    return {
        "ticker": ticker,
        "trade_date": trade_date,
        "underlying_px": stk,
        "expiry_count": len(per_expiry),
        "dte_min": min(e["dte"] for e in per_expiry),
        "dte_max": max(e["dte"] for e in per_expiry),
        "atm_30d": constant_maturity(atm_points, 30.0),
        "atm_60d": constant_maturity(atm_points, 60.0),
        "atm_90d": constant_maturity(atm_points, 90.0),
        "put25_30d": constant_maturity(put_points, 30.0),
        "call25_30d": constant_maturity(call_points, 30.0),
    }
