"""HAR wrapper reusing the production implementation (study branch only).

Nothing here re-derives HAR mathematics: recursive multi-step forecasts
come from ``services.volatility_snapshot.causal_log_har_forecasts`` and the
daily/weekly/monthly log components come from its ``_log_har_row``
constructor. Parity tests pin this delegation, so any production change
fails loudly here instead of silently shifting study results.

Warm-up semantics (inherited, documented): components need 22 realized
observations; recursive forecasts need origin >= 60 with at least 20 fitted
rows, refit every 5 origins. Every value at origin ``t`` depends only on
observations through ``t``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from services.volatility_snapshot import (
    _log_har_row,
    causal_log_har_forecasts,
    realized_variance_proxies,
)

HAR_HORIZONS: tuple[int, int, int] = (5, 10, 20)
HAR_MINIMUM_HISTORY = 60
HAR_REFIT_EVERY = 5
HAR_RIDGE = 1e-4


def daily_proxy(close: pd.Series) -> pd.Series:
    """Close-to-close daily realized-variance proxy (production definition)."""
    prices = pd.Series(close, dtype=float)
    if prices.empty:
        raise ValueError("Close input is empty")
    proxy = realized_variance_proxies(pd.DataFrame({"Close": prices}))
    return proxy["RV_C2C"]


def har_log_components(close: pd.Series) -> pd.DataFrame:
    """Trailing daily/weekly/monthly log components via the production row constructor."""
    proxy = daily_proxy(close)
    rows = []
    for end in range(len(proxy)):
        if end < 21:
            rows.append((np.nan, np.nan, np.nan))
            continue
        window = proxy.to_numpy()[end - 21 : end + 1]
        if not np.isfinite(window).all():
            rows.append((np.nan, np.nan, np.nan))
            continue
        _, daily, weekly, monthly = _log_har_row(window)
        rows.append((float(daily), float(weekly), float(monthly)))
    return pd.DataFrame(
        rows, columns=["har_log_daily", "har_log_weekly", "har_log_monthly"], index=proxy.index
    )


def har_forecasts(
    close: pd.Series,
    horizons: tuple[int, ...] = HAR_HORIZONS,
    *,
    minimum_history: int = HAR_MINIMUM_HISTORY,
    refit_every: int = HAR_REFIT_EVERY,
    ridge: float = HAR_RIDGE,
) -> pd.DataFrame:
    """Causal recursive cumulative-variance forecasts per origin (production engine)."""
    proxy = daily_proxy(close)
    values = causal_log_har_forecasts(
        proxy,
        list(horizons),
        minimum_history=minimum_history,
        refit_every=refit_every,
        ridge=ridge,
    )
    return pd.DataFrame(values, columns=[f"har_h{h}" for h in horizons], index=proxy.index)
