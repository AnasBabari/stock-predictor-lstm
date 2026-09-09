"""Sign-conditioned daily variation and leverage terms (study branch only).

Terminology is deliberately narrow: with daily bars we compute
sign-conditioned daily squared-return variation, NOT realized semivariance
in the high-frequency sense. All quantities are raw and causal (day ``t``
uses closes up to ``t`` only); flooring or transforms belong in the harness.
The first row is always NaN (no prior close); NaN closes propagate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def asymmetry_features(close: pd.Series) -> pd.DataFrame:
    """Return per-session downside/upside squared returns and leverage terms.

    Columns: ``downside_sq`` (r^2 where r<0 else 0), ``upside_sq`` (r^2 where
    r>0 else 0), ``neg_indicator`` (1.0 where r<0 else 0.0),
    ``neg_return`` (r where r<0 else 0.0), with r the log return.
    """
    prices = pd.Series(close, dtype=float)
    if prices.empty:
        raise ValueError("Close input is empty")
    returns = np.log(prices / prices.shift(1))
    valid = np.isfinite(returns.to_numpy())
    negative = valid & (returns.to_numpy() < 0.0)
    positive = valid & (returns.to_numpy() > 0.0)
    squared = np.where(valid, returns.to_numpy() ** 2, np.nan)
    return pd.DataFrame(
        {
            "downside_sq": np.where(negative, squared, np.where(valid, 0.0, np.nan)),
            "upside_sq": np.where(positive, squared, np.where(valid, 0.0, np.nan)),
            "neg_indicator": np.where(valid, np.where(negative, 1.0, 0.0), np.nan),
            "neg_return": np.where(negative, returns.to_numpy(), np.where(valid, 0.0, np.nan)),
        },
        index=prices.index,
    )
