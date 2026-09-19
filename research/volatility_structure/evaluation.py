"""Econometric evaluation metrics for volatility models, including HAC standard errors."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import norm


def hac_mean(values: Any, bandwidth: int) -> dict[str, Any]:
    """Calculate Bartlett Heteroskedasticity and Autocorrelation Consistent (HAC) standard error."""
    vals = np.asarray(values, dtype=float)
    n = len(vals)
    if n < 3 or bandwidth < 0 or bandwidth >= n or not np.isfinite(vals).all():
        raise ValueError("Invalid HAC series/bandwidth")
    mean = float(vals.mean())
    e = vals - mean
    meat = float(e @ e)
    for lag in range(1, bandwidth + 1):
        meat += 2 * (1 - lag / (bandwidth + 1)) * float(e[lag:] @ e[:-lag])
    se = float(np.sqrt(max(0.0, meat / (n * (n - 1)))))
    return {
        "dates": n,
        "mean_loss_improvement": mean,
        "se": se,
        "ci95": [mean - 1.95996398454 * se, mean + 1.95996398454 * se],
        "p_two_sided": float(2 * norm.sf(abs(mean / se))) if se else None,
    }
