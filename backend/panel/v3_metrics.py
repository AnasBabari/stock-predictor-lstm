"""Session-level Rank Information Coefficient (IC) and statistical inference for Protocol V3.

Implements:
1. Daily/session Spearman rank IC computation across active cross-section (never pooled panel correlation).
2. Newey-West HAC variance estimation for overlapping forward-return horizons (lag = horizon - 1).
3. Deterministic moving-block bootstrap for empirical 95% confidence intervals on mean IC.
4. Holm familywise multiple-testing correction across (horizon, candidate_name) hypotheses.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class SessionICMetrics:
    n_eligible_sessions: int
    n_valid_ic_sessions: int
    ic_session_coverage: float
    mean_spearman_ic: float
    median_spearman_ic: float
    std_spearman_ic: float
    positive_ic_hit_rate: float
    min_daily_asset_breadth: int
    median_daily_asset_breadth: float
    prediction_row_coverage: float

    # Statistical inference
    hac_lag: int
    hac_se: float
    hac_t_stat: float
    raw_one_sided_hac_p: float
    mean_ic_ci_lower_95: float
    mean_ic_ci_upper_95: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_session_rank_ic(
    scores: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    min_daily_asset_count: int = 30,
) -> tuple[pd.Series, pd.Series]:
    """Computes Spearman rank correlation per session across active universe.

    scores: Date x Ticker DataFrame of predicted ranking scores.
    targets: Date x Ticker DataFrame of realized relative forward returns.

    A session IC is valid only when:
    - At least min_daily_asset_count tickers have finite score and target on date t.
    - Score is not constant (nunique > 1, variance > 1e-12).
    - Target is not constant (nunique > 1, variance > 1e-12).

    Returns:
        (daily_ic_series, daily_breadth_series) where invalid dates are NaN.
    """
    common_dates = scores.index.intersection(targets.index)
    common_tickers = scores.columns.intersection(targets.columns)

    if len(common_dates) == 0 or len(common_tickers) == 0:
        return pd.Series(dtype=float), pd.Series(dtype=int)

    s_aligned = scores.loc[common_dates, common_tickers]
    t_aligned = targets.loc[common_dates, common_tickers]

    daily_ic = pd.Series(np.nan, index=common_dates, dtype=float)
    daily_breadth = pd.Series(0, index=common_dates, dtype=int)

    for date in common_dates:
        s_row = s_aligned.loc[date].values.astype(float)
        t_row = t_aligned.loc[date].values.astype(float)

        valid_mask = np.isfinite(s_row) & np.isfinite(t_row)
        n_valid = int(np.sum(valid_mask))
        daily_breadth.at[date] = n_valid

        if n_valid < min_daily_asset_count:
            continue

        s_val = s_row[valid_mask]
        t_val = t_row[valid_mask]

        if np.std(s_val) < 1e-12 or np.std(t_val) < 1e-12:
            continue

        # Spearman rank correlation
        res = stats.spearmanr(s_val, t_val)
        corr = float(res.statistic if hasattr(res, "statistic") else res[0])
        if np.isfinite(corr):
            daily_ic.at[date] = corr

    return daily_ic, daily_breadth


def compute_ic_hac(
    daily_ic: pd.Series | np.ndarray,
    horizon: int,
    *,
    custom_lag: int | None = None,
) -> tuple[float, float, float]:
    """Computes Newey-West HAC standard error, t-stat, and one-sided p-value for mean IC.

    Hypothesis test:
        H0: E[IC] <= 0
        H1: E[IC] > 0

    Lag policy:
        lag(h) = h - 1 (predeclared horizon-aware lag).

    Returns:
        (hac_se, hac_t_stat, raw_one_sided_p_value)
    """
    if isinstance(daily_ic, pd.Series):
        arr = daily_ic.dropna().values.astype(float)
    else:
        arr = np.asarray(daily_ic, dtype=float)
        arr = arr[np.isfinite(arr)]

    n = len(arr)
    if n < 5:
        return float("inf"), 0.0, 1.0

    mean_ic = float(np.mean(arr))
    demeaned = arr - mean_ic

    lag_max = custom_lag if custom_lag is not None else max(0, horizon - 1)
    lag_max = min(lag_max, n - 2)

    var = float(np.var(demeaned, ddof=0))
    for lag in range(1, lag_max + 1):
        cov = float(np.mean(demeaned[:-lag] * demeaned[lag:]))
        weight = 1.0 - (lag / (lag_max + 1.0))
        var += 2.0 * weight * cov

    var = max(var, 0.0)
    se = float(np.sqrt(var / n))

    if se < 1e-12:
        t_stat = 100.0 if mean_ic > 0 else (-100.0 if mean_ic < 0 else 0.0)
    else:
        t_stat = float(mean_ic / se)

    # One-sided p-value: P(Z >= t_stat) under N(0, 1)
    p_val = float(1.0 - 0.5 * (1.0 + math.erf(t_stat / math.sqrt(2.0))))
    p_val = max(0.0, min(1.0, p_val))

    return se, t_stat, p_val


def compute_ic_moving_block_bootstrap(
    daily_ic: pd.Series | np.ndarray,
    horizon: int,
    *,
    resamples: int = 2000,
    seed: int = 42,
    custom_block_length: int | None = None,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Computes moving-block bootstrap confidence interval for mean daily IC.

    Block length policy:
        block_length(h) = max(5, h)

    Returns:
        (lower_bound_ci, upper_bound_ci) for (1 - alpha) * 100% confidence.
    """
    if isinstance(daily_ic, pd.Series):
        arr = daily_ic.dropna().values.astype(float)
    else:
        arr = np.asarray(daily_ic, dtype=float)
        arr = arr[np.isfinite(arr)]

    n = len(arr)
    if n < 5:
        return -float("inf"), float("inf")

    bl = custom_block_length if custom_block_length is not None else max(5, horizon)
    bl = max(1, min(bl, n // 2))

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / bl))
    num_starts = n - bl + 1

    boot_means = np.empty(resamples, dtype=float)

    for i in range(resamples):
        starts = rng.integers(0, num_starts, size=n_blocks)
        indices = np.concatenate([np.arange(s, s + bl) for s in starts])[:n]
        boot_means[i] = float(np.mean(arr[indices]))

    lo_pct = (alpha / 2.0) * 100.0
    hi_pct = (1.0 - alpha / 2.0) * 100.0

    lo = float(np.percentile(boot_means, lo_pct))
    hi = float(np.percentile(boot_means, hi_pct))

    return lo, hi


def holm_bonferroni_family(
    p_values: dict[tuple[int, str], float],
    *,
    alpha: float = 0.05,
) -> dict[tuple[int, str], tuple[bool, float]]:
    """Applies Holm-Bonferroni step-down correction across (horizon, candidate_name) family.

    Returns:
        Dictionary mapping (horizon, candidate_name) -> (reject_null_decision, adjusted_p_value)
    """
    if not p_values:
        return {}

    items = list(p_values.items())
    # Sort items by raw p-value ascending
    sorted_items = sorted(items, key=lambda x: x[1])
    m = len(sorted_items)

    results: dict[tuple[int, str], tuple[bool, float]] = {}

    cum_adj = 0.0
    for rank, (key, raw_p) in enumerate(sorted_items):
        multiplier = m - rank
        adj_p = min(1.0, raw_p * multiplier)
        # Monotonicity enforcement for step-down adjusted p-values
        cum_adj = max(cum_adj, adj_p)
        adj_p = min(1.0, cum_adj)
        reject = adj_p <= alpha
        results[key] = (reject, float(adj_p))

    return results


def evaluate_session_ic_statistics(
    scores: pd.DataFrame,
    targets: pd.DataFrame,
    horizon: int,
    *,
    min_daily_asset_count: int = 30,
    resamples: int = 2000,
    seed: int = 42,
) -> SessionICMetrics:
    """Full statistical evaluation of daily IC for a candidate score matrix."""
    daily_ic, daily_breadth = compute_session_rank_ic(
        scores, targets, min_daily_asset_count=min_daily_asset_count
    )

    n_eligible = len(scores.index.intersection(targets.index))
    valid_ic = daily_ic.dropna()
    n_valid = len(valid_ic)

    ic_session_coverage = float(n_valid / max(1, n_eligible))

    # Prediction row coverage
    total_possible_rows = scores.size
    finite_pred_rows = int(np.sum(np.isfinite(scores.values)))
    pred_row_coverage = float(finite_pred_rows / max(1, total_possible_rows))

    if n_valid == 0:
        return SessionICMetrics(
            n_eligible_sessions=n_eligible,
            n_valid_ic_sessions=0,
            ic_session_coverage=0.0,
            mean_spearman_ic=0.0,
            median_spearman_ic=0.0,
            std_spearman_ic=0.0,
            positive_ic_hit_rate=0.0,
            min_daily_asset_breadth=0,
            median_daily_asset_breadth=0.0,
            prediction_row_coverage=pred_row_coverage,
            hac_lag=max(0, horizon - 1),
            hac_se=float("inf"),
            hac_t_stat=0.0,
            raw_one_sided_hac_p=1.0,
            mean_ic_ci_lower_95=-float("inf"),
            mean_ic_ci_upper_95=float("inf"),
        )

    mean_ic = float(valid_ic.mean())
    med_ic = float(valid_ic.median())
    std_ic = float(valid_ic.std(ddof=1)) if n_valid > 1 else 0.0
    hit_rate = float((valid_ic > 0).mean())

    valid_breadths = daily_breadth.loc[valid_ic.index]
    min_breadth = int(valid_breadths.min())
    med_breadth = float(valid_breadths.median())

    hac_lag = max(0, horizon - 1)
    hac_se, hac_t, raw_p = compute_ic_hac(valid_ic, horizon)
    ci_lo, ci_hi = compute_ic_moving_block_bootstrap(
        valid_ic, horizon, resamples=resamples, seed=seed
    )

    return SessionICMetrics(
        n_eligible_sessions=n_eligible,
        n_valid_ic_sessions=n_valid,
        ic_session_coverage=ic_session_coverage,
        mean_spearman_ic=mean_ic,
        median_spearman_ic=med_ic,
        std_spearman_ic=std_ic,
        positive_ic_hit_rate=hit_rate,
        min_daily_asset_breadth=min_breadth,
        median_daily_asset_breadth=med_breadth,
        prediction_row_coverage=pred_row_coverage,
        hac_lag=hac_lag,
        hac_se=hac_se,
        hac_t_stat=hac_t,
        raw_one_sided_hac_p=raw_p,
        mean_ic_ci_lower_95=ci_lo,
        mean_ic_ci_upper_95=ci_hi,
    )
