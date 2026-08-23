"""Causal cross-sectional feature transformations and relative-return targets for Protocol V3.

Enforces strict asset-transfer isolation:
1. Development assets (D) are ranked against valid development assets at time t.
2. Held-out asset-transfer assets (H) are ranked against the empirical distribution of D_t
   without modifying any D asset's rank.
3. Development relative targets use a leave-one-out mean of valid development assets.
4. Held-out relative targets use the mean of valid development assets (H never contributes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from panel.features import FEATURE_COLUMNS_V5

V3_FEATURE_CONTRACT_VERSION = "cross_sectional_v3_rank_v1"
V3_TARGET_CONTRACT_VERSION = "relative_forward_log_return_dev_loo_v1"

# Base features eligible for cross-sectional ranking
V3_RANK_BASE_COLUMNS: tuple[str, ...] = (
    "Return_1D",
    "Return_5D",
    "Return_10D",
    "Return_20D",
    "Overnight_Return",
    "OpenToClose_Return",
    "Vol_C2C_20",
    "EWMA_Var",
    "Vol_Percentile_252",
    "Volume_Surprise",
    "Log_Dollar_Volume",
    "Amihud_Illiquidity_20",
)

V3_RANKED_COLUMNS: tuple[str, ...] = tuple(f"{col}_CS_Rank" for col in V3_RANK_BASE_COLUMNS)
V3_INTERACTION_COLUMNS: tuple[str, ...] = ("Return_20D_x_Vol_C2C_20_CS_Rank",)

V3_ALL_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    FEATURE_COLUMNS_V5 + list(V3_RANKED_COLUMNS) + list(V3_INTERACTION_COLUMNS)
)


@dataclass(frozen=True)
class CrossSectionalFeatureContract:
    contract_version: str = V3_FEATURE_CONTRACT_VERSION
    base_columns: tuple[str, ...] = V3_RANK_BASE_COLUMNS
    ranked_columns: tuple[str, ...] = V3_RANKED_COLUMNS
    interaction_columns: tuple[str, ...] = V3_INTERACTION_COLUMNS
    rank_range: tuple[float, float] = (-0.5, 0.5)
    min_reference_asset_count: int = 30

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "base_columns": list(self.base_columns),
            "ranked_columns": list(self.ranked_columns),
            "interaction_columns": list(self.interaction_columns),
            "rank_range": list(self.rank_range),
            "min_reference_asset_count": self.min_reference_asset_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CrossSectionalFeatureContract:
        return cls(
            contract_version=str(data.get("contract_version", V3_FEATURE_CONTRACT_VERSION)),
            base_columns=tuple(data.get("base_columns", V3_RANK_BASE_COLUMNS)),
            ranked_columns=tuple(data.get("ranked_columns", V3_RANKED_COLUMNS)),
            interaction_columns=tuple(data.get("interaction_columns", V3_INTERACTION_COLUMNS)),
            rank_range=tuple(data.get("rank_range", (-0.5, 0.5))),
            min_reference_asset_count=int(data.get("min_reference_asset_count", 30)),
        )


def compute_cross_sectional_ranks(
    panels: dict[str, pd.DataFrame],
    *,
    dev_tickers: list[str] | set[str] | None = None,
    base_columns: tuple[str, ...] = V3_RANK_BASE_COLUMNS,
    min_reference_assets: int = 30,
) -> dict[str, pd.DataFrame]:
    """Computes cross-sectional ranks mapped to [-0.5, 0.5] with strict D/H isolation.

    For development assets (i in D):
        Rank is computed among valid development assets at date t:
        z_{i, t} = (Rank(x_{i, t}) - 1) / (N_{D, t} - 1) - 0.5

    For held-out assets (k in H):
        Rank is computed relative to the empirical distribution of D at date t:
        Count number of D assets strictly less than x_{k, t}, plus 0.5 * (count equal),
        divided by N_{D, t}, then shifted by -0.5.
        Held-out assets NEVER alter any D asset's rank.

    If fewer than min_reference_assets valid D observations exist on date t,
    all ranks for that feature on date t are set to NaN.
    """
    if not panels:
        return {}

    all_tickers = sorted(panels.keys())
    dev_set = set(all_tickers) if dev_tickers is None else set(dev_tickers)

    # Stack data into panel format
    # Index: (Ticker, Date)
    stacked = {ticker: panels[ticker][list(base_columns)].copy() for ticker in all_tickers}
    combined = pd.concat(stacked, names=["Ticker", "Date"])

    # Output dictionary of DataFrames to return
    result_panels: dict[str, pd.DataFrame] = {t: panels[t].copy() for t in all_tickers}

    # Process each feature column across all dates
    for col in base_columns:
        rank_col_name = f"{col}_CS_Rank"

        # Unstack col to shape (Date, Ticker)
        unstacked = combined[col].unstack(level="Ticker")
        dev_cols = [t for t in unstacked.columns if t in dev_set]
        trans_cols = [t for t in unstacked.columns if t not in dev_set]

        dev_data = unstacked[dev_cols]

        # 1. Rank development assets cross-sectionally per date
        # Count non-NaN valid assets per date
        valid_counts = dev_data.notna().sum(axis=1)

        # Standard average rank on valid development rows: 1..N
        dev_ranks = dev_data.rank(axis=1, method="average", ascending=True, na_option="keep")

        # Map 1..N to [-0.5, 0.5]: (rank - 1.0) / (n - 1.0) - 0.5
        # When N == 1, rank becomes 0.0
        n_minus_1 = (valid_counts - 1.0).clip(lower=1.0)
        dev_scaled = (dev_ranks - 1.0).div(n_minus_1, axis=0) - 0.5
        dev_scaled[valid_counts == 1] = 0.0

        # Mask out dates with insufficient reference assets
        dev_scaled[valid_counts < min_reference_assets] = np.nan

        # 2. Evaluate held-out transfer assets against the empirical D distribution
        if trans_cols:
            trans_data = unstacked[trans_cols]
            trans_scaled = pd.DataFrame(np.nan, index=unstacked.index, columns=trans_cols)

            # For each date where valid D assets >= min_reference_assets:
            valid_dates = unstacked.index[valid_counts >= min_reference_assets]
            for date in valid_dates:
                d_vals = dev_data.loc[date].dropna().values
                n_d = len(d_vals)
                if n_d < min_reference_assets:
                    continue

                h_vals = trans_data.loc[date]
                for t in trans_cols:
                    val = h_vals[t]
                    if pd.isna(val):
                        continue
                    # Fractional rank against empirical CDF of D:
                    # (count(d < val) + 0.5 * count(d == val)) / n_d - 0.5
                    n_less = np.sum(d_vals < val)
                    n_equal = np.sum(d_vals == val)
                    pct = (n_less + 0.5 * n_equal) / n_d - 0.5
                    trans_scaled.at[date, t] = pct
        else:
            trans_scaled = pd.DataFrame(index=unstacked.index)

        # Merge ranked columns back to ticker frames
        for t in dev_cols:
            result_panels[t][rank_col_name] = dev_scaled[t]
        for t in trans_cols:
            result_panels[t][rank_col_name] = trans_scaled[t]

    # Compute interaction feature: Return_20D_CS_Rank * Vol_C2C_20_CS_Rank
    for t in all_tickers:
        df = result_panels[t]
        if "Return_20D_CS_Rank" in df.columns and "Vol_C2C_20_CS_Rank" in df.columns:
            df["Return_20D_x_Vol_C2C_20_CS_Rank"] = (
                df["Return_20D_CS_Rank"] * df["Vol_C2C_20_CS_Rank"]
            )
        else:
            df["Return_20D_x_Vol_C2C_20_CS_Rank"] = np.nan

    return result_panels


def compute_relative_forward_returns(
    panels: dict[str, pd.DataFrame],
    horizon: int,
    *,
    dev_tickers: list[str] | set[str] | None = None,
    price_column: str = "Close",
    min_reference_assets: int = 30,
) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    """Computes causal raw forward log returns and leave-one-out relative returns.

    r(i, t, h) = log(P(i, t+h) / P(i, t))

    For development assets (i in D):
        benchmark(i, t, h) = mean_{j in D_t, j != i}( r(j, t, h) )
        y_rel(i, t, h) = r(i, t, h) - benchmark(i, t, h)

    For held-out assets (k in H):
        benchmark_H(t, h) = mean_{j in D_t}( r(j, t, h) )
        y_rel(k, t, h) = r(k, t, h) - benchmark_H(t, h)

    Returns:
        (raw_forward_returns_by_ticker, relative_forward_returns_by_ticker)
    """
    if not panels:
        return {}, {}

    all_tickers = sorted(panels.keys())
    dev_set = set(all_tickers) if dev_tickers is None else set(dev_tickers)

    # 1. Compute raw forward log returns for all tickers: log(Close.shift(-h) / Close)
    raw_returns: dict[str, pd.Series] = {}
    for ticker, df in panels.items():
        close = df[price_column]
        log_close = np.log(close.where(close > 0))
        raw_returns[ticker] = log_close.shift(-horizon) - log_close

    # Combine into Date x Ticker DataFrame
    raw_df = pd.DataFrame(raw_returns)
    dev_cols = [t for t in raw_df.columns if t in dev_set]
    trans_cols = [t for t in raw_df.columns if t not in dev_set]

    dev_raw = raw_df[dev_cols]

    # Valid development counts per date
    dev_valid_counts = dev_raw.notna().sum(axis=1)
    dev_sums = dev_raw.sum(axis=1, min_count=1)

    # 2. Leave-One-Out benchmark for development assets:
    # LOO_mean(i, t) = (Sum_{D} - r(i, t)) / (N_D - 1)
    n_minus_1 = (dev_valid_counts - 1).replace(0, np.nan)
    dev_loo_mean = dev_raw.apply(lambda col: (dev_sums - col.fillna(0)) / n_minus_1, axis=0)

    # Relative return for D
    dev_relative = dev_raw - dev_loo_mean

    # Mask dates where valid reference assets < min_reference_assets
    dev_relative[dev_valid_counts < min_reference_assets] = np.nan

    # 3. Development benchmark for held-out transfer assets:
    # benchmark_H(t) = Sum_{D} / N_D
    dev_mean = dev_sums / dev_valid_counts.replace(0, np.nan)
    dev_mean[dev_valid_counts < min_reference_assets] = np.nan

    if trans_cols:
        trans_raw = raw_df[trans_cols]
        trans_relative = trans_raw.sub(dev_mean, axis=0)
    else:
        trans_relative = pd.DataFrame(index=raw_df.index)

    # Package output dictionaries
    rel_returns_dict: dict[str, pd.Series] = {}
    for t in dev_cols:
        rel_returns_dict[t] = dev_relative[t]
    for t in trans_cols:
        rel_returns_dict[t] = trans_relative[t]

    return raw_returns, rel_returns_dict
