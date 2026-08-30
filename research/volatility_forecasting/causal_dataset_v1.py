"""Strict point-in-time stationary causal dataset loader and feature pipeline."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

SECURITY_ID_REGEX = re.compile(r"^SEC_[A-Z0-9.\-]{1,12}_[0-9]{3,6}$")

STATIONARY_FEATURE_COLUMNS_V1: tuple[str, ...] = (
    "log_return_1d",
    "overnight_return",
    "intraday_range",
    "parkinson_vol_5d",
    "parkinson_vol_20d",
    "garman_klass_vol_5d",
    "log_volume_change_1d",
    "volume_ratio_20d",
    "momentum_3d",
    "momentum_5d",
    "momentum_10d",
    "momentum_20d",
    "drawdown_20d",
    "close_to_high_ratio",
    "close_to_low_ratio",
    "hl_spread_ma_ratio_5d",
)


@dataclass(frozen=True)
class CausalDatasetMetadata:
    snapshot_hash: str
    feature_names: list[str]
    security_count: int
    session_count: int
    start_session: str
    end_session: str


class StrictCausalDatasetLoader:
    """Strict point-in-time causal dataset loader with stationary features and zero lookahead."""

    @staticmethod
    def compute_stationary_features(ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        """Derive strictly stationary, causal features from OHLCV prices without future lookahead."""
        required = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in ohlcv_df.columns for col in required):
            raise ValueError(f"OHLCV DataFrame must contain all columns: {required}")

        df = ohlcv_df.copy()
        for col in required:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].isna().any() or (df[col] <= 0).any():
                raise ValueError(f"Column {col} contains invalid non-positive or NaN values.")

        c = df["Close"].to_numpy(dtype=float)
        o = df["Open"].to_numpy(dtype=float)
        h = df["High"].to_numpy(dtype=float)
        low_p = df["Low"].to_numpy(dtype=float)
        v = df["Volume"].to_numpy(dtype=float)

        feats = pd.DataFrame(index=df.index)

        # 1. Returns
        log_ret = np.zeros_like(c)
        log_ret[1:] = np.log(c[1:] / c[:-1])
        feats["log_return_1d"] = log_ret

        overnight = np.zeros_like(c)
        overnight[1:] = np.log(o[1:] / c[:-1])
        feats["overnight_return"] = overnight

        # 2. Range
        feats["intraday_range"] = (h - low_p) / c

        # 3. Parkinson Volatility (5d, 20d)
        log_hl_sq = (np.log(h / low_p)) ** 2 / (4.0 * np.log(2.0))
        p_vol5 = pd.Series(log_hl_sq, index=df.index).rolling(5, min_periods=5).mean()
        p_vol20 = pd.Series(log_hl_sq, index=df.index).rolling(20, min_periods=20).mean()
        feats["parkinson_vol_5d"] = np.sqrt(np.maximum(p_vol5.to_numpy(), 1e-10))
        feats["parkinson_vol_20d"] = np.sqrt(np.maximum(p_vol20.to_numpy(), 1e-10))

        # 4. Garman-Klass Volatility (5d)
        gk_comp = 0.5 * (np.log(h / low_p)) ** 2 - (2.0 * np.log(2.0) - 1.0) * (np.log(c / o)) ** 2
        gk_vol5 = pd.Series(gk_comp, index=df.index).rolling(5, min_periods=5).mean()
        feats["garman_klass_vol_5d"] = np.sqrt(np.maximum(gk_vol5.to_numpy(), 1e-10))

        # 5. Volume metrics
        log_vol = np.log(np.maximum(v, 1.0))
        d_vol = np.zeros_like(log_vol)
        d_vol[1:] = log_vol[1:] - log_vol[:-1]
        feats["log_volume_change_1d"] = d_vol

        v_ma20 = pd.Series(v, index=df.index).rolling(20, min_periods=20).mean().to_numpy()
        feats["volume_ratio_20d"] = v / np.maximum(v_ma20, 1.0)

        # 6. Multi-horizon momentum
        for window in [3, 5, 10, 20]:
            mom = np.zeros_like(c)
            mom[window:] = np.log(c[window:] / c[:-window])
            feats[f"momentum_{window}d"] = mom

        # 7. Drawdown (20d peak)
        rolling_max20 = pd.Series(c, index=df.index).rolling(20, min_periods=20).max().to_numpy()
        feats["drawdown_20d"] = (c - rolling_max20) / np.maximum(rolling_max20, 1e-8)

        # 8. Shape ratios
        feats["close_to_high_ratio"] = c / np.maximum(h, 1e-8)
        feats["close_to_low_ratio"] = c / np.maximum(low_p, 1e-8)
        hl_spread = (h - low_p) / c
        hl_ma5 = pd.Series(hl_spread, index=df.index).rolling(5, min_periods=5).mean().to_numpy()
        feats["hl_spread_ma_ratio_5d"] = hl_spread / np.maximum(hl_ma5, 1e-8)

        # Drop warmup rows (first 20 rows)
        clean_feats = feats.iloc[20:].copy()
        if clean_feats.isna().any().any():
            raise ValueError("NaN detected in generated stationary features after warmup.")
        return clean_feats[list(STATIONARY_FEATURE_COLUMNS_V1)]

    @staticmethod
    def validate_panel_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, CausalDatasetMetadata]:
        """Validate panel dataframe adhering to strict canonical sorting and schemas."""
        required_cols = ["Date", "SecurityID"] + list(STATIONARY_FEATURE_COLUMNS_V1)
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required panel column: {col}")

        # Check SecurityID syntax
        for sec in df["SecurityID"].unique():
            if not SECURITY_ID_REGEX.match(str(sec)):
                raise ValueError(
                    f"SecurityID '{sec}' does not match contract format SEC_<TICKER>_<ID>."
                )

        # Check duplicates
        dups = df.duplicated(subset=["SecurityID", "Date"])
        if dups.any():
            dup_example = df[dups].iloc[0]
            raise ValueError(
                f"Duplicate (SecurityID, Date) found: ({dup_example['SecurityID']}, {dup_example['Date']})"
            )

        # Canonical sort
        sorted_df = df.sort_values(by=["SecurityID", "Date"]).reset_index(drop=True)

        # Validate finite numeric values
        for feat in STATIONARY_FEATURE_COLUMNS_V1:
            vals = sorted_df[feat].to_numpy(dtype=float)
            if not np.isfinite(vals).all():
                raise ValueError(f"Non-finite values detected in feature column {feat}.")

        # Checksum
        hasher = hashlib.sha256()
        hasher.update("|".join(STATIONARY_FEATURE_COLUMNS_V1).encode("utf-8"))
        for col in STATIONARY_FEATURE_COLUMNS_V1:
            hasher.update(sorted_df[col].to_numpy(dtype=float).tobytes())
        snapshot_hash = hasher.hexdigest()

        meta = CausalDatasetMetadata(
            snapshot_hash=snapshot_hash,
            feature_names=list(STATIONARY_FEATURE_COLUMNS_V1),
            security_count=len(sorted_df["SecurityID"].unique()),
            session_count=len(sorted_df["Date"].unique()),
            start_session=str(sorted_df["Date"].min()),
            end_session=str(sorted_df["Date"].max()),
        )
        return sorted_df, meta
