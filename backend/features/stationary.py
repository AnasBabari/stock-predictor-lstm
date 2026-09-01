"""Stationary feature transformations for the browser schema v4.

Every price-level feature is replaced by a ratio, a return, or a realized
volatility measure so the input distribution stays meaningful as the stock
price grows. All rolling windows use only past observations (causal).

The input DataFrame must already contain the technical indicator columns
(SMA_20, EMA_20, RSI_14, MACD, MACD_Signal, BB_Upper, BB_Lower, ATR_14, OBV)
and the market return columns (SPY_Return_1D, QQQ_Return_1D, VIX_Return_1D,
TNX_Return_1D) produced by the shared pipeline stages.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STATIONARY_SCHEMA_VERSION = 1


def add_stationary_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert absolute price levels into ratios and returns relative to the close."""
    result = df.copy()
    close = result["Close"]
    prev_close = close.shift(1)

    # Price returns relative to the previous close.
    result["Log_Open_Rel"] = np.log(result["Open"] / prev_close)
    result["Log_High_Rel"] = np.log(result["High"] / prev_close)
    result["Log_Low_Rel"] = np.log(result["Low"] / prev_close)
    result["Return_1D"] = np.log(close / prev_close)
    result["Volume_Log1p_Change"] = np.log1p(result["Volume"]) - np.log1p(result["Volume"].shift(1))

    # Technical indicators expressed as ratios of the current close.
    result["Close_SMA_20"] = close / result["SMA_20"] - 1
    result["Close_EMA_20"] = close / result["EMA_20"] - 1
    result["RSI_14_Centered"] = (result["RSI_14"] - 50) / 50
    result["MACD_Close"] = result["MACD"] / close
    result["MACD_Signal_Close"] = result["MACD_Signal"] / close
    result["BB_Upper_Rel"] = result["BB_Upper"] / close - 1
    result["BB_Lower_Rel"] = close / result["BB_Lower"] - 1
    result["ATR_14_Rel"] = result["ATR_14"] / close
    obv_change = result["OBV"].diff()
    obv_mean = obv_change.rolling(20).mean()
    obv_std = obv_change.rolling(20).std()
    result["OBV_Change_Z"] = (obv_change - obv_mean) / obv_std

    # Multi-day momentum and realized volatility.
    result["Return_5D"] = np.log(close / close.shift(5))
    result["Return_20D"] = np.log(close / close.shift(20))
    result["Realized_Vol_5D"] = result["Return_1D"].rolling(5).std()
    result["Realized_Vol_20D"] = result["Return_1D"].rolling(20).std()

    # Market-relative return and rolling beta to SPY.
    result["Return_Rel_SPY_1D"] = result["Return_1D"] - result["SPY_Return_1D"]
    covariance = result["Return_1D"].rolling(20).cov(result["SPY_Return_1D"])
    market_variance = result["SPY_Return_1D"].rolling(20).var()
    result["Beta_SPY_20D"] = (
        covariance.div(market_variance.where(market_variance.abs() > 1e-12))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    return result
