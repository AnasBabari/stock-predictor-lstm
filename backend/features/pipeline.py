"""Feature pipeline orchestrator and validator."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler  # type: ignore[import-untyped]

from features.calendar import add_calendar_features
from features.market import add_market_context
from features.stationary import add_stationary_features
from features.technical import add_technical_indicators


def make_feature_scaler() -> RobustScaler:
    """Authoritative scaler contract for server-side feature preprocessing."""
    return RobustScaler(quantile_range=(25.0, 75.0))


def validate_features(df: pd.DataFrame, expected_features: list[str]) -> None:
    """
    Assert data quality requirements before training or inference:
    - No NaNs or Infinities
    - Exact matching column set and column order
    - No duplicate column names or indices
    """
    if df.columns.duplicated().any():
        raise ValueError("Duplicate columns detected in feature DataFrame.")

    if df.index.duplicated().any():
        raise ValueError("Duplicate index timestamps detected in feature DataFrame.")

    missing_cols = set(expected_features) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing expected features: {missing_cols}")

    # Verify column order
    current_features = [col for col in df.columns if col in expected_features]
    if current_features != expected_features:
        raise ValueError(
            f"Feature column order mismatch. Expected {expected_features}, got {current_features}"
        )

    # Check for NaNs or Infinities in expected features
    feature_df = df[expected_features]
    if feature_df.isna().any().any():
        raise ValueError("NaN values detected in feature matrix after preprocessing.")

    if np.isinf(feature_df.values).any():
        raise ValueError("Infinite values detected in feature matrix after preprocessing.")


def build_features(
    raw_df: pd.DataFrame, expected_features: list[str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Build technical, market, and calendar features, validate schema, and clean NaNs.
    """
    df = raw_df.copy()

    # Apply feature engineering stages
    df = add_technical_indicators(df)
    df, market_metadata = add_market_context(df)
    df = add_calendar_features(df)

    # Reorder columns explicitly to match expected_features schema
    df = df[expected_features]

    # Drop initial NaN rows created by rolling windows (e.g., SMA_20, MACD, ATR_14)
    df = df.dropna()

    # Validate output feature matrix
    validate_features(df, expected_features)

    metadata = {
        "feature_names": expected_features,
        "feature_count": len(expected_features),
        "date_range": f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}",
        "market_context": market_metadata,
        "rows": int(len(df)),
    }

    return df, metadata


def build_browser_features(
    raw_df: pd.DataFrame, expected_features: list[str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Build the stationary schema-v4 feature matrix used by browser training.

    The raw OHLCV frame passes through the technical, market, and stationary
    transformation stages in order, so the ratio features always reference a
    causal previous close and rolling windows never see future observations.
    """
    df = raw_df.copy()

    # Apply feature engineering stages in dependency order.
    df = add_technical_indicators(df)
    df, market_metadata = add_market_context(df)
    df = add_stationary_features(df)
    df = add_calendar_features(df)

    # Reorder columns explicitly to match expected_features schema.
    df = df[expected_features]

    # Drop initial NaN rows created by causal rolling windows (SMA_20,
    # ATR_14, MACD, Return_20D, Realized_Vol_20D, Beta_SPY_20D, OBV z-score).
    df = df.dropna()

    validate_features(df, expected_features)

    metadata = {
        "feature_names": expected_features,
        "feature_count": len(expected_features),
        "schema_version": 4,
        "feature_schema": "stationary_v4",
        "date_range": f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}",
        "market_context": market_metadata,
        "rows": int(len(df)),
    }

    return df, metadata
