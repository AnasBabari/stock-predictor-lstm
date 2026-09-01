"""Tests for feature engineering, indicators, cyclical calendar features, and schema validation."""

import numpy as np
import pandas as pd
import pytest

from config import FEATURES
from features.calendar import add_calendar_features
from features.pipeline import validate_features
from features.technical import (
    compute_atr,
    compute_bollinger_bands,
    compute_ema,
    compute_macd,
    compute_obv,
    compute_rsi,
    compute_sma,
)


@pytest.fixture
def dummy_price_df():
    dates = pd.date_range(start="2024-01-01", periods=50, freq="D")
    prices = np.linspace(100.0, 150.0, 50)
    highs = prices + 2.0
    lows = prices - 2.0
    volumes = np.random.randint(1000, 5000, size=50)

    df = pd.DataFrame(
        {
            "Open": prices - 0.5,
            "High": highs,
            "Low": lows,
            "Close": prices,
            "Volume": volumes,
        },
        index=dates,
    )
    return df


def test_technical_indicators_shapes(dummy_price_df):
    sma = compute_sma(dummy_price_df, window=10)
    ema = compute_ema(dummy_price_df, window=10)
    rsi = compute_rsi(dummy_price_df, window=14)
    macd = compute_macd(dummy_price_df)
    bb = compute_bollinger_bands(dummy_price_df)
    atr = compute_atr(dummy_price_df)
    obv = compute_obv(dummy_price_df)

    assert len(sma) == len(dummy_price_df)
    assert len(ema) == len(dummy_price_df)
    assert len(rsi) == len(dummy_price_df)
    assert len(macd) == len(dummy_price_df)
    assert len(bb) == len(dummy_price_df)
    assert len(atr) == len(dummy_price_df)
    assert len(obv) == len(dummy_price_df)

    # Assert RSI bounds
    assert (rsi >= 0.0).all() and (rsi <= 100.0).all()


def test_calendar_cyclical_encoding(dummy_price_df):
    df_cal = add_calendar_features(dummy_price_df)
    assert "Month_Sin" in df_cal.columns
    assert "Month_Cos" in df_cal.columns
    assert "Day_Sin" in df_cal.columns
    assert "Day_Cos" in df_cal.columns

    # Verify Pythagorean identity sin^2 + cos^2 == 1.0
    month_identity = (df_cal["Month_Sin"] ** 2 + df_cal["Month_Cos"] ** 2).values
    day_identity = (df_cal["Day_Sin"] ** 2 + df_cal["Day_Cos"] ** 2).values

    np.testing.assert_allclose(month_identity, 1.0, rtol=1e-5)
    np.testing.assert_allclose(day_identity, 1.0, rtol=1e-5)


def test_validate_features_pass(dummy_price_df):
    # Construct valid mock feature df matching config.FEATURES
    for feat in FEATURES:
        if feat not in dummy_price_df.columns:
            dummy_price_df[feat] = 1.0

    valid_df = dummy_price_df[FEATURES]
    validate_features(valid_df, FEATURES)  # Should not raise


def test_validate_features_failures(dummy_price_df):
    for feat in FEATURES:
        if feat not in dummy_price_df.columns:
            dummy_price_df[feat] = 1.0

    valid_df = dummy_price_df[FEATURES].copy()

    # 1. NaN check
    nan_df = valid_df.copy()
    nan_df.iloc[5, 2] = np.nan
    with pytest.raises(ValueError, match="NaN values detected"):
        validate_features(nan_df, FEATURES)

    # 2. Inf check
    inf_df = valid_df.copy()
    inf_df.iloc[5, 2] = np.inf
    with pytest.raises(ValueError, match="Infinite values detected"):
        validate_features(inf_df, FEATURES)

    # 3. Duplicate columns
    dup_cols_df = pd.concat([valid_df, valid_df[["Close"]]], axis=1)
    with pytest.raises(ValueError, match="Duplicate columns detected"):
        validate_features(dup_cols_df, FEATURES)

    # 4. Column order mismatch
    wrong_order = valid_df[FEATURES[::-1]]
    with pytest.raises(ValueError, match="Feature column order mismatch"):
        validate_features(wrong_order, FEATURES)
