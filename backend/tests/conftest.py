# backend/tests/conftest.py
import numpy as np
import pandas as pd
import pytest

from config import FEATURES


@pytest.fixture
def synthetic_feature_df():
    rng = np.random.default_rng(42)
    n_days = 500
    dates = pd.date_range(start="2024-01-01", periods=n_days, freq="D")
    prices = (100 + rng.normal(0, 1, n_days).cumsum()).clip(1)

    df = pd.DataFrame(index=dates)
    df["Open"] = prices - 0.5
    df["High"] = prices + 1.0
    df["Low"] = prices - 1.0
    df["Close"] = prices
    df["Volume"] = rng.integers(1000, 10000, size=n_days)

    # Populate remaining FEATURES with synthetic numeric values
    for feat in FEATURES:
        if feat not in df.columns:
            df[feat] = rng.uniform(0.1, 10.0, size=n_days)

    return df[FEATURES]


@pytest.fixture
def large_synthetic_feature_df():
    """2000-row fixture for walk-forward tests that need enough data across 5 folds."""
    rng = np.random.default_rng(99)
    n_days = 2000
    dates = pd.date_range(start="2018-01-01", periods=n_days, freq="D")
    prices = (100 + rng.normal(0, 1, n_days).cumsum()).clip(1)

    df = pd.DataFrame(index=dates)
    df["Open"] = prices - 0.5
    df["High"] = prices + 1.0
    df["Low"] = prices - 1.0
    df["Close"] = prices
    df["Volume"] = rng.integers(1000, 10000, size=n_days)

    for feat in FEATURES:
        if feat not in df.columns:
            df[feat] = rng.uniform(0.1, 10.0, size=n_days)

    return df[FEATURES]


@pytest.fixture
def preprocessed(synthetic_feature_df):
    from data_pipeline import preprocess

    # Phase 3: preprocess now returns 7 items (adds train_dates, test_dates)
    X_train, X_test, y_train, y_test, scaler, _train_dates, _test_dates = preprocess(
        synthetic_feature_df
    )
    # Return 5-tuple for backward compatibility with existing tests
    return X_train, X_test, y_train, y_test, scaler


@pytest.fixture
def preprocessed_with_dates(synthetic_feature_df):
    from data_pipeline import preprocess

    return preprocess(synthetic_feature_df)


@pytest.fixture
def trained_model(preprocessed):
    from model import train_model

    X_train, X_test, y_train, y_test, scaler = preprocessed
    model, _ = train_model(X_train, y_train, X_test, y_test, ticker="TEST", scaler=scaler)
    return model, preprocessed
