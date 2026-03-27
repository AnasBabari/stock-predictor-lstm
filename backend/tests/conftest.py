# backend/tests/conftest.py
import numpy as np
import pytest


@pytest.fixture
def synthetic_prices():
    rng = np.random.default_rng(42)
    return (100 + rng.normal(0, 1, 800).cumsum()).reshape(-1, 1).clip(1)


@pytest.fixture
def preprocessed(synthetic_prices):
    from data_pipeline import preprocess

    return preprocess(synthetic_prices)


@pytest.fixture
def trained_model(preprocessed):
    from model import train_model

    X_train, X_test, y_train, y_test, scaler = preprocessed
    return train_model(X_train, y_train, X_test, y_test, ticker="TEST"), preprocessed
