# backend/tests/test_data_pipeline.py
from config import MAX_FORECAST_DAYS, TRAIN_SPLIT, WINDOW_SIZE


def test_preprocess_X_shape(preprocessed, synthetic_prices):
    X_train, X_test, y_train, y_test, scaler = preprocessed
    assert X_train.ndim == 3
    assert X_train.shape[1] == WINDOW_SIZE
    assert X_train.shape[2] == 1


def test_preprocess_y_shape(preprocessed):
    _, _, y_train, y_test, _ = preprocessed
    assert y_train.ndim == 2
    assert y_train.shape[1] == MAX_FORECAST_DAYS


def test_train_test_split_ratio(preprocessed):
    X_train, X_test, *_ = preprocessed
    total = len(X_train) + len(X_test)
    assert abs(len(X_train) / total - 0.80) < 0.05


def test_scaler_fit_on_train_only(preprocessed, synthetic_prices):
    """Scaler max must not exceed the training-partition max."""
    X_train, _, y_train, _, scaler = preprocessed
    n_samples = len(synthetic_prices) - WINDOW_SIZE - MAX_FORECAST_DAYS + 1
    split = int(n_samples * TRAIN_SPLIT)
    train_max = synthetic_prices[: split + WINDOW_SIZE].max()
    assert scaler.data_max_[0] <= train_max + 1e-6  # tolerance for float


def test_scaler_values_in_01(preprocessed):
    X_train, X_test, _, _, _ = preprocessed
    assert X_train.min() >= -0.01 and X_train.max() <= 1.01


def test_fetch_data_bad_ticker_raises():
    import pytest

    from data_pipeline import fetch_data

    with pytest.raises(ValueError, match="Not enough historical data"):
        fetch_data("ZZZZZZZZZ_FAKE")


def test_preprocess_insufficient_data():
    import numpy as np
    import pytest

    from data_pipeline import preprocess

    tiny = np.linspace(1, 10, 5).reshape(-1, 1)
    with pytest.raises(ValueError):
        preprocess(tiny)
