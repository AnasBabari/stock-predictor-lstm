# backend/tests/test_data_pipeline.py
from config import FEATURES, MAX_FORECAST_DAYS, TRAIN_SPLIT, WINDOW_SIZE


def test_preprocess_X_shape(preprocessed):
    X_train, X_test, y_train, y_test, scaler = preprocessed
    assert X_train.ndim == 3
    assert X_train.shape[1] == WINDOW_SIZE
    assert X_train.shape[2] == len(FEATURES)


def test_preprocess_y_shape(preprocessed):
    _, _, y_train, y_test, _ = preprocessed
    assert y_train.ndim == 2
    assert y_train.shape[1] == MAX_FORECAST_DAYS


def test_train_test_split_ratio(preprocessed):
    X_train, X_test, *_ = preprocessed
    total = len(X_train) + len(X_test)
    assert abs(len(X_train) / total - 0.80) < 0.05


def test_scaler_fit_on_train_only(synthetic_feature_df):
    """Scaler max must not exceed the training-partition max across all features."""
    from data_pipeline import preprocess

    X_train, _, y_train, _, scaler, _, _ = preprocess(synthetic_feature_df)
    n_samples = len(synthetic_feature_df) - WINDOW_SIZE - MAX_FORECAST_DAYS + 1
    split = int(n_samples * TRAIN_SPLIT)
    train_values = synthetic_feature_df[FEATURES].values[: split + WINDOW_SIZE]
    train_max = train_values.max(axis=0)

    for col_idx in range(len(FEATURES)):
        assert scaler.data_max_[col_idx] <= train_max[col_idx] + 1e-5


def test_scaler_values_in_01(preprocessed):
    X_train, X_test, _, _, _ = preprocessed
    assert X_train.min() >= -0.01 and X_train.max() <= 1.01


def test_fetch_data_bad_ticker_raises():
    import pytest

    from data_pipeline import fetch_data

    with pytest.raises(ValueError, match="Not enough historical data"):
        fetch_data("ZZZZZZZZZ_FAKE")


def test_preprocess_insufficient_data():
    import pandas as pd
    import pytest

    from data_pipeline import preprocess

    tiny_df = pd.DataFrame(columns=FEATURES)
    with pytest.raises(ValueError):
        preprocess(tiny_df)


def test_preprocess_returns_dates(synthetic_feature_df):
    """Phase 3: preprocess returns train_dates and test_dates alongside arrays."""
    from data_pipeline import preprocess

    result = preprocess(synthetic_feature_df)
    assert len(result) == 7, (
        "preprocess must return 7 items: X_train, X_test, y_train, y_test, scaler, train_dates, test_dates"
    )
    X_train, X_test, y_train, y_test, scaler, train_dates, test_dates = result
    assert len(train_dates) == len(X_train)
    assert len(test_dates) == len(X_test)
    # dates should be ISO date strings
    assert isinstance(train_dates[0], str)
    assert "-" in train_dates[0]


def test_create_sequences_shape(synthetic_feature_df):
    """create_sequences must produce correct 3D array shapes and matching dates."""
    from sklearn.preprocessing import MinMaxScaler

    from data_pipeline import create_sequences

    feature_values = synthetic_feature_df[FEATURES].values
    close_idx = FEATURES.index("Close")
    scaler = MinMaxScaler()
    scaler.fit(feature_values)
    scaled = scaler.transform(feature_values)

    X, y, seq_dates = create_sequences(
        scaled, synthetic_feature_df.index, close_idx, forecast_days=7
    )

    assert X.ndim == 3
    assert X.shape[1] == WINDOW_SIZE
    assert X.shape[2] == len(FEATURES)
    assert y.shape[1] == 7
    assert len(seq_dates) == len(X)


def test_create_sequences_no_leakage(synthetic_feature_df):
    """Each sequence window must only contain data at or before the sequence date."""
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler

    from data_pipeline import create_sequences

    feature_values = synthetic_feature_df[FEATURES].values
    close_idx = FEATURES.index("Close")
    scaler = MinMaxScaler()
    scaler.fit(feature_values)
    scaled = scaler.transform(feature_values)

    X, y, seq_dates = create_sequences(
        scaled, synthetic_feature_df.index, close_idx, forecast_days=7
    )

    # Verify all dates are valid ISO strings parseable by pandas
    for d in seq_dates[:5]:
        assert pd.Timestamp(d) <= synthetic_feature_df.index[-1]


def test_prepare_return_data_lookahead_bias(synthetic_feature_df):
    """Ensure return scaler is fitted only on training data, preventing look-ahead bias."""
    from data_pipeline import prepare_return_data

    X_train, X_test, y_train, y_test, scaler, train_dates, test_dates = prepare_return_data(
        synthetic_feature_df, forecast_days=MAX_FORECAST_DAYS
    )

    n_samples = len(synthetic_feature_df) - 1 - WINDOW_SIZE - MAX_FORECAST_DAYS + 1
    split = int(n_samples * TRAIN_SPLIT)

    train_values = synthetic_feature_df[FEATURES].values[1 : split + WINDOW_SIZE]
    train_max = train_values.max(axis=0)

    for col_idx in range(len(FEATURES)):
        assert scaler.data_max_[col_idx] <= train_max[col_idx] + 1e-5


def test_prepare_return_data_shapes_and_values(synthetic_feature_df):
    """Test features shape and ensure targets are binary for next N days."""
    import numpy as np

    from data_pipeline import prepare_return_data

    X_train, X_test, y_train, y_test, scaler, train_dates, test_dates = prepare_return_data(
        synthetic_feature_df, forecast_days=MAX_FORECAST_DAYS
    )

    assert X_train.ndim == 3
    assert X_train.shape[1] == WINDOW_SIZE
    assert X_train.shape[2] == len(FEATURES)

    assert y_train.ndim == 2
    assert y_train.shape[1] == MAX_FORECAST_DAYS

    assert set(np.unique(y_train)).issubset({0, 1})
    assert set(np.unique(y_test)).issubset({0, 1})

    # Dates returned correctly
    assert len(train_dates) == len(X_train)
    assert len(test_dates) == len(X_test)


def test_get_raw_feature_arrays(synthetic_feature_df):
    """get_raw_feature_arrays returns unscaled data suitable for walk-forward splitting."""

    from data_pipeline import get_raw_feature_arrays

    feature_values, close_values, log_returns, dates = get_raw_feature_arrays(synthetic_feature_df)

    assert feature_values.shape[0] == len(synthetic_feature_df)
    assert feature_values.shape[1] == len(FEATURES)
    # log_returns has length n-1 (first row dropped)
    assert len(log_returns) == len(close_values) - 1
    assert len(dates) == len(synthetic_feature_df)
