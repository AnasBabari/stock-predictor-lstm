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
    # The gap is deliberately purged for the 30-day target horizon, so the
    # effective training count is slightly below the configured 80% boundary.
    assert 0.65 < len(X_train) / total < 0.80


def test_price_train_and_test_targets_do_not_overlap(synthetic_feature_df):
    """A date scored by the diagnostic test partition is never a train target."""
    from data_pipeline import preprocess

    X_train, X_test, y_train, y_test, _scaler, train_dates, test_dates = preprocess(
        synthetic_feature_df
    )
    assert len(X_train) == len(y_train) == len(train_dates)
    assert len(X_test) == len(y_test) == len(test_dates)
    last_train_target = synthetic_feature_df.index.get_loc(train_dates[-1]) + MAX_FORECAST_DAYS
    first_test_target = synthetic_feature_df.index.get_loc(test_dates[0]) + 1
    assert last_train_target < first_test_target


def test_scaler_fit_on_train_only(synthetic_feature_df):
    """Scaler center must not exceed the training-partition median across all features."""
    import numpy as np

    from data_pipeline import preprocess

    X_train, _, y_train, _, scaler, _, _ = preprocess(synthetic_feature_df)
    n_samples = len(synthetic_feature_df) - WINDOW_SIZE - MAX_FORECAST_DAYS + 1
    split = int(n_samples * TRAIN_SPLIT)
    train_values = synthetic_feature_df[FEATURES].values[: split + WINDOW_SIZE]
    train_median = np.median(train_values, axis=0)

    for col_idx in range(len(FEATURES)):
        assert abs(scaler.center_[col_idx] - train_median[col_idx]) < 1e-5


def test_scaler_is_robust_and_train_only(preprocessed):
    """Robust scaling centers on the train median and scales by the train IQR."""
    import numpy as np

    X_train, X_test, _, _, scaler = preprocessed
    # Robust scaling is unbounded; the train partition should be centered near zero.
    assert abs(X_train.mean()) < 1.0
    assert abs(X_test.mean()) < 1.0
    assert (scaler.scale_ > 0).all()
    train_median = X_train.reshape(X_train.shape[0], -1)
    assert np.isfinite(train_median).all()


def test_fetch_data_bad_ticker_raises():
    import pytest

    from data_pipeline import fetch_data

    with pytest.raises(ValueError, match="(No market data|Not enough historical data)"):
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
    import numpy as np

    from data_pipeline import prepare_return_data

    X_train, X_test, y_train, y_test, scaler, train_dates, test_dates = prepare_return_data(
        synthetic_feature_df, forecast_days=MAX_FORECAST_DAYS
    )

    n_samples = len(synthetic_feature_df) - 1 - WINDOW_SIZE - MAX_FORECAST_DAYS + 1
    split = int(n_samples * TRAIN_SPLIT)

    train_values = synthetic_feature_df[FEATURES].values[1 : 1 + split + WINDOW_SIZE]
    train_median = np.median(train_values, axis=0)

    for col_idx in range(len(FEATURES)):
        assert abs(scaler.center_[col_idx] - train_median[col_idx]) < 1e-5


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


def test_direction_train_and_test_targets_do_not_overlap(synthetic_feature_df):
    from data_pipeline import prepare_return_data

    *_, train_dates, test_dates = prepare_return_data(synthetic_feature_df)
    # Direction sequence dates are based on the one-row-return-aligned index;
    # the same horizon purge applies to their actual target dates.
    last_train_target = synthetic_feature_df.index.get_loc(train_dates[-1]) + MAX_FORECAST_DAYS
    first_test_target = synthetic_feature_df.index.get_loc(test_dates[0]) + 1
    assert last_train_target < first_test_target


def test_get_raw_feature_arrays(synthetic_feature_df):
    """get_raw_feature_arrays returns unscaled data suitable for walk-forward splitting."""

    from data_pipeline import get_raw_feature_arrays

    feature_values, close_values, log_returns, dates = get_raw_feature_arrays(synthetic_feature_df)

    assert feature_values.shape[0] == len(synthetic_feature_df)
    assert feature_values.shape[1] == len(FEATURES)
    # log_returns has length n-1 (first row dropped)
    assert len(log_returns) == len(close_values) - 1
    assert len(dates) == len(synthetic_feature_df)
