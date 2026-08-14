"""Contract tests for server-side preprocessing and scaler parity."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import RobustScaler

from features.pipeline import make_feature_scaler


def test_make_feature_scaler_contract() -> None:
    scaler = make_feature_scaler()
    assert isinstance(scaler, RobustScaler)
    assert scaler.quantile_range == (25.0, 75.0)


def test_scaler_handles_outliers_and_constant_columns() -> None:
    scaler = make_feature_scaler()
    # 100 samples, 3 features: normal, extreme outlier, constant
    rng = np.random.default_rng(42)
    data = rng.normal(loc=0.0, scale=1.0, size=(100, 3))
    data[0, 1] = 1e6  # extreme outlier
    data[:, 2] = 5.0  # constant column

    scaler.fit(data)
    transformed = scaler.transform(data)

    assert np.isfinite(transformed).all()
    # Constant column median is subtracted and spread is 0 so center subtraction occurs
    assert np.allclose(transformed[:, 2], 0.0)
    # Outlier does not distort the median center of the feature
    assert abs(scaler.center_[1]) < 1.0


def test_experiment_and_server_model_scaler_consistency() -> None:
    from experiments.runner import _scale_windows

    rng = np.random.default_rng(123)
    train_windows = rng.normal(size=(50, 60, 28))
    val_windows = rng.normal(size=(10, 60, 28))

    scaled_train, scaled_val = _scale_windows(train_windows, val_windows)

    # Reference manual fit using make_feature_scaler
    reference_scaler = make_feature_scaler()
    reference_scaler.fit(train_windows.reshape(-1, 28))
    expected_val = reference_scaler.transform(val_windows.reshape(-1, 28)).reshape(
        val_windows.shape
    )

    np.testing.assert_allclose(scaled_val, expected_val, atol=1e-10)
