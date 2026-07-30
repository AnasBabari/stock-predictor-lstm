import numpy as np

from experiments.baselines import (
    DriftForecaster,
    HistogramGradientBoostingForecaster,
    PersistenceForecaster,
    RidgeForecaster,
)
from experiments.targets import (
    build_supervised_dataset,
    reconstruct_prices,
    transform_price_targets,
)


def test_target_transforms_round_trip_to_original_prices():
    origins = np.array([100.0, 200.0])
    future = np.array([[101.0, 105.0], [198.0, 210.0]])
    for target_type in ("price_level", "simple_return", "log_return", "persistence_residual"):
        targets = transform_price_targets(origins, future, target_type)
        reconstructed = reconstruct_prices(origins, targets, target_type)
        np.testing.assert_allclose(reconstructed, future)


def test_supervised_windows_end_at_origin_and_targets_are_in_the_future():
    close = np.arange(1.0, 21.0)
    features = np.column_stack([close, close * 10])
    dataset = build_supervised_dataset(
        features,
        close,
        lookback=4,
        horizons=(1, 3),
        target_type="persistence_residual",
    )
    assert dataset.origin_indices[0] == 3
    np.testing.assert_array_equal(dataset.features[0, :, 0], [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(dataset.actual_prices[0], [5.0, 7.0])
    np.testing.assert_array_equal(dataset.targets[0], [1.0, 3.0])


def test_persistence_and_drift_use_each_forecast_origin():
    features = np.array(
        [
            [[8.0], [9.0], [10.0]],
            [[18.0], [19.0], [20.0]],
        ]
    )
    origins = np.array([10.0, 20.0])
    persistence = PersistenceForecaster().predict_prices(origins=origins, horizons=(1, 3))
    drift = DriftForecaster(0).predict_prices(features, origins=origins, horizons=(1, 3))
    np.testing.assert_array_equal(persistence, [[10.0, 10.0], [20.0, 20.0]])
    np.testing.assert_array_equal(drift, [[11.0, 13.0], [21.0, 23.0]])


def test_ridge_learns_direct_horizon_targets():
    rng = np.random.default_rng(4)
    features = rng.normal(size=(40, 3, 2))
    flattened = features.reshape(40, -1)
    targets = np.column_stack([flattened[:, 0] * 2, flattened[:, 1] * -3])
    model = RidgeForecaster(alpha=1e-8).fit(features, targets)
    np.testing.assert_allclose(model.predict(features), targets, atol=1e-6)


def test_histogram_gradient_boosting_returns_one_column_per_horizon():
    rng = np.random.default_rng(5)
    features = rng.normal(size=(50, 4, 2))
    targets = np.column_stack([features[:, -1, 0], features[:, -1, 1]])
    model = HistogramGradientBoostingForecaster(max_iter=5).fit(features, targets)
    assert model.predict(features).shape == targets.shape
