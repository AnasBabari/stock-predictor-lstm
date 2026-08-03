import numpy as np
import pytest

from evaluation.metrics import pinball_loss
from experiments import QuantileForecaster as ExportedQuantileForecaster
from experiments.baselines import QuantileForecaster, quantile_crossing_rate


def test_quantile_forecaster_is_exported_from_experiments():
    assert ExportedQuantileForecaster is QuantileForecaster


def _trend_fixture():
    rng = np.random.default_rng(7)
    count = 120
    lookback = 4
    horizons = 2
    close = 100.0 + np.arange(count) * 0.1 + rng.normal(scale=0.02, size=count)
    windows = []
    targets = []
    for start in range(lookback, count - horizons + 1):
        windows.append(close[start - lookback : start])
        targets.append(close[start : start + horizons])
    features = np.asarray(windows)[:, :, None]
    return features, np.asarray(targets)


def test_pinball_loss_matches_hand_computed_value():
    actual = [1.0, 2.0, 3.0]
    quantile_pred = [1.5, 1.5, 4.0]
    # Errors: -0.5, 0.5, -1.0; tau=0.9 losses: 0.05, 0.45, 0.10 -> mean 0.2.
    assert pinball_loss(actual, quantile_pred, 0.9) == pytest.approx(0.2)


def test_pinball_loss_at_half_tau_equals_half_mae():
    rng = np.random.default_rng(3)
    actual = rng.normal(size=20)
    quantile_pred = rng.normal(size=20)
    mae = float(np.mean(np.abs(actual - quantile_pred)))
    assert pinball_loss(actual, quantile_pred, 0.5) == pytest.approx(mae / 2)


@pytest.mark.parametrize("tau", [0.0, 1.0, -0.1, 1.5, float("nan")])
def test_pinball_loss_rejects_invalid_tau(tau):
    with pytest.raises(ValueError):
        pinball_loss([1.0, 2.0], [1.0, 2.0], tau)


def test_pinball_loss_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        pinball_loss([1.0, 2.0], [1.0], 0.5)


def test_pinball_loss_rejects_non_finite_inputs():
    with pytest.raises(ValueError):
        pinball_loss([1.0, np.nan], [1.0, 2.0], 0.5)
    with pytest.raises(ValueError):
        pinball_loss([1.0, 2.0], [1.0, np.inf], 0.5)


def test_quantile_forecaster_predicts_expected_shape_and_median():
    features, targets = _trend_fixture()
    model = QuantileForecaster(max_iter=200).fit(features, targets)
    assert model.name == "quantile_hist_gradient_boosting"
    predicted = model.predict(features)
    assert predicted.shape == (features.shape[0], targets.shape[1], 3)
    median = model.predict_median(features)
    assert median.shape == targets.shape
    np.testing.assert_allclose(median, predicted[:, :, 1])
    # The median column must track the trend far better than a constant guess.
    median_mae = float(np.mean(np.abs(median - targets)))
    baseline_mae = float(np.mean(np.abs(targets.mean(axis=0, keepdims=True) - targets)))
    assert median_mae < 0.5 * baseline_mae
    pooled_median = median.reshape(-1)
    pooled_targets = targets.reshape(-1)
    assert np.corrcoef(pooled_median, pooled_targets)[0, 1] > 0.99


def test_quantile_forecaster_crossing_rate_is_bounded():
    features, targets = _trend_fixture()
    model = QuantileForecaster(max_iter=200).fit(features, targets)
    rate = quantile_crossing_rate(model.predict(features))
    assert 0.0 <= rate <= 1.0


def test_quantile_crossing_rate_counts_rows_with_inverted_quantiles():
    predicted = np.array(
        [
            [[1.0, 2.0], [1.0, 2.0]],  # ordered
            [[2.0, 1.0], [1.0, 2.0]],  # crossing in first horizon
            [[1.0, 2.0], [3.0, 2.0]],  # crossing in second horizon
            [[1.0, 1.0], [2.0, 2.0]],  # equal is not a crossing
        ]
    )
    assert quantile_crossing_rate(predicted) == pytest.approx(0.5)


def test_quantile_crossing_rate_validates_shape():
    with pytest.raises(ValueError):
        quantile_crossing_rate(np.zeros((3, 2)))
    with pytest.raises(ValueError):
        quantile_crossing_rate(np.zeros((3, 2, 1)))
    with pytest.raises(ValueError):
        quantile_crossing_rate(np.full((2, 2, 2), np.nan))


@pytest.mark.parametrize(
    "quantiles",
    [
        (0.9, 0.5),  # not increasing
        (0.5, 0.5),  # not strictly increasing
        (0.0, 0.5, 0.95),  # boundary included
        (0.05, 0.5, 1.0),  # boundary included
        (0.5,),  # too few
    ],
)
def test_quantile_forecaster_rejects_invalid_quantiles(quantiles):
    with pytest.raises(ValueError):
        QuantileForecaster(quantiles=quantiles)


def test_quantile_forecaster_rejects_unfitted_prediction():
    features, _ = _trend_fixture()
    with pytest.raises(ValueError):
        QuantileForecaster().predict(features)
    with pytest.raises(ValueError):
        QuantileForecaster().predict_median(features)


def test_quantile_forecaster_rejects_misaligned_targets():
    features, targets = _trend_fixture()
    with pytest.raises(ValueError):
        QuantileForecaster().fit(features, targets[:-1])
    with pytest.raises(ValueError):
        QuantileForecaster().fit(features, targets[:, 0])
