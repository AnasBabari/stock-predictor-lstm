import pytest

from services.baselines import base_rate_direction_forecast, persistence_price_forecast


def test_persistence_forecast_repeats_latest_close_and_labels_evidence():
    values, metrics = persistence_price_forecast([100, 101, 103], 3)
    assert values == [103.0, 103.0, 103.0]
    assert metrics["relative_mae"] == metrics["relative_rmse"] == 1.0
    assert metrics["metric_source"] == "baseline_definition"


def test_base_rate_direction_is_smoothed_and_bounded():
    directions, probabilities, metrics = base_rate_direction_forecast([100, 101, 100, 102], 2)
    assert directions == ["Up", "Up"]
    assert 0 < probabilities[0] < 1
    assert probabilities == [probabilities[0]] * 2
    assert metrics["metric_scope"] == "recent_observed_base_rate"


def test_baselines_reject_invalid_inputs():
    with pytest.raises(ValueError):
        persistence_price_forecast([], 1)
    with pytest.raises(ValueError):
        base_rate_direction_forecast([1, 2], 1)
