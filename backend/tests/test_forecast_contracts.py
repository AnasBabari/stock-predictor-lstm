"""Unit tests for forecast contracts."""

import math

import numpy as np
from backend.contracts.forecast_contracts import (
    DirectionThreeWayContract,
    FutureRealizedVarianceContract,
    PriceReturnDistributionContract,
)


def test_cumulative_log_returns():
    prices = np.array([100.0, 105.0, 110.0, 108.0, 115.0])
    # h=1
    r_h1 = PriceReturnDistributionContract.compute_cumulative_log_returns(prices, 1)
    assert len(r_h1) == 4
    assert math.isclose(r_h1[0], math.log(105.0 / 100.0))
    assert math.isclose(r_h1[1], math.log(110.0 / 105.0))

    # h=2
    r_h2 = PriceReturnDistributionContract.compute_cumulative_log_returns(prices, 2)
    assert len(r_h2) == 3
    assert math.isclose(r_h2[0], math.log(110.0 / 100.0))


def test_anchored_price_reconstruction():
    p0 = 42.15
    pred_returns = [0.0, math.log(45.0 / 42.15), math.log(40.0 / 42.15)]
    reconstructed = PriceReturnDistributionContract.reconstruct_anchored_prices(p0, pred_returns)
    assert math.isclose(reconstructed[0], 42.15)
    assert math.isclose(reconstructed[1], 45.0)
    assert math.isclose(reconstructed[2], 40.0)


def test_direction_three_way_labels():
    rets = np.array([-0.05, -0.005, 0.0, 0.005, 0.04])
    tau = 0.01  # 1% neutral band
    labels = DirectionThreeWayContract.compute_direction_labels(rets, tau)
    assert np.array_equal(labels, [-1, 0, 0, 0, 1])


def test_future_realized_variance_contract():
    daily_var = np.array([0.0001, 0.0002, 0.00015, 0.0003, 0.00025, 0.00018])
    # h=2 cumulative target
    target_h2 = FutureRealizedVarianceContract.compute_cumulative_target(daily_var, 2)
    # target at t=0 covers v_{t+1} + v_{t+2} = 0.0002 + 0.00015 = 0.00035
    assert math.isclose(target_h2[0], 0.0002 + 0.00015)
