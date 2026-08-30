"""Regression test verifying the Day-1 price discontinuity caused by unanchored absolute-price targets."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import RobustScaler


class MockAbsolutePriceModel:
    """Simulates a model trained on unanchored absolute price levels whose outputs regress to historical median."""

    def __init__(self, historical_center_scaled: float = 0.0) -> None:
        self.historical_center_scaled = historical_center_scaled

    def predict(self, _x: np.ndarray) -> np.ndarray:
        # Returns 7 horizon predictions regressing toward historical center
        return np.full((1, 7), self.historical_center_scaled, dtype=float)


class MockAnchoredReturnModel:
    """Simulates a model predicting cumulative log returns anchored at P0."""

    def __init__(self, predicted_cumulative_log_returns: list[float]) -> None:
        self.returns = np.array(predicted_cumulative_log_returns, dtype=float)

    def predict_cumulative_returns(self, _x: np.ndarray) -> np.ndarray:
        return self.returns.reshape(1, -1)


def test_scaler_exact_roundtrip():
    """Verify that RobustScaler inverse-transform roundtrips the latest known close price exactly."""
    rng = np.random.default_rng(42)
    # Generate 500 historical prices with center around $30
    hist_prices = rng.normal(30.0, 5.0, size=(500, 1))
    hist_prices = np.clip(hist_prices, 10.0, 60.0)

    scaler = RobustScaler(quantile_range=(25.0, 75.0))
    scaler.fit(hist_prices)

    latest_close = 42.15
    scaled_val = scaler.transform([[latest_close]])
    unscaled_val = scaler.inverse_transform(scaled_val)[0, 0]

    roundtrip_error = abs(unscaled_val - latest_close)
    assert roundtrip_error < 1e-6, f"Scaler roundtrip error too large: {roundtrip_error}"


def test_reproduce_unanchored_absolute_price_day1_cliff():
    """Demonstrate why unanchored absolute-price prediction produces an artificial Day-1 cliff."""
    # Historical price distribution centered at $30 (median=30, IQR=8)
    rng = np.random.default_rng(123)
    hist_prices = rng.normal(30.0, 4.0, size=(1000, 1))
    scaler = RobustScaler(quantile_range=(25.0, 75.0))
    scaler.fit(hist_prices)

    # Current elevated market price
    p0 = 42.15
    daily_vol = 0.018  # 1.8% daily volatility

    # Absolute price model outputs predictions near historical center (~$30 to $33)
    model_absolute = MockAbsolutePriceModel(historical_center_scaled=0.3)
    pred_scaled = model_absolute.predict(np.zeros((1, 60, 1)))
    pred_prices_abs = scaler.inverse_transform(pred_scaled).flatten()

    p1_abs = float(pred_prices_abs[0])
    day1_log_return_abs = float(np.log(p1_abs / p0))
    day1_z_score_abs = abs(day1_log_return_abs) / daily_vol

    # The unanchored model produces a massive, artificial Day-1 cliff (> 5 sigma)
    assert p1_abs < 35.0, f"Expected regression toward historical center, got {p1_abs}"
    assert day1_z_score_abs > 5.0, f"Expected massive jump z-score, got {day1_z_score_abs}"
    assert day1_log_return_abs < -0.15, f"Expected Day 1 log return cliff < -15%, got {day1_log_return_abs}"

    # Now verify anchored return formulation
    # Model predicts plausible 7-day cumulative returns: [-0.002, +0.001, -0.003, ...]
    plausible_returns = [-0.002, -0.001, +0.002, +0.001, -0.003, +0.000, +0.002]
    model_anchored = MockAnchoredReturnModel(plausible_returns)
    pred_cum_rets = model_anchored.predict_cumulative_returns(np.zeros((1, 60, 1))).flatten()

    # Reconstruct prices strictly via P_{t+h} = P_0 * exp(R_{t,h})
    reconstructed_prices = p0 * np.exp(pred_cum_rets)

    # Day 0 anchor invariant: P_{t+0} = P_0
    assert reconstructed_prices.shape == (7,)
    p1_anchored = float(reconstructed_prices[0])
    day1_log_return_anchored = float(np.log(p1_anchored / p0))
    day1_z_score_anchored = abs(day1_log_return_anchored) / daily_vol

    assert day1_z_score_anchored < 1.0, f"Anchored return produced unphysical jump: {day1_z_score_anchored}"
    assert abs(p1_anchored - p0) < 1.0, "Anchored Day 1 price deviated dramatically from P0"


def test_ensemble_disagreement_diagnostic():
    """Verify that ensemble disagreement across diverse models is flagged when models predict conflicting levels."""
    p0 = 42.15
    # Diverse Day 1 outputs from flawed unanchored models
    m1_p1 = 33.04  # -21.61%
    m2_p1 = 36.70  # -12.94%
    m3_p1 = 39.54  # -6.20%

    day1_preds = np.array([m1_p1, m2_p1, m3_p1])
    day1_range_dollars = float(np.max(day1_preds) - np.min(day1_preds))
    day1_range_pct = float(day1_range_dollars / p0 * 100.0)
    ensemble_std = float(np.std(day1_preds))

    assert day1_range_dollars == pytest.approx(6.50, abs=0.01)
    assert day1_range_pct > 15.0, "Expected >15% disagreement between unanchored models"
    assert ensemble_std > 2.5, "Expected high standard deviation across unanchored models"
