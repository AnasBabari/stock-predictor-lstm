"""Unit tests for HonestEvaluationEngine and statistical tests."""

import numpy as np

from research.volatility_forecasting.honest_metrics_v10 import (
    HonestEvaluationEngine,
)


def test_evaluate_returns_and_anchored_prices():
    y_true = np.array([0.01, -0.02, 0.03, 0.00, 0.015])
    y_pred = np.array([0.008, -0.018, 0.025, 0.002, 0.012])
    p0 = np.full(5, 100.0)

    metrics = HonestEvaluationEngine.evaluate_returns(y_true, y_pred, p0)
    assert metrics.return_mae > 0.0
    assert metrics.price_mae > 0.0
    assert metrics.oos_r2_vs_zero > 0.5


def test_evaluate_volatility_qlike():
    y_true = np.array([0.0004, 0.0006, 0.0005, 0.0008])
    y_pred = np.array([0.00042, 0.00058, 0.00052, 0.00078])
    base = np.array([0.0003, 0.0003, 0.0003, 0.0003])

    vol_metrics = HonestEvaluationEngine.evaluate_volatility_qlike(y_true, y_pred, base)
    assert vol_metrics.qlike > 0.0
    assert vol_metrics.relative_qlike_vs_persistence < 1.0


def test_probabilistic_coverage():
    rng = np.random.default_rng(42)
    y_true = rng.normal(0, 0.02, size=200)

    # Well-calibrated quantiles
    quantiles = {
        5: np.full(200, -0.033),
        10: np.full(200, -0.0256),
        25: np.full(200, -0.0135),
        50: np.full(200, 0.0),
        75: np.full(200, 0.0135),
        90: np.full(200, 0.0256),
        95: np.full(200, 0.033),
    }

    prob_metrics = HonestEvaluationEngine.evaluate_probabilistic_coverage(y_true, quantiles)
    # Empirical 50% coverage should be close to 0.50
    assert 0.40 <= prob_metrics.coverage_50pct <= 0.60
    assert 0.70 <= prob_metrics.coverage_80pct <= 0.90
    assert prob_metrics.crps_score > 0.0


def test_diebold_mariano_and_holm_bonferroni():
    rng = np.random.default_rng(123)
    loss_m1 = rng.normal(0.01, 0.002, size=100)
    loss_base = rng.normal(0.02, 0.002, size=100)

    dm_stat, p_val = HonestEvaluationEngine.diebold_mariano_test(loss_m1, loss_base, horizon=3)
    assert dm_stat < 0.0  # Model 1 has significantly lower loss
    assert p_val < 0.001

    # Holm-Bonferroni correction
    p_vals = [0.001, 0.01, 0.04, 0.20, 0.80]
    sig = HonestEvaluationEngine.holm_bonferroni_correction(p_vals)
    assert sig[0] is True
    assert sig[1] is True
    assert sig[4] is False
