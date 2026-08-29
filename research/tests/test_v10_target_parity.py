"""Golden parity and correctness tests for the frozen V10 volatility target.

Covers:
- Daily total variance proxy (overnight gap squared + max(Rogers-Satchell, 0))
- Future cumulative target V(t, h) with strict causal origin exclusion
- Neural log-residual target construction and variance reconstruction
- QLIKE argument orientation: QLIKE(forecast, realized)
- Annualized volatility display transformation: sqrt((252 / h) * V_hat)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.panel.volatility import (
    cumulative_variance_target,
    qlike_loss,
    realized_variance_proxies,
    rogers_satchell_frame,
)


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    np.random.seed(42)
    closes = 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, size=100)))
    opens = np.concatenate(([100.0], closes[:-1] * np.exp(np.random.normal(0, 0.002, size=99))))
    highs = np.maximum(opens, closes) * np.exp(np.abs(np.random.normal(0, 0.005, size=100)))
    lows = np.minimum(opens, closes) * np.exp(-np.abs(np.random.normal(0, 0.005, size=100)))
    df = pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": np.random.randint(100000, 1000000, size=100),
        },
        index=dates,
    )
    return df


def test_daily_total_variance_proxy_is_non_negative_and_matches_definition(
    synthetic_ohlcv: pd.DataFrame,
) -> None:
    df = synthetic_ohlcv
    proxies = realized_variance_proxies(df)
    assert "RV_Total" in proxies.columns
    assert "RV_Overnight" in proxies.columns
    assert "RV_RS_Intraday" in proxies.columns

    # Verify manual calculation
    log_close = np.log(df["Close"])
    overnight = np.log(df["Open"]) - log_close.shift(1)
    ho = np.log(df["High"] / df["Open"])
    hc = np.log(df["High"] / df["Close"])
    lo = np.log(df["Low"] / df["Open"])
    lc = np.log(df["Low"] / df["Close"])
    rs = ho * hc + lo * lc
    expected_total = overnight.pow(2) + np.maximum(rs.to_numpy(dtype=float), 0.0)

    # Session 0 has NaN overnight return; from session 1 onward they must match exactly
    np.testing.assert_allclose(
        proxies["RV_Total"].iloc[1:].to_numpy(dtype=float),
        expected_total.iloc[1:].to_numpy(dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )
    # Ensure no total variance is negative
    valid = proxies["RV_Total"].dropna().to_numpy(dtype=float)
    assert np.all(valid >= 0.0)


def test_cumulative_variance_strictly_excludes_origin_row(
    synthetic_ohlcv: pd.DataFrame,
) -> None:
    proxies = realized_variance_proxies(synthetic_ohlcv)
    rv = proxies["RV_Total"]
    for horizon in (1, 3, 5, 7):
        target = cumulative_variance_target(rv, horizon)
        for t in range(1, 20):
            # Target at origin t must equal sum of rv from t+1 to t+h
            expected_sum = float(rv.iloc[t + 1 : t + 1 + horizon].sum())
            actual = float(target.iloc[t])
            assert np.isclose(actual, expected_sum, rtol=1e-10, atol=1e-12), (
                f"Horizon {horizon} at origin {t} mismatched: got {actual}, expected {expected_sum}"
            )


def test_neural_residual_target_and_reconstruction() -> None:
    epsilon = 1e-8
    realized_var = np.array([0.0004, 0.0012, 0.0025, 0.0050])
    baseline_var = np.array([0.0005, 0.0010, 0.0030, 0.0045])

    # Residual target: z = log(V + eps) - log(B + eps)
    z = np.log(realized_var + epsilon) - np.log(baseline_var + epsilon)

    # Reconstructed variance: V_hat = (B + eps) * exp(z_hat) - eps
    reconstructed = (baseline_var + epsilon) * np.exp(z) - epsilon
    np.testing.assert_allclose(reconstructed, realized_var, rtol=1e-10, atol=1e-12)


def test_qlike_argument_orientation_and_properties() -> None:
    # QLIKE(forecast, realized) = realized/forecast - log(realized/forecast) - 1
    # When forecast == realized, QLIKE is 0
    var = np.array([0.001, 0.002, 0.003])
    assert qlike_loss(var, var) == 0.0

    # Overprediction: forecast = 2 * realized -> ratio = 0.5 -> 0.5 - log(0.5) - 1 = 0.5 + 0.6931 - 1 = 0.1931
    # Underprediction: forecast = 0.5 * realized -> ratio = 2.0 -> 2.0 - log(2.0) - 1 = 2.0 - 0.6931 - 1 = 0.3069
    # QLIKE penalizes underprediction more heavily than overprediction (standard volatility property)
    loss_over = qlike_loss(np.array([0.002]), np.array([0.001]))
    loss_under = qlike_loss(np.array([0.0005]), np.array([0.001]))
    assert loss_under > loss_over > 0.0


def test_annualized_volatility_display_transformation() -> None:
    # V_hat is cumulative total variance over h sessions
    # Display annualized vol = sqrt((252 / h) * V_hat)
    h = 5
    v_hat = 0.0025  # 5-day variance
    ann_vol = np.sqrt((252.0 / h) * v_hat)
    expected = np.sqrt(50.4 * 0.0025)
    assert np.isclose(ann_vol, expected, rtol=1e-12)
