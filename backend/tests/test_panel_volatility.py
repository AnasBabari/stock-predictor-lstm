"""Slice-6 tests: volatility proxies, econometric baselines, QLIKE."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from panel.volatility import (
    cumulative_variance_target,
    ewma_forecast_cumulative,
    ewma_variance,
    fit_garch,
    fit_har,
    garch_forecast_cumulative,
    har_forecast_path,
    log_variance_errors,
    qlike_loss,
    realized_variance_proxies,
    relative_qlike,
)


def make_ohlcv(rows: int = 400, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2021-05-03", periods=rows)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, rows)))
    openp = close * np.exp(rng.normal(0, 0.003, rows))
    high = np.maximum(openp, close) * (1 + np.abs(rng.normal(0, 0.004, rows)))
    low = np.minimum(openp, close) * (1 - np.abs(rng.normal(0, 0.004, rows)))
    return pd.DataFrame(
        {"Open": openp, "High": high, "Low": low, "Close": close, "Volume": np.full(rows, 1e6)},
        index=index,
    )


def test_rogers_satchell_hand_check() -> None:
    df = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [110.0],
            "Low": [95.0],
            "Close": [105.0],
            "Volume": [1.0],
        }
    )
    rs = realized_variance_proxies(df)["RV_RS_Intraday"].iloc[0]
    ho, hc = np.log(110 / 100), np.log(110 / 105)
    lo, lc = np.log(95 / 100), np.log(95 / 105)
    assert rs == pytest.approx(ho * hc + lo * lc, rel=1e-12)


def test_total_proxy_equals_overnight_plus_intraday() -> None:
    proxies = realized_variance_proxies(make_ohlcv(60))
    total = proxies["RV_Overnight"] + proxies["RV_RS_Intraday"]
    np.testing.assert_allclose(proxies["RV_Total"].to_numpy(), total.to_numpy())


def test_all_proxies_non_negative_and_finite() -> None:
    proxies = realized_variance_proxies(make_ohlcv(200))
    for column in proxies.columns:
        values = proxies[column].dropna().to_numpy()
        assert np.isfinite(values).all(), column
        assert (values >= 0).all(), column


def test_cumulative_target_is_strictly_future_and_horizon_long() -> None:
    rv = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    target = cumulative_variance_target(rv, 2)
    # Origin 0 sums rows 1..2 → 0.05.
    assert target.iloc[0] == pytest.approx(0.05)
    assert target.iloc[1] == pytest.approx(0.07)
    # Origins whose horizon runs past the end are NaN, never wrapped.
    assert np.isnan(target.iloc[-1])
    bounded = cumulative_variance_target(rv, 2, origin_index=2)
    assert np.isnan(bounded.iloc[3])


def test_ewma_two_step_recursion() -> None:
    returns = np.array([0.02, 0.04])
    var = ewma_variance(returns, lam=0.5)
    seed = float(np.mean(returns[: min(20, len(returns))] ** 2))
    assert var[0] == pytest.approx(seed)
    # RiskMetrics recursion uses the PREVIOUS return at each step.
    assert var[1] == pytest.approx(0.5 * seed + 0.5 * returns[0] ** 2)


def test_ewma_cumulative_scales_linearly_with_horizon() -> None:
    rng = np.random.default_rng(1)
    returns = rng.normal(0, 0.01, 300)
    h1 = ewma_forecast_cumulative(returns, 1)
    h5 = ewma_forecast_cumulative(returns, 5)
    assert h5 == pytest.approx(5 * h1)


def test_har_recovers_synthetic_linear_structure() -> None:
    rng = np.random.default_rng(4)
    rv_true = np.abs(rng.normal(1e-4, 2e-5, 500)) + 1e-6
    rv_series = pd.Series(rv_true).rolling(1).mean().to_numpy()
    coef = fit_har(rv_series[:450])
    path = har_forecast_path(rv_series, coef, horizon=1)
    valid = path[~np.isnan(path)][-40:]
    # HAR on a near-white series should track the level within an order of magnitude.
    level = np.nanmean(rv_series[-40:])
    assert (valid > 0).all()
    assert (np.abs(valid - level) < 5 * level).mean() > 0.9


def test_garch_recovers_plausible_persistence_and_forecasts_positive() -> None:
    rng = np.random.default_rng(9)
    n = 1500
    omega, alpha, beta = 2e-7, 0.08, 0.88
    var = np.empty(n)
    var[0] = omega / (1 - alpha - beta)
    rets = np.empty(n)
    for i in range(1, n):
        sigma2 = omega + alpha * rets[i - 1] ** 2 + beta * var[i - 1]
        rets[i] = np.sqrt(sigma2) * rng.standard_normal()
        var[i] = sigma2
    params = fit_garch(rets[:1200])
    assert 0.5 < params.persistence < 1.0
    forecast = garch_forecast_cumulative(rets[:1200], params, 5)
    long_run_5 = 5 * params.omega / (1 - params.persistence)
    assert forecast > 0
    assert 0.1 * long_run_5 < forecast < 10 * long_run_5


def test_gjr_penalises_negative_shocks_via_leverage_term() -> None:
    from panel.volatility import GarchParams, _garch_filter

    p_pos = GarchParams(omega=1e-8, alpha=0.05, gamma=0.10, beta=0.80)
    r_after_up = np.array([0.01, 0.0])
    r_after_down = np.array([-0.01, 0.0])
    v_up = _garch_filter(r_after_up, p_pos)[-1]
    v_down = _garch_filter(r_after_down, p_pos)[-1]
    assert v_down > v_up


def test_qlike_zero_for_perfect_and_ranks_correctly() -> None:
    target = np.array([1e-4, 2e-4, 5e-5])
    perfect = target.copy()
    biased = target * 4
    assert qlike_loss(perfect, target) == pytest.approx(0, abs=1e-12)
    assert qlike_loss(biased, target) > qlike_loss(perfect, target)
    assert qlike_loss(target * 0.25, target) > qlike_loss(target, target)


def test_relative_qlike_none_when_baseline_degenerate() -> None:
    assert relative_qlike(0.1, 0.0) is None
    assert relative_qlike(0.1, float("nan")) is None
    assert relative_qlike(0.1, 0.5) == pytest.approx(0.2)


def test_log_variance_errors_finite_and_symmetric_shape() -> None:
    f = np.array([1e-4, 2e-4])
    a = np.array([2e-4, 2e-4])
    out = log_variance_errors(f, a)
    expected_mae = float(np.mean(np.abs(np.log(a / f))))
    assert out["logvar_mae"] == pytest.approx(expected_mae / 1)
