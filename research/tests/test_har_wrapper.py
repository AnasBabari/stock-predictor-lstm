import numpy as np
import pandas as pd
import pytest
from services.volatility_snapshot import (
    _log_har_row,
    causal_log_har_forecasts,
    realized_variance_proxies,
)

from research.volatility_structure.har_wrapper import (
    HAR_HORIZONS,
    daily_proxy,
    har_forecasts,
    har_log_components,
)


def _gbm_closes(n=400, seed=5):
    rng = np.random.default_rng(seed)
    return pd.Series(
        100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, n))),
        index=pd.bdate_range("2022-01-03", periods=n),
    )


def test_proxy_matches_production_definition():
    closes = _gbm_closes()
    expected = realized_variance_proxies(pd.DataFrame({"Close": closes}))["RV_C2C"]
    pd.testing.assert_series_equal(daily_proxy(closes), expected, check_names=False)


def test_forecasts_match_production_engine_exactly():
    closes = _gbm_closes()
    got = har_forecasts(closes)
    assert list(got.columns) == [f"har_h{h}" for h in HAR_HORIZONS]
    direct = causal_log_har_forecasts(daily_proxy(closes), list(HAR_HORIZONS))
    np.testing.assert_array_equal(got.to_numpy(), direct)
    # Warm-up: NaN before origin 60, finite from 60 on (38 fitted rows clear
    # the 20-row gate exactly at the boundary).
    assert got.iloc[:60].isna().all().all()
    assert np.isfinite(got.iloc[60:].to_numpy()).all()


def test_components_match_production_row_constructor():
    closes = _gbm_closes()
    comps = har_log_components(closes)
    assert list(comps.columns) == ["har_log_daily", "har_log_weekly", "har_log_monthly"]
    assert comps.iloc[:21].isna().all().all()
    assert np.isfinite(comps.iloc[21:].to_numpy()).all()
    proxy = daily_proxy(closes).to_numpy()
    _, daily, weekly, monthly = _log_har_row(np.maximum(proxy[79:101], 1e-12))
    assert comps["har_log_daily"].iloc[100] == pytest.approx(daily)
    assert comps["har_log_weekly"].iloc[100] == pytest.approx(weekly)
    assert comps["har_log_monthly"].iloc[100] == pytest.approx(monthly)


def test_empty_input_raises():
    with pytest.raises(ValueError, match="empty"):
        daily_proxy(pd.Series([], dtype=float))
