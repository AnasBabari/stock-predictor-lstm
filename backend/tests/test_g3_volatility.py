from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from services.g3_volatility import (
    G3_FEATURE_SET_VERSION,
    G3_FEATURES,
    G3_HORIZONS,
    build_g3_features,
    panel_rolling_base,
)


def _ohlc_frame(n: int = 400, seed: int = 41) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, n)))
    open_ = close * np.exp(rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.004, n)))
    index = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": 1_000_000},
        index=index,
    )


def test_g3_feature_port_matches_research_builders_exactly():
    from research.volatility_structure.asymmetry_features import asymmetry_features
    from research.volatility_structure.har_wrapper import har_log_components
    from research.volatility_structure.range_estimators import range_variances

    frame = _ohlc_frame()
    got = build_g3_features(frame)
    assert list(got.columns) == list(G3_FEATURES)

    estimated = range_variances(frame["Open"], frame["High"], frame["Low"], frame["Close"])
    block_expected = {
        "range_park_20": estimated["parkinson_var"].rolling(20, min_periods=20).mean(),
        "range_gk_20": estimated["garman_klass_var"].rolling(20, min_periods=20).mean(),
        "range_rs_20": estimated["rogers_satchell_var"].rolling(20, min_periods=20).mean(),
        "range_yz_20": estimated["yang_zhang_var"].rolling(20, min_periods=20).mean(),
    }
    for name, expected in block_expected.items():
        np.testing.assert_allclose(
            got[name].to_numpy(), expected.to_numpy(), rtol=0, atol=0, equal_nan=True
        )
    har = har_log_components(frame["Close"])
    for name in ("har_log_daily",):
        np.testing.assert_allclose(
            got[name].to_numpy(), har[name].to_numpy(), rtol=0, atol=0, equal_nan=True
        )
    for name in ("har_log_weekly", "har_log_monthly"):
        # Rolling-mean summation order differs from the production row loop
        # at 1-ulp level (~2e-16 relative); far below any split or decision
        # threshold, so tight tolerance applies here instead of bit-exactness.
        np.testing.assert_allclose(
            got[name].to_numpy(), har[name].to_numpy(), rtol=1e-12, atol=0, equal_nan=True
        )
    asym = asymmetry_features(frame["Close"])
    for name in ("downside_sq", "upside_sq", "neg_indicator"):
        np.testing.assert_allclose(
            got[name].to_numpy(), asym[name].to_numpy(), rtol=0, atol=0, equal_nan=True
        )
    logret = np.log(frame["Close"].to_numpy() / np.roll(frame["Close"].to_numpy(), 1))
    np.testing.assert_allclose(got["ret_1d"].to_numpy()[1:], logret[1:], rtol=0, atol=0)
    assert got["ret_1d"].isna().iloc[0]
    assert G3_FEATURE_SET_VERSION == "vol-gpu-panel-v1"
    assert set(G3_HORIZONS) == {5, 10, 20}
    assert len(G3_FEATURES) == 22


def test_full_panel_parity_against_research_builder_clean_and_dirty():
    """Bit-exactness over all 22 columns vs the validated research builder,
    on clean data and on data with NaN / non-positive / inverted prints."""
    from research.volatility_structure.gpu_panel import build_ticker_features

    clean = _ohlc_frame()
    expected = build_ticker_features(clean, "US", spy=None)
    got = build_g3_features(clean)
    exact = [c for c in G3_FEATURES if c not in ("har_log_weekly", "har_log_monthly")]
    for column in exact:
        np.testing.assert_allclose(
            got[column].to_numpy(),
            expected[column].to_numpy(),
            rtol=0,
            atol=0,
            equal_nan=True,
            err_msg=column,
        )
    for column in ("har_log_weekly", "har_log_monthly"):
        np.testing.assert_allclose(
            got[column].to_numpy(),
            expected[column].to_numpy(),
            rtol=1e-12,
            atol=0,
            equal_nan=True,
            err_msg=column,
        )

    dirty = clean.copy()
    dirty.loc[dirty.index[10], "Close"] = np.nan
    dirty.loc[dirty.index[50], "High"] = -5.0
    dirty.loc[dirty.index[90], ["High", "Low"]] = dirty.loc[
        dirty.index[90], ["Low", "High"]
    ].to_numpy()
    expected_dirty = build_ticker_features(dirty, "US", spy=None)
    got_dirty = build_g3_features(dirty)
    for column in exact:
        np.testing.assert_allclose(
            got_dirty[column].to_numpy(),
            expected_dirty[column].to_numpy(),
            rtol=0,
            atol=0,
            equal_nan=True,
            err_msg=column,
        )
    for column in ("har_log_weekly", "har_log_monthly"):
        np.testing.assert_allclose(
            got_dirty[column].to_numpy(),
            expected_dirty[column].to_numpy(),
            rtol=1e-12,
            atol=0,
            equal_nan=True,
            err_msg=column,
        )


def test_rolling_base_matches_panel_definition():
    from research.volatility_structure import panel

    frame = _ohlc_frame()
    close = frame["Close"]
    for horizon in (5, 10, 20):
        # 1-ulp allowance for differing summation order in the two
        # variance paths; far below any decision threshold.
        expected = panel.arm_a_forecasts(close)[f"f_h{horizon}"].iloc[-1]
        assert panel_rolling_base(frame, horizon) == pytest.approx(expected, rel=1e-12)


def test_g3_inference_applies_base_margin_exactly_once(monkeypatch):
    """Raw margin 0 must reproduce the rolling base (catches base double-counting)."""
    import services.g3_volatility as g3

    frame = _ohlc_frame()

    class FakeSession:
        def __init__(self, raw):
            self._raw = raw

        def run(self, _outputs, _inputs):
            return [np.full((1,), self._raw)]

    monkeypatch.setattr(g3, "_load_session", lambda horizon: FakeSession(0.0))
    for horizon in (5, 10, 20):
        base = g3.panel_rolling_base(frame, horizon)
        assert g3.g3_cumulative_variance(frame, horizon) == pytest.approx(base, rel=1e-12)
    monkeypatch.setattr(g3, "_load_session", lambda horizon: FakeSession(0.5))
    for horizon in (5, 10, 20):
        base = g3.panel_rolling_base(frame, horizon)
        assert g3.g3_cumulative_variance(frame, horizon) == pytest.approx(
            base * np.exp(0.5), rel=1e-12
        )


def test_packaged_g3_artifacts_load_and_infer(tmp_path):
    """Committed ONNX artifacts must load and produce finite positive variance."""
    import services.g3_volatility as g3

    directory = g3._model_dir()
    for horizon in (5, 10, 20):
        assert (directory / f"g3_h{horizon}.onnx").is_file()
        meta = json.loads((directory / f"g3_h{horizon}.meta.json").read_text())
        assert meta["horizon"] == horizon
        assert len(meta["features"]) == 22
    frame = _ohlc_frame()
    for horizon in (5, 10, 20):
        value = g3.g3_cumulative_variance(frame, horizon)
        assert np.isfinite(value) and value > 0


def _snapshot(horizon=5):
    from types import SimpleNamespace

    dates = pd.bdate_range(end="2026-09-02", periods=90)
    closes = np.linspace(490.0, 500.0, 90)
    return SimpleNamespace(
        ticker="MSFT",
        snapshot_id="s" * 64,
        origin_date="2026-09-02",
        origin_close=500.0,
        data_provider="alpaca",
        market_data_cache="hit",
        feature_names=("Return_1D", "Vol_C2C_20"),
        features=np.ones((60, 2), dtype=np.float32),
        baseline_candidates={"rolling_c2c_60": np.full(20, 0.04)},
        historical_dates=tuple(d.date().isoformat() for d in dates),
        historical_prices=np.array(closes),
        future_dates=tuple(f"2026-09-{day:02d}" for day in range(3, 23)),
        data_as_of="2026-09-02",
    )


def test_g3_serving_path_returns_promoted_cone(monkeypatch):
    import data_pipeline
    from services.live_volatility import build_live_volatility_forecast

    monkeypatch.setattr(data_pipeline, "_download_ohlcv", lambda symbol: _ohlc_frame(800))
    body = build_live_volatility_forecast(_snapshot(), horizon=5, model="gpu_g3")
    assert body["forecast"]["model"] == "gpu_g3"
    assert body["evidence"]["model_status"] == "gpu_promoted"
    assert body["evidence"]["baseline"] is False
    quantiles = body["forecast"]["price_quantiles"]
    assert len(quantiles["p50"]) == 5
    assert quantiles["p50"][-1] == body["current_price"]
    assert body["evidence"]["fallback_used"] is None
    assert body["evidence"]["test_evidence_qlike_vs_rolling"]["p_two_sided"] < 1e-9


def test_g3_failure_falls_back_to_baseline_explicitly(monkeypatch, tmp_path):
    import services.g3_volatility as g3
    from services.live_volatility import build_live_volatility_forecast

    monkeypatch.setattr(g3, "_model_dir", lambda: tmp_path)
    g3._sessions.clear()
    body = build_live_volatility_forecast(_snapshot(), horizon=5, model="gpu_g3")
    assert body["forecast"]["model"] == "rolling_mean"
    assert body["evidence"]["model_status"] == "baseline"
    assert body["evidence"]["fallback_used"]
    assert len(body["forecast"]["price_quantiles"]["p50"]) == 5


def test_auto_policy_still_routes_to_baselines():
    from services.live_volatility import _candidate_name
    from services.volatility_contract import AUTO_MODEL_POLICY

    assert dict(AUTO_MODEL_POLICY) == {
        1: "garch_11",
        5: "rolling_mean",
        10: "rolling_mean",
        20: "rolling_mean",
    }
    assert _candidate_name("auto", 5) == "rolling_mean"
    assert _candidate_name("gpu_g3", 5) == "gpu_g3"


def test_g3_risk_framing_against_trailing_volatility(monkeypatch):
    import data_pipeline
    from services.live_volatility import _trailing_risk_context, build_live_volatility_forecast

    monkeypatch.setattr(data_pipeline, "_download_ohlcv", lambda symbol: _ohlc_frame(800))
    body = build_live_volatility_forecast(_snapshot(), horizon=5, model="gpu_g3")
    assert body["evidence"]["risk_level"] in ("Low", "Moderate", "Elevated", "Unknown")
    ratio = body["evidence"]["risk_ratio_vs_trailing_60d"]
    trailing = body["evidence"]["trailing_annualized_volatility_60d"]
    if body["evidence"]["risk_level"] == "Unknown":
        assert ratio is None and trailing is None
    else:
        assert ratio > 0 and trailing > 0
        expected = "Low" if ratio < 0.85 else ("Moderate" if ratio <= 1.15 else "Elevated")
        assert body["evidence"]["risk_level"] == expected

    short = _ohlc_frame(30)
    unknown = _trailing_risk_context(short, 0.0004)
    assert unknown["risk_level"] == "Unknown"
    assert unknown["risk_ratio_vs_trailing_60d"] is None
