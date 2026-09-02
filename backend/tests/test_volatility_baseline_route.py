from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient

from api import app
from routes import volatility

CLIENT = TestClient(app)


def _snapshot(ticker: str = "MSFT") -> SimpleNamespace:
    return SimpleNamespace(
        ticker=ticker,
        snapshot_id="a" * 64,
        origin_date="2026-08-28",
        origin_close=500.0,
        data_provider="alpaca",
        market_data_cache="miss",
        feature_names=("Return_1D", "Vol_C2C_20"),
        features=np.ones((60, 2), dtype=np.float32),
        causal_har_variance=np.full(6, 0.04, dtype=np.float32),
        baseline_candidates={
            "rolling_c2c_5": np.full(6, 0.05, dtype=np.float64),
            "rolling_c2c_20": np.full(6, 0.04, dtype=np.float64),
            "rolling_c2c_60": np.full(6, 0.03, dtype=np.float64),
            "riskmetrics_ewma_c2c": np.full(6, 0.03, dtype=np.float64),
            "causal_log_har": np.full(6, 0.02, dtype=np.float64),
        },
        historical_dates=("2026-08-27", "2026-08-28"),
        historical_prices=np.array([498.0, 500.0]),
        future_dates=tuple(f"2026-09-{day:02d}" for day in range(1, 31)),
    )


def test_active_route_returns_explicit_causal_baseline(monkeypatch):
    monkeypatch.setattr(volatility, "build_volatility_inference_snapshot", _snapshot)
    response = CLIENT.get(
        "/api/v1/volatility/forecast",
        params={"ticker": "msft", "horizon": 7, "model": "har_rv"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ticker"] == "MSFT"
    assert body["horizon"] == 7
    assert len(body["forecast"]["future_dates"]) == 7
    assert len(body["forecast"]["price_quantiles"]["p50"]) == 7
    assert body["evidence"]["model_status"] == "baseline"
    assert body["evidence"]["metric_source"] == "baseline_definition"
    assert body["evidence"]["news_status"] == "not_used"
    assert body["evidence"]["data_provider"] == "alpaca"
    assert body["evidence"]["interval_method"] == "gaussian_reference_scenario"
    assert body["evidence"]["interval_nominal_coverage"] == 0.90
    assert "not_empirically_calibrated" in body["evidence"]["interval_scope"]
    assert len(body["forecast"]["expected_cumulative_variance_path"]) == 7


def test_baseline_names_map_to_the_documented_windows():
    snapshot = _snapshot()
    persistence = volatility.build_live_volatility_forecast(
        snapshot, horizon=5, model="persistence"
    )
    rolling = volatility.build_live_volatility_forecast(snapshot, horizon=5, model="rolling_mean")
    assert persistence["forecast"]["model"] == "persistence"
    assert persistence["forecast"]["expected_cumulative_variance"] == 0.04
    assert rolling["forecast"]["expected_cumulative_variance"] == 0.03


def test_price_quantiles_use_supplied_cumulative_variance_path():
    snapshot = _snapshot()
    snapshot.baseline_variance_paths = {
        "causal_log_har": np.array([0.01, 0.015, 0.03, 0.07, 0.12, 0.20]),
    }
    forecast = volatility.build_live_volatility_forecast(snapshot, horizon=5, model="har_rv")
    path = forecast["forecast"]["expected_cumulative_variance_path"]
    assert path == [0.01, 0.015, 0.03, 0.07, 0.12]
    assert forecast["forecast"]["price_quantiles"]["p50"] == [500.0] * 5
    assert (
        forecast["forecast"]["price_quantiles"]["p95"][1]
        < forecast["forecast"]["price_quantiles"]["p95"][2]
    )


def test_active_route_rejects_unsupported_horizon_and_model(monkeypatch):
    monkeypatch.setattr(volatility, "build_volatility_inference_snapshot", _snapshot)
    assert CLIENT.get("/api/v1/volatility/forecast?horizon=2").status_code == 400
    assert CLIENT.get("/api/v1/volatility/forecast?model=unknown").status_code == 400


def test_active_route_rejects_invalid_ticker():
    response = CLIENT.get("/api/v1/volatility/forecast?ticker=../model")
    assert response.status_code == 400


def test_models_advertises_train_free_active_contract():
    body = CLIENT.get("/models").json()
    active = body["volatility_forecasting"]
    assert active["status"] == "available"
    assert active["endpoint"] == "/api/v1/volatility/forecast"
    assert active["metric_source"] == "baseline_definition"
    assert body["model_storage"]["required"] is False


def test_volatility_ledger_routes(monkeypatch, tmp_path):
    from services.forecast_ledger import ForecastLedger

    test_db = tmp_path / "route_ledger.db"
    test_ledger = ForecastLedger(test_db)
    monkeypatch.setattr(volatility, "get_forecast_ledger", lambda: test_ledger)
    monkeypatch.setattr(volatility, "build_volatility_inference_snapshot", _snapshot)

    # 1. Forecast logs entry to ledger
    resp = CLIENT.get(
        "/api/v1/volatility/forecast", params={"ticker": "MSFT", "horizon": 7, "model": "auto"}
    )
    assert resp.status_code == 200
    assert resp.json()["forecast"]["model"] == "rolling_mean"

    # 2. Query ledger
    ledger_resp = CLIENT.get("/api/v1/volatility/ledger", params={"ticker": "MSFT", "horizon": 7})
    assert ledger_resp.status_code == 200
    l_body = ledger_resp.json()
    assert l_body["ticker"] == "MSFT"
    assert "live_track_record" in l_body
    assert "replay_track_record" in l_body
    assert len(l_body["entries"]) >= 1
    assert l_body["entries"][0]["ticker"] == "MSFT"
    assert l_body["entries"][0]["status"] == "pending"
    assert l_body["entries"][0]["record_source"] == "live"
    assert l_body["entries"][0]["data_provider"] == "alpaca"


def test_forecast_can_skip_ledger_for_deployment_smoke(monkeypatch):
    monkeypatch.setattr(volatility, "build_volatility_inference_snapshot", _snapshot)
    monkeypatch.setattr(
        volatility,
        "get_forecast_ledger",
        lambda: (_ for _ in ()).throw(AssertionError("ledger must not be opened")),
    )
    response = CLIENT.get(
        "/api/v1/volatility/forecast",
        params={"ticker": "MSFT", "horizon": 7, "record_ledger": "false"},
    )
    assert response.status_code == 200


def test_transport_failure_uses_stable_sanitized_503(monkeypatch):
    from data_pipeline import MarketTransportError

    def fail(_ticker):
        raise MarketTransportError("credential or provider secret")

    monkeypatch.setattr(volatility, "build_volatility_inference_snapshot", fail)
    response = CLIENT.get("/api/v1/volatility/forecast?ticker=MSFT")
    assert response.status_code == 503
    assert response.json() == {
        "error": "MARKET_DATA_UNAVAILABLE",
        "message": "Current market data is temporarily unavailable. Please try again later.",
    }
    assert "secret" not in response.text


def test_unknown_symbol_remains_404(monkeypatch):
    from data_pipeline import UnknownTickerError

    monkeypatch.setattr(
        volatility,
        "build_volatility_inference_snapshot",
        lambda _ticker: (_ for _ in ()).throw(UnknownTickerError("provider detail")),
    )
    response = CLIENT.get("/api/v1/volatility/forecast?ticker=ZZZZ")
    assert response.status_code == 404
