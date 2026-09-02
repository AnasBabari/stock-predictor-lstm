from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient

from api import app
from routes import volatility
from services.forecast_ledger import LedgerUnavailableError
from services.volatility_snapshot import VOLATILITY_HORIZONS

CLIENT = TestClient(app)


def _snapshot(ticker: str = "MSFT", origin_date: str = "2026-09-02") -> SimpleNamespace:
    return SimpleNamespace(
        ticker=ticker,
        snapshot_id="a" * 64,
        origin_date=origin_date,
        data_as_of=origin_date,
        origin_close=500.0,
        data_provider="alpaca",
        market_data_cache="miss",
        feature_names=("Return_1D", "Vol_C2C_20"),
        features=np.ones((60, 2), dtype=np.float32),
        causal_har_variance=np.full(len(VOLATILITY_HORIZONS), 0.04, dtype=np.float32),
        baseline_candidates={
            "rolling_c2c_5": np.full(len(VOLATILITY_HORIZONS), 0.05, dtype=np.float64),
            "rolling_c2c_20": np.full(len(VOLATILITY_HORIZONS), 0.04, dtype=np.float64),
            "rolling_c2c_60": np.full(len(VOLATILITY_HORIZONS), 0.03, dtype=np.float64),
            "riskmetrics_ewma_c2c": np.full(len(VOLATILITY_HORIZONS), 0.03, dtype=np.float64),
            "causal_log_har": np.full(len(VOLATILITY_HORIZONS), 0.02, dtype=np.float64),
            "garch_11": np.full(len(VOLATILITY_HORIZONS), 0.025, dtype=np.float64),
        },
        historical_dates=("2026-08-27", "2026-08-28"),
        historical_prices=np.array([498.0, 500.0]),
        future_dates=tuple(f"2026-09-{day:02d}" for day in range(1, 21)),
    )


def test_active_route_returns_explicit_causal_baseline(monkeypatch):
    monkeypatch.setattr(volatility, "build_volatility_inference_snapshot", _snapshot)
    response = CLIENT.get(
        "/api/v1/volatility/forecast",
        params={"ticker": "msft", "horizon": 5, "model": "har_rv", "record_ledger": "false"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ticker"] == "MSFT"
    assert body["horizon"] == 5
    assert len(body["forecast"]["future_dates"]) == 5
    assert len(body["forecast"]["price_quantiles"]["p50"]) == 5
    assert body["evidence"]["model_status"] == "baseline"
    assert body["evidence"]["metric_source"] == "baseline_definition"
    assert body["evidence"]["news_status"] == "not_used"
    assert body["evidence"]["data_provider"] == "alpaca"
    assert body["evidence"]["data_as_of"] == "2026-09-02"
    assert body["evidence"]["model_version"] == "deployable_v5"
    assert body["evidence"]["model_policy_version"] == "empirical_volatility_benchmark_v3"
    assert body["evidence"]["code_commit"]
    assert len(body["evidence"]["forecast_fingerprint"]) == 64
    assert body["evidence"]["interval_method"] == "gaussian_reference_scenario"
    assert body["evidence"]["interval_nominal_coverage"] == 0.90
    assert "not_empirically_calibrated" in body["evidence"]["interval_scope"]
    assert len(body["forecast"]["expected_cumulative_variance_path"]) == 5


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
    assert CLIENT.get("/api/v1/volatility/forecast?horizon=7").status_code == 400
    assert CLIENT.get("/api/v1/volatility/forecast?model=unknown").status_code == 400


def test_auto_policy_is_horizon_specific(monkeypatch):
    monkeypatch.setattr(volatility, "build_volatility_inference_snapshot", _snapshot)
    one_day = CLIENT.get("/api/v1/volatility/forecast?horizon=1&record_ledger=false")
    five_day = CLIENT.get("/api/v1/volatility/forecast?horizon=5&record_ledger=false")
    assert one_day.status_code == 200
    assert five_day.status_code == 200
    assert one_day.json()["forecast"]["model"] == "garch_11"
    assert five_day.json()["forecast"]["model"] == "rolling_mean"
    assert one_day.json()["evidence"]["auto_model_policy"] == {
        "1": "garch_11",
        "5": "rolling_mean",
        "10": "rolling_mean",
        "20": "rolling_mean",
    }


def test_active_route_rejects_invalid_ticker():
    response = CLIENT.get("/api/v1/volatility/forecast?ticker=../model")
    assert response.status_code == 400


def test_models_advertises_train_free_active_contract():
    body = CLIENT.get("/models").json()
    active = body["volatility_forecasting"]
    assert active["status"] == "available"
    assert active["endpoint"] == "/api/v1/volatility/forecast"
    assert active["public_forecast_mode"] == "read_only_preview"
    assert active["live_collection_endpoint"] == "/api/v1/volatility/collect"
    assert active["live_collection_authentication"] == "bearer_token_required"
    assert active["metric_source"] == "baseline_definition"
    assert body["model_storage"]["required"] is False


def test_public_forecast_and_ledger_reads_never_write(monkeypatch, tmp_path):
    from services.forecast_ledger import ForecastLedger

    test_db = tmp_path / "route_ledger.db"
    test_ledger = ForecastLedger(test_db)
    monkeypatch.setattr(volatility, "get_forecast_ledger", lambda: test_ledger)
    monkeypatch.setattr(volatility, "build_volatility_inference_snapshot", _snapshot)

    # Public forecasts are previews even when a caller supplies the removed
    # legacy query parameter.
    resp = CLIENT.get(
        "/api/v1/volatility/forecast",
        params={
            "ticker": "MSFT",
            "horizon": 5,
            "model": "auto",
            "record_ledger": "true",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["forecast"]["model"] == "rolling_mean"
    assert resp.json()["evidence"]["ledger_write"] == "disabled_public_preview"

    # 2. Query ledger
    ledger_resp = CLIENT.get("/api/v1/volatility/ledger", params={"ticker": "MSFT", "horizon": 5})
    assert ledger_resp.status_code == 200
    l_body = ledger_resp.json()
    assert l_body["ticker"] == "MSFT"
    assert "live_track_record" in l_body
    assert "replay_track_record" in l_body
    assert l_body["entries"] == []
    assert test_ledger.export_records(record_source="live") == []


def test_public_forecast_never_opens_ledger(monkeypatch):
    monkeypatch.setattr(volatility, "build_volatility_inference_snapshot", _snapshot)
    monkeypatch.setattr(
        volatility,
        "get_forecast_ledger",
        lambda: (_ for _ in ()).throw(AssertionError("ledger must not be opened")),
    )
    response = CLIENT.get(
        "/api/v1/volatility/forecast",
        params={"ticker": "MSFT", "horizon": 5, "record_ledger": "true"},
    )
    assert response.status_code == 200
    assert response.json()["evidence"]["ledger_write"] == "disabled_public_preview"


def test_collector_requires_configured_authentication(monkeypatch):
    monkeypatch.delenv("FORECAST_COLLECTOR_TOKEN", raising=False)
    response = CLIENT.post("/api/v1/volatility/collect", params={"ticker": "MSFT", "horizon": 5})
    assert response.status_code == 503
    assert response.json() == {"detail": "Collector authentication is unavailable."}


def test_unauthenticated_collection_and_settlement_are_rejected(monkeypatch):
    monkeypatch.setenv("FORECAST_COLLECTOR_TOKEN", "test-collector-secret")
    collection = CLIENT.post("/api/v1/volatility/collect", params={"ticker": "MSFT", "horizon": 5})
    settlement = CLIENT.post("/api/v1/volatility/score-ledger?ticker=MSFT")
    assert collection.status_code == 401
    assert settlement.status_code == 401
    assert "test-collector-secret" not in collection.text + settlement.text


def test_authenticated_collection_is_idempotent(monkeypatch, tmp_path):
    from services.forecast_ledger import ForecastLedger

    ledger = ForecastLedger(tmp_path / "collector.db")
    monkeypatch.setenv("FORECAST_COLLECTOR_TOKEN", "test-collector-secret")
    monkeypatch.setattr(volatility, "get_forecast_ledger", lambda: ledger)
    monkeypatch.setattr(volatility, "build_volatility_inference_snapshot", _snapshot)
    headers = {"Authorization": "Bearer test-collector-secret"}

    first = CLIENT.post(
        "/api/v1/volatility/collect",
        params={"ticker": "MSFT", "horizon": 5},
        headers=headers,
    )
    second = CLIENT.post(
        "/api/v1/volatility/collect",
        params={"ticker": "MSFT", "horizon": 5},
        headers=headers,
    )

    assert first.status_code == second.status_code == 200
    assert (
        first.json()["evidence"]["forecast_fingerprint"]
        == second.json()["evidence"]["forecast_fingerprint"]
    )
    assert first.json()["evidence"]["ledger_write"] == "recorded_live"
    assert len(ledger.export_records(record_source="live")) == 1


def test_collection_rejects_out_of_contract_item(monkeypatch):
    monkeypatch.setenv("FORECAST_COLLECTOR_TOKEN", "test-collector-secret")
    response = CLIENT.post(
        "/api/v1/volatility/collect",
        params={"ticker": "TSLA", "horizon": 5},
        headers={"Authorization": "Bearer test-collector-secret"},
    )
    assert response.status_code == 400
    assert "frozen live universe" in response.json()["detail"]


def test_collection_enforces_live_start_date(monkeypatch):
    monkeypatch.setenv("FORECAST_COLLECTOR_TOKEN", "test-collector-secret")
    monkeypatch.setattr(
        volatility,
        "build_volatility_inference_snapshot",
        lambda ticker: _snapshot(ticker, "2026-09-01"),
    )
    response = CLIENT.post(
        "/api/v1/volatility/collect",
        params={"ticker": "MSFT", "horizon": 5},
        headers={"Authorization": "Bearer test-collector-secret"},
    )
    assert response.status_code == 409
    assert response.json()["error"] == "LIVE_COLLECTION_NOT_ELIGIBLE"


def test_collector_rejects_recording_when_ledger_is_unavailable(monkeypatch):
    monkeypatch.setenv("FORECAST_COLLECTOR_TOKEN", "test-collector-secret")
    monkeypatch.setattr(volatility, "build_volatility_inference_snapshot", _snapshot)
    monkeypatch.setattr(
        volatility,
        "get_forecast_ledger",
        lambda: (_ for _ in ()).throw(LedgerUnavailableError("database offline")),
    )

    response = CLIENT.post(
        "/api/v1/volatility/collect",
        params={"ticker": "MSFT", "horizon": 5},
        headers={"Authorization": "Bearer test-collector-secret"},
    )

    assert response.status_code == 503
    assert response.json() == {
        "error": "FORECAST_LEDGER_UNAVAILABLE",
        "message": "The forecast was not recorded because the forecast ledger is temporarily unavailable.",
    }


def test_authenticated_export_contains_live_records_only(monkeypatch, tmp_path):
    from services.forecast_ledger import ForecastLedger

    ledger = ForecastLedger(tmp_path / "export.db")
    common = {
        "forecast_date": "2026-09-02",
        "ticker": "MSFT",
        "horizon": 5,
        "target_date": "2026-09-10",
        "model_name": "rolling_mean",
        "predicted_volatility": 0.25,
        "recent_realized_volatility": 0.2,
        "origin_price": 500.0,
        "lower_scenario_price": 475.0,
        "upper_scenario_price": 525.0,
        "data_as_of": "2026-09-02",
        "data_provider": "alpaca",
    }
    ledger.record_forecast(**common, record_source="live")
    ledger.record_forecast(**common, record_source="historical_replay")
    monkeypatch.setenv("FORECAST_COLLECTOR_TOKEN", "test-collector-secret")
    monkeypatch.setattr(volatility, "get_forecast_ledger", lambda: ledger)

    response = CLIENT.get(
        "/api/v1/volatility/export-ledger",
        headers={"Authorization": "Bearer test-collector-secret"},
    )
    assert response.status_code == 200
    assert response.json()["record_source"] == "live"
    assert len(response.json()["entries"]) == 1
    assert response.json()["entries"][0]["record_source"] == "live"


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
