import asyncio
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import api
from api import WorkCoordinator, app
from config import FEATURES, FEATURES_V4, MAX_FORECAST_DAYS, WINDOW_SIZE
from server_models.response_models import (
    ARTIFACT_ACTIONS,
    ARTIFACT_STATES,
    EXECUTION_MODES,
    TIMING_FIELDS,
)

client = TestClient(app)


def _response(ticker="AAPL", days=7, direction=False):
    base = {
        "ticker": ticker,
        "forecast_days": days,
        "future_dates": [f"2026-08-{day + 1:02d}" for day in range(days)],
        "metrics": {"metric_source": "walk_forward_out_of_fold"},
        "metadata": {"calendar": "NYSE"},
    }
    if direction:
        return {
            **base,
            "directions": ["Up"] * days,
            "probabilities": [0.6] * days,
            "attention_weights": [],
            "sentiment": {"score": 0.0, "status": "fallback"},
        }
    return {
        **base,
        "historical_dates": ["2026-07-01"],
        "historical_prices": [100.0],
        "predicted_prices": [101.0] * days,
    }


def _feature_snapshot(rows=500):
    index = pd.date_range("2024-01-01", periods=rows, freq="B")
    return pd.DataFrame({name: np.arange(rows, dtype=float) + 1 for name in FEATURES}, index=index)


def setup_function():
    api.limiter._storage.reset()
    with api._predict_cache_lock:
        api._predict_cache.clear()
    with api._info_cache_lock:
        api._info_cache.clear()
    api._status_registry = api.PredictionStatusRegistry()
    api._record_upstream("available")


def test_health_and_readiness():
    api._record_upstream("available")
    assert client.get("/health").status_code == 200
    response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["dependencies"]["model_storage"]["required"] is False
    assert body["dependencies"]["model_storage"]["writable"] is None


def test_ready_reports_server_forecast_infrastructure_by_mode(monkeypatch):
    api._record_upstream("available")
    monkeypatch.setattr(api.settings, "training_mode", "server_pretrained")
    monkeypatch.setattr(api.settings, "server_forecast_serving_enabled", True)
    monkeypatch.setattr(api.settings, "registry_database_url", None)
    monkeypatch.setattr(api.settings, "s3_bucket", None)
    response = client.get("/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["dependencies"]["server_forecasts"] == {
        "configured": False,
        "status": "unconfigured",
        "required": True,
        "bundle_retention_days": 30,
    }
    assert body["dependencies"]["model_storage"]["required"] is True

    monkeypatch.setattr(api.settings, "training_mode", "browser_only")
    healthy = client.get("/ready")
    assert healthy.status_code == 200
    assert healthy.json()["dependencies"]["server_forecasts"]["required"] is False


def test_ready_reflects_market_circuit_breaker_open_and_recovery():
    api._record_upstream("available")
    assert client.get("/ready").status_code == 200
    assert client.get("/health").status_code == 200

    # Trip breaker by recording failures
    for _ in range(3):
        api._record_upstream("unavailable", "Simulated connection refused")

    assert client.get("/health").status_code == 200  # Liveness remains 200
    unready = client.get("/ready")
    assert unready.status_code == 503
    body = unready.json()
    assert body["status"] == "degraded"
    assert body["dependencies"]["market_data"]["status"] == "unavailable"
    assert body["dependencies"]["market_data"]["circuit"] == "open"

    # Recover
    api._record_upstream("available")
    recovered = client.get("/ready")
    assert recovered.status_code == 200
    assert recovered.json()["status"] == "ready"
    assert recovered.json()["dependencies"]["market_data"]["status"] == "available"


def test_root_discloses_service_routes():

    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "StockLSTM API"
    assert body["status"] == "online"
    assert body["docs"] == "/docs"
    assert body["readiness"] == "/ready"


def test_validate_ticker_and_horizon():
    assert client.get("/api/v1/predict?ticker=../etc/passwd").status_code == 400
    assert client.get("/api/v1/predict?ticker=ABCDEFGHIJKLM").status_code == 400
    assert client.get("/api/v1/predict?ticker=AAPL&days=99").status_code == 422


def test_missing_artifact_serves_labelled_baseline_after_market_data():
    snapshot = _feature_snapshot()
    with (
        patch(
            "api.load_fresh_artifact",
            side_effect=api.ArtifactValidationError("No versioned artifact is active."),
        ),
        patch(
            "api.fetch_data",
            return_value=(snapshot, snapshot.Close.values, snapshot.index, {}),
        ) as fetch,
    ):
        response = client.get("/api/v1/predict?ticker=AAPL&days=7")
    assert response.status_code == 200
    body = response.json()
    assert body["metadata"]["engine"]["baseline_fallback"]
    assert body["metadata"]["execution"]["mode"] == "baseline_fallback"
    assert body["predicted_prices"] == [float(snapshot.Close.iloc[-1])] * 7
    fetch.assert_called_once()


def _request(peer: str, forwarded: str | None = None):
    headers = [] if forwarded is None else [(b"x-forwarded-for", forwarded.encode())]
    return api.Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/v1/predict",
            "headers": headers,
            "client": (peer, 1234),
            "server": ("test", 80),
            "scheme": "http",
            "query_string": b"",
        }
    )


def test_rate_limit_identity_uses_only_explicitly_trusted_proxy(monkeypatch):
    monkeypatch.setattr(api, "_trusted_proxy_ips", frozenset({"172.30.30.10", "10.0.0.0/24"}))
    assert api.rate_limit_identity(_request("198.51.100.7", "203.0.113.9")) == "198.51.100.7"
    assert api.rate_limit_identity(_request("172.30.30.10", "203.0.113.9")) == "203.0.113.9"
    assert api.rate_limit_identity(_request("10.0.0.50", "203.0.113.9")) == "203.0.113.9"
    assert (
        api.rate_limit_identity(_request("172.30.30.10", "spoofed, 203.0.113.9")) == "172.30.30.10"
    )
    assert (
        api.rate_limit_identity(_request("172.30.30.10", "198.51.100.1, 203.0.113.9, 172.30.30.10"))
        == "203.0.113.9"
    )


def test_forecast_rate_limits_are_isolated_by_trusted_forwarded_client(monkeypatch):
    monkeypatch.setattr(api, "_trusted_proxy_ips", frozenset({"172.30.30.10"}))

    with (
        patch("api._price_prediction_pipeline", return_value=_response()),
        patch("api.load_fresh_artifact", return_value=(MagicMock(), MagicMock())),
        TestClient(app, client=("172.30.30.10", 1234)) as proxy_client,
    ):
        for address in ("198.51.100.21", "198.51.100.22"):
            headers = {"X-Forwarded-For": address}
            for _ in range(5):
                assert (
                    proxy_client.get("/api/v1/predict?ticker=AAPL", headers=headers).status_code
                    == 200
                )
            assert (
                proxy_client.get("/api/v1/predict?ticker=AAPL", headers=headers).status_code == 429
            )


def test_direct_forecast_requests_cannot_spoof_rate_limit_identity(monkeypatch):
    monkeypatch.setattr(api, "_trusted_proxy_ips", frozenset({"172.30.30.10"}))

    with (
        patch("api._price_prediction_pipeline", return_value=_response()),
        patch("api.load_fresh_artifact", return_value=(MagicMock(), MagicMock())),
        TestClient(app, client=("198.51.100.23", 1234)) as direct_client,
    ):
        for number in range(5):
            response = direct_client.get(
                "/api/v1/predict?ticker=AAPL",
                headers={"X-Forwarded-For": f"203.0.113.{number + 1}"},
            )
            assert response.status_code == 200
        assert (
            direct_client.get(
                "/api/v1/predict?ticker=AAPL",
                headers={"X-Forwarded-For": "203.0.113.99"},
            ).status_code
            == 429
        )


def test_openapi_contains_public_routes_and_horizon_constraints():
    schema = client.get("/openapi.json").json()
    for route in (
        "/api/v1/predict",
        "/api/v1/predict/direction",
        "/api/v1/search",
        "/api/v1/info",
        "/api/v1/training-data",
        "/api/v1/prediction-status/{request_id}",
        "/api/v1/model-performance/{ticker}",
    ):
        assert route in schema["paths"]
    days = next(
        parameter
        for parameter in schema["paths"]["/api/v1/predict"]["get"]["parameters"]
        if parameter["name"] == "days"
    )
    assert days["schema"]["minimum"] == 1
    assert days["schema"]["maximum"] == MAX_FORECAST_DAYS


def test_model_performance_discloses_global_volatility_engine():
    response = client.get("/api/v1/model-performance/AAPL")
    assert response.status_code == 200
    body = response.json()
    assert body["engine"]["family"] == "baseline_residual_tcn_ensemble"
    assert body["engine"]["role"] == "global_volatility"
    assert body["engine"]["status"] == "unconfigured"
    assert body["metrics"]["metric_source"] == "locked_purged_walk_forward"
    assert body["benchmark"]["validation_folds"] == 5


def test_forecast_openapi_declares_shared_telemetry_contract():
    schema = app.openapi()
    paths = schema["paths"]
    schemas = schema["components"]["schemas"]

    price_response = paths["/api/v1/predict"]["get"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"]
    direction_response = paths["/api/v1/predict/direction"]["get"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"]
    assert price_response["$ref"].endswith("/PriceForecastResponse")
    assert direction_response["$ref"].endswith("/DirectionForecastResponse")

    for response_name in ("PriceForecastResponse", "DirectionForecastResponse"):
        metadata = schemas[response_name]["properties"]["metadata"]
        assert metadata["$ref"].endswith("/ForecastMetadata")

    timing_schema = schemas["PredictionTimings"]
    assert set(TIMING_FIELDS) == set(timing_schema["properties"])
    assert set(TIMING_FIELDS).issubset(timing_schema["required"])
    for name in TIMING_FIELDS:
        field_schema = timing_schema["properties"][name]
        if name == "total":
            assert field_schema["type"] == "number"
            assert field_schema["minimum"] == 0
        else:
            variants = field_schema["anyOf"]
            assert {variant["type"] for variant in variants} == {"number", "null"}
            number_variant = next(variant for variant in variants if variant["type"] == "number")
            assert number_variant["minimum"] == 0

    execution = schemas["PredictionExecution"]["properties"]
    assert set(execution["mode"]["enum"]) == set(EXECUTION_MODES)
    assert execution["coalesced"]["type"] == "boolean"

    metadata = schemas["ForecastMetadata"]["properties"]
    artifact_states = next(
        variant["enum"]
        for variant in metadata["artifact_state_before"]["anyOf"]
        if "enum" in variant
    )
    assert set(artifact_states) == set(ARTIFACT_STATES)
    assert set(metadata["artifact_action"]["enum"]) == set(ARTIFACT_ACTIONS)

    status_response = paths["/api/v1/prediction-status/{request_id}"]["get"]["responses"]["200"][
        "content"
    ]["application/json"]["schema"]
    assert status_response["$ref"].endswith("/PredictionStatusResponse")
    for path in ("/api/v1/predict", "/api/v1/predict/direction"):
        assert "503" in paths[path]["get"]["responses"]
        headers = {
            parameter["name"]
            for parameter in paths[path]["get"]["parameters"]
            if parameter["in"] == "header"
        }
        assert "X-Prediction-Request-ID" in headers


def test_forecast_contract_rejects_malformed_telemetry():
    payload = _response()
    payload["metadata"].update(
        {
            "timings_seconds": {
                "queue_wait": None,
                "market_data": None,
                "feature_preparation": None,
                "artifact_load_validation": None,
                "training": None,
                "inference": None,
                "total": -1,
            },
            "execution": {"mode": "artifact_loaded", "coalesced": False},
            "artifact_state_before": "fresh",
            "artifact_action": "loaded",
        }
    )
    with pytest.raises(ValidationError):
        api.PriceForecastResponse.model_validate(payload)


def test_predict_response_schema():
    with patch("api._price_prediction_pipeline", return_value=_response()):
        response = client.get("/api/v1/predict?ticker=AAPL&days=7")
    assert response.status_code == 200
    body = response.json()
    assert body["ticker"] == "AAPL"
    assert len(body["predicted_prices"]) == len(body["future_dates"]) == 7
    assert body["metadata"]["timings_seconds"]["total"] is not None
    assert body["metadata"]["execution"]["mode"] == "artifact_loaded"

    with patch("api._direction_prediction_pipeline", return_value=_response(direction=True)):
        direction = client.get("/api/v1/predict/direction?ticker=AAPL&days=7")
    assert direction.status_code == 200
    assert direction.json()["directions"] == ["Up"] * 7


def test_price_pipeline_serves_labelled_persistence_when_artifact_is_unavailable(
    synthetic_feature_df,
):
    dates = synthetic_feature_df.index
    closes = synthetic_feature_df["Close"].to_numpy()
    with (
        patch(
            "api.load_fresh_artifact",
            side_effect=api.ArtifactValidationError("missing"),
        ),
        patch(
            "api._fetch_snapshot",
            return_value=(synthetic_feature_df, closes, dates, {"market_context": {}}),
        ),
    ):
        response = api._price_prediction_pipeline("AAPL", 3)

    assert response["predicted_prices"] == [float(closes[-1])] * 3
    assert response["metadata"]["engine"] == {
        "family": "persistence",
        "role": "server_disabled_fallback",
        "baseline_fallback": True,
    }
    assert response["metrics"]["metric_source"] == "baseline_definition"


def test_direction_pipeline_serves_labelled_base_rate_without_attention(
    synthetic_feature_df,
):
    dates = synthetic_feature_df.index
    closes = synthetic_feature_df["Close"].to_numpy()
    with (
        patch(
            "api.load_fresh_artifact",
            side_effect=api.ArtifactValidationError("missing"),
        ),
        patch(
            "api._fetch_snapshot",
            return_value=(synthetic_feature_df, closes, dates, {"market_context": {}}),
        ),
        patch("api.get_financial_sentiment", return_value={"score": 0.0}),
    ):
        response = api._direction_prediction_pipeline("AAPL", 3)

    assert len(response["directions"]) == len(response["probabilities"]) == 3
    assert response["attention_weights"] == []
    assert response["metadata"]["engine"]["baseline_fallback"]
    assert response["metrics"]["metric_source"] == "baseline_definition"


def test_response_cache_timing_and_status_are_truthful():
    with patch("api._price_prediction_pipeline", return_value=_response()):
        assert client.get("/api/v1/predict?ticker=AAPL&days=7").status_code == 200
        request_id = str(uuid.uuid4())
        response = client.get(
            "/api/v1/predict?ticker=AAPL&days=7",
            headers={"X-Prediction-Request-ID": request_id},
        )
    assert response.status_code == 200
    metadata = response.json()["metadata"]
    assert metadata["execution"] == {"mode": "response_cache_hit", "coalesced": False}
    assert metadata["artifact_action"] == "not_applicable"
    assert metadata["artifact_state_before"] is None
    assert metadata["timings_seconds"]["total"] is not None
    assert all(
        metadata["timings_seconds"][name] is None for name in TIMING_FIELDS if name != "total"
    )
    status = client.get(f"/api/v1/prediction-status/{request_id}")
    assert status.status_code == 200
    assert status.headers["cache-control"] == "no-store"
    assert status.json()["stage"] == "completed"


def test_response_cache_does_not_depend_on_server_artifact_freshness():
    with patch("api._price_prediction_pipeline", return_value=_response()):
        first = client.get("/api/v1/predict?ticker=AAPL&days=7")
    assert first.status_code == 200
    response = client.get("/api/v1/predict?ticker=AAPL&days=7")
    assert response.status_code == 200
    assert response.json()["metadata"]["execution"]["mode"] == "response_cache_hit"


def test_status_registry_coalesces_caller_views_and_hides_unknown_ids():
    registry = api.PredictionStatusRegistry(max_entries=2, ttl_seconds=600)
    owner_id, joiner_id = str(uuid.uuid4()), str(uuid.uuid4())
    owner = api.PredictionJob("AAPL_7")
    assert registry.attach(owner_id, owner, coalesced=False)
    assert registry.attach(joiner_id, owner, coalesced=True)
    owner.start()
    owner.set_stage("training")
    assert registry.get(joiner_id) == {"status": "running", "stage": "training", "coalesced": True}
    unknown = client.get(f"/api/v1/prediction-status/{uuid.uuid4()}")
    malformed = client.get("/api/v1/prediction-status/not-a-uuid")
    assert unknown.status_code == malformed.status_code == 404
    assert unknown.headers["cache-control"] == malformed.headers["cache-control"] == "no-store"


def test_status_registry_capacity_prefers_terminal_eviction_and_remains_bounded():
    registry = api.PredictionStatusRegistry(max_entries=2, ttl_seconds=600)
    active_id = str(uuid.uuid4())
    terminal_id = str(uuid.uuid4())
    replacement_id = str(uuid.uuid4())

    assert registry.attach(active_id, api.PredictionJob("active"), coalesced=False)
    assert registry.attach(
        terminal_id, api.PredictionJob("terminal"), coalesced=False, terminal=True
    )
    assert registry.attach(replacement_id, api.PredictionJob("replacement"), coalesced=False)

    assert len(registry._views) == registry.max_entries
    assert active_id in registry._views
    assert replacement_id in registry._views
    assert terminal_id not in registry._views

    api._status_registry = registry
    evicted = client.get(f"/api/v1/prediction-status/{terminal_id}")
    assert evicted.status_code == 404
    assert evicted.json() == {"detail": "Prediction status is unavailable."}
    assert evicted.headers["cache-control"] == "no-store"

    active_only = api.PredictionStatusRegistry(max_entries=1, ttl_seconds=600)
    first_active = str(uuid.uuid4())
    assert active_only.attach(first_active, api.PredictionJob("first"), coalesced=False)
    assert not active_only.attach(str(uuid.uuid4()), api.PredictionJob("second"), coalesced=False)
    assert list(active_only._views) == [first_active]


def test_active_registry_views_have_a_finite_watchdog_deadline():
    registry = api.PredictionStatusRegistry(max_entries=4, ttl_seconds=60)
    active_id = str(uuid.uuid4())
    assert registry.attach(active_id, api.PredictionJob("active"), coalesced=False)
    view = registry._views[active_id]
    assert view["expires_at"] < float("inf")
    assert view["expires_at"] == view["expires_at"] <= time.monotonic() + 60 * 61


def test_orphaned_active_view_is_reclaimed_after_the_watchdog(monkeypatch):
    registry = api.PredictionStatusRegistry(max_entries=1, ttl_seconds=1)
    orphan_id = str(uuid.uuid4())
    assert registry.attach(orphan_id, api.PredictionJob("orphan"), coalesced=False)
    registry._views[orphan_id]["expires_at"] = time.monotonic() - 1

    assert registry.get(orphan_id) is None
    replacement_id = str(uuid.uuid4())
    assert registry.attach(replacement_id, api.PredictionJob("replacement"), coalesced=False)
    assert list(registry._views) == [replacement_id]


def test_cancelled_prediction_expires_its_status_view(monkeypatch):
    coordinator = WorkCoordinator(workers=1, queue_size=0)
    monkeypatch.setattr(api, "_work_coordinator", coordinator)
    request_id = str(uuid.uuid4())
    gate = threading.Event()

    def pipeline(*_args, **_kwargs):
        gate.wait(2)
        return _response()

    async def cancel_request():
        task = asyncio.create_task(
            api._await_prediction(
                "cancel_AAPL", pipeline, "AAPL", 7, time.perf_counter(), request_id
            )
        )
        await asyncio.sleep(0.01)
        task.cancel()
        with np.testing.assert_raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_request())
    view = api._status_registry._views[request_id]
    assert view["terminal"] is True
    assert view["lifecycle"] == "failed"
    assert view["expires_at"] < float("inf")
    gate.set()


def test_status_joiner_binds_to_the_coordinator_job(monkeypatch):
    coordinator = WorkCoordinator(workers=1, queue_size=1)
    monkeypatch.setattr(api, "_work_coordinator", coordinator)
    request_id = str(uuid.uuid4())
    started = threading.Event()
    gate = threading.Event()

    def pipeline(*_args, job, **_kwargs):
        job.add_timing("market_data", 0.25)
        job.add_timing("feature_preparation", 0.5)
        job.add_timing("artifact_load_validation", 0.125)
        job.add_timing("training", 0.75)
        job.add_timing("inference", 0.375)
        job.set_artifact("stale", "retrained")
        job.set_stage("training")
        started.set()
        gate.wait(2)
        return _response()

    async def coalesce_requests():
        owner = asyncio.create_task(
            api._await_prediction("shared_AAPL", pipeline, "AAPL", 7, time.perf_counter())
        )
        assert await asyncio.to_thread(started.wait, 1)
        joiner_started = time.perf_counter()
        joiner = asyncio.create_task(
            api._await_prediction("shared_AAPL", pipeline, "AAPL", 7, joiner_started, request_id)
        )
        await asyncio.sleep(0)
        assert api._status_registry.get(request_id) == {
            "status": "running",
            "stage": "training",
            "coalesced": True,
        }
        gate.set()
        return await asyncio.gather(owner, joiner)

    owner_result, joiner_result = asyncio.run(coalesce_requests())
    owner_metadata = owner_result["metadata"]
    joiner_metadata = joiner_result["metadata"]

    assert owner_metadata["execution"] == {"mode": "trained", "coalesced": False}
    assert owner_metadata["artifact_state_before"] == "stale"
    assert owner_metadata["artifact_action"] == "retrained"
    assert owner_metadata["timings_seconds"]["market_data"] == 0.25
    assert owner_metadata["timings_seconds"]["training"] == 0.75
    assert owner_metadata["timings_seconds"]["queue_wait"] is not None

    assert joiner_metadata["execution"] == {"mode": "coalesced", "coalesced": True}
    assert joiner_metadata["artifact_state_before"] == "stale"
    assert joiner_metadata["artifact_action"] == "retrained"
    assert joiner_metadata["timings_seconds"]["total"] is not None
    assert all(
        joiner_metadata["timings_seconds"][name] is None
        for name in TIMING_FIELDS
        if name != "total"
    )


def test_request_identifier_must_be_uuidv4():
    invalid = client.get(
        "/api/v1/predict?ticker=AAPL",
        headers={"X-Prediction-Request-ID": str(uuid.uuid1())},
    )
    assert invalid.status_code == 400


def test_cors_allows_prediction_request_identifier_header():
    response = client.options(
        "/api/v1/predict",
        headers={
            "Origin": api.settings.allowed_origins[0],
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "X-Prediction-Request-ID",
        },
    )
    assert response.status_code == 200
    assert "x-prediction-request-id" in response.headers["access-control-allow-headers"].lower()


def test_search_and_info_cache_are_sanitised_and_thread_safe():
    with patch("api.yf.Search") as search:
        search.return_value.quotes = [
            {"symbol": "AAPL", "longname": "Apple Inc.", "quoteType": "EQUITY"}
        ]
        assert client.get("/api/v1/search?query=Apple").json()["results"][0]["ticker"] == "AAPL"
    with patch("api.yf.Ticker") as ticker:
        ticker.return_value.info = {"longName": "Apple Inc."}
        assert client.get("/api/v1/info?ticker=AAPL").status_code == 200
        assert client.get("/api/v1/info?ticker=AAPL").status_code == 200
        assert ticker.call_count == 1


def test_direction_fixed_width_is_horizon_safe_in_both_request_orders():
    snapshot = _feature_snapshot()
    dates = snapshot.index
    scaler = MagicMock()

    def prediction(_model, _frame, _scaler, days):
        return ["Up"] * days, [0.6] * days, [1 / WINDOW_SIZE] * WINDOW_SIZE

    for order in ((3, 30), (30, 3)):
        with (
            patch(
                "api.fetch_data", return_value=(snapshot, snapshot.Close.values, dates, {})
            ) as fetch,
            patch("api.load_fresh_artifact", return_value=(MagicMock(), scaler)) as load,
            patch("api.predict_direction", side_effect=prediction),
            patch(
                "api.future_trading_dates",
                side_effect=lambda _t, _d, count: ([f"d{i}" for i in range(count)], "NYSE"),
            ),
            patch("api.get_financial_sentiment", return_value={"sentiment": {"score": 0.0}}),
            patch("api.load_metrics", return_value={"metric_source": "walk_forward_out_of_fold"}),
        ):
            results = [api._direction_prediction_pipeline("AAPL", days) for days in order]
        assert [result["forecast_days"] for result in results] == list(order)
        assert all(call.args[2] == MAX_FORECAST_DAYS for call in load.call_args_list)
        assert fetch.call_count == 2


def test_price_pipeline_uses_one_coherent_snapshot():
    snapshot = _feature_snapshot()
    closes = snapshot.Close.to_numpy()
    dates = snapshot.index
    with (
        patch(
            "api._fetch_snapshot",
            return_value=(snapshot, closes, dates, {"snapshot_id": "one", "market_context": {}}),
        ) as fetch,
        patch("api.future_trading_dates", return_value=(["d1", "d2", "d3"], "NYSE")),
    ):
        result = api._price_prediction_pipeline("AAPL", 3)
    assert fetch.call_count == 1
    assert result["metadata"]["data_snapshot"]["snapshot_id"] == "one"
    assert result["historical_prices"] == closes.tolist()
    assert result["predicted_prices"] == [float(closes[-1])] * 3


def test_work_coordinator_coalesces_and_bounds_queue():
    coordinator = WorkCoordinator(workers=1, queue_size=1)
    gate = threading.Event()
    calls = 0

    def work():
        nonlocal calls
        calls += 1
        gate.wait(2)
        return "done"

    first = coordinator.submit("same", work)
    assert coordinator.submit("same", work) is first
    queued = coordinator.submit("other", work)
    with np.testing.assert_raises(api.ServiceBusyError):
        coordinator.submit("overflow", work)
    gate.set()
    assert first.result(2) == queued.result(2) == "done"
    assert calls == 2


def test_health_remains_responsive_during_cold_work():
    gate = threading.Event()
    coordinator = WorkCoordinator(workers=1, queue_size=0)
    future = coordinator.submit("cold", lambda: gate.wait(2))
    with ThreadPoolExecutor(max_workers=1) as pool:
        response = pool.submit(client.get, "/health").result(1)
    gate.set()
    future.result(2)
    assert response.status_code == 200


def test_diagnostics_404_when_not_trained():
    with (
        patch("api.load_cross_validation", return_value={}),
        patch("api.load_validation_results", return_value=[]),
        patch("api.load_metadata", return_value={}),
    ):
        assert client.get("/api/v1/diagnostics/AAPL").status_code == 404


def test_models_advertises_signed_global_volatility_and_disabled_browser_training():
    body = client.get("/models").json()
    assert body["server_models"] == {
        "status": "disabled",
        "reason": "Legacy per-ticker server models are disabled; use the global volatility contract.",
        "training_mode": "browser_only",
    }
    assert body["global_volatility"]["endpoint"] == "/api/v2/forecast"
    assert body["global_volatility"]["certified_heads"]["volatility"] is True
    assert body["global_volatility"]["certified_heads"]["direction"] is False
    assert body["browser_training"]["status"] == "disabled"
    assert body["model_storage"]["location"] == "none"


def test_training_data_route_returns_validated_snapshot(monkeypatch):
    snapshot = {
        "ticker": "MSFT",
        "schema_version": 4,
        "snapshot_id": "snapshot-test",
        "feature_names": list(FEATURES_V4),
        "window_size": WINDOW_SIZE,
        "output_width": MAX_FORECAST_DAYS,
        "dates": ["2026-07-30"],
        "features": [[1.0] * len(FEATURES_V4)],
        "historical_prices": [100.0],
        "future_dates": ["2026-07-31"],
    }
    monkeypatch.setattr(
        api, "build_training_snapshot", lambda ticker: {**snapshot, "ticker": ticker}
    )
    response = client.get("/api/v1/training-data?ticker=MSFT")
    assert response.status_code == 200
    assert response.json()["feature_names"] == list(FEATURES_V4)
    assert client.get("/api/v1/training-data?ticker=../MSFT").status_code == 400


def test_training_data_coalesces_and_caches_with_lru_eviction(monkeypatch):
    build_count = 0

    def mock_build(ticker):
        nonlocal build_count
        build_count += 1
        return {
            "ticker": ticker,
            "schema_version": 4,
            "snapshot_id": f"snap-{ticker}",
            "feature_names": list(FEATURES_V4),
            "window_size": WINDOW_SIZE,
            "output_width": MAX_FORECAST_DAYS,
            "dates": ["2026-07-30"],
            "features": [[1.0] * len(FEATURES_V4)],
            "historical_prices": [100.0],
            "future_dates": ["2026-07-31"],
        }

    monkeypatch.setattr(api, "build_training_snapshot", mock_build)
    api._snapshot_cache.clear()

    # 1. First call builds
    r1 = client.get("/api/v1/training-data?ticker=AAPL")
    assert r1.status_code == 200
    assert build_count == 1

    # 2. Second call hits cache (no new build)
    r2 = client.get("/api/v1/training-data?ticker=AAPL")
    assert r2.status_code == 200
    assert build_count == 1

    # 3. Cache returns independent copies
    body1 = r1.json()
    body1["historical_prices"].append(999.0)
    body2 = client.get("/api/v1/training-data?ticker=AAPL").json()
    assert body2["historical_prices"] == [100.0]

    # 4. Fill cache beyond max (6) to trigger LRU eviction
    for tkr in ["MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]:
        client.get(f"/api/v1/training-data?ticker={tkr}")

    # AAPL was oldest and should have been evicted
    client.get("/api/v1/training-data?ticker=AAPL")
    assert build_count == 8  # AAPL rebuilt
