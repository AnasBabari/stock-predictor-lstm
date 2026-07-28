import asyncio
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

import api
from api import WorkCoordinator, app
from config import FEATURES, MAX_FORECAST_DAYS, WINDOW_SIZE

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


def test_health_and_readiness(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "MODEL_DIR", str(tmp_path))
    api._record_upstream("available")
    assert client.get("/health").status_code == 200
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["dependencies"]["model_storage"]["writable"] is True


def test_validate_ticker_and_horizon():
    assert client.get("/api/v1/predict?ticker=../etc/passwd").status_code == 400
    assert client.get("/api/v1/predict?ticker=ABCDEFGHIJKLM").status_code == 400
    assert client.get("/api/v1/predict?ticker=AAPL&days=99").status_code == 422


def test_openapi_contains_public_routes_and_horizon_constraints():
    schema = client.get("/openapi.json").json()
    for route in (
        "/api/v1/predict",
        "/api/v1/predict/direction",
        "/api/v1/search",
        "/api/v1/info",
        "/api/v1/prediction-status/{request_id}",
    ):
        assert route in schema["paths"]
    days = next(
        parameter
        for parameter in schema["paths"]["/api/v1/predict"]["get"]["parameters"]
        if parameter["name"] == "days"
    )
    assert days["schema"]["minimum"] == 1
    assert days["schema"]["maximum"] == MAX_FORECAST_DAYS


def test_predict_response_schema():
    with patch("api._price_prediction_pipeline", return_value=_response()):
        response = client.get("/api/v1/predict?ticker=AAPL&days=7")
    assert response.status_code == 200
    body = response.json()
    assert body["ticker"] == "AAPL"
    assert len(body["predicted_prices"]) == len(body["future_dates"]) == 7
    assert body["metadata"]["timings_seconds"]["total"] is not None
    assert body["metadata"]["execution"]["mode"] == "artifact_loaded"


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
        metadata["timings_seconds"][name] is None for name in api.TIMING_FIELDS if name != "total"
    )

    status = client.get(f"/api/v1/prediction-status/{request_id}")
    assert status.status_code == 200
    assert status.headers["cache-control"] == "no-store"
    assert status.json()["stage"] == "completed"


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
        for name in api.TIMING_FIELDS
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
    arrays = (
        np.zeros((2, WINDOW_SIZE, len(FEATURES))),
        np.zeros((1, WINDOW_SIZE, len(FEATURES))),
        np.zeros((2, MAX_FORECAST_DAYS)),
        np.zeros((1, MAX_FORECAST_DAYS)),
        MagicMock(),
        [],
        [],
    )

    def prediction(_model, _frame, _scaler, days):
        return ["Up"] * days, [0.6] * days, [1 / WINDOW_SIZE] * WINDOW_SIZE

    for order in ((3, 30), (30, 3)):
        with (
            patch(
                "api.fetch_data", return_value=(snapshot, snapshot.Close.values, dates, {})
            ) as fetch,
            patch("api.prepare_return_data", return_value=arrays) as prepare,
            patch("api.load_or_train", return_value=(MagicMock(), arrays[4])),
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
        assert all(
            call.kwargs["forecast_days"] == MAX_FORECAST_DAYS for call in prepare.call_args_list
        )
        assert fetch.call_count == 2


def test_price_pipeline_uses_one_coherent_snapshot():
    snapshot = _feature_snapshot()
    original_id = id(snapshot)
    observed = []
    arrays = (
        np.zeros((2, WINDOW_SIZE, len(FEATURES))),
        np.zeros((1, WINDOW_SIZE, len(FEATURES))),
        np.zeros((2, MAX_FORECAST_DAYS)),
        np.zeros((1, MAX_FORECAST_DAYS)),
        MagicMock(),
        [],
        [],
    )

    def preprocess(frame, **_kwargs):
        observed.append(frame.attrs["snapshot"])
        return arrays

    snapshot.attrs["snapshot"] = "immutable-one"
    with (
        patch(
            "api.fetch_data",
            return_value=(snapshot, snapshot.Close.values, snapshot.index, {"snapshot_id": "one"}),
        ) as fetch,
        patch("api.preprocess", side_effect=preprocess),
        patch("api.load_or_train", return_value=(MagicMock(), arrays[4])) as load,
        patch("api.predict_future", return_value=[101.0] * 3),
        patch("api.future_trading_dates", return_value=(["d1", "d2", "d3"], "NYSE")),
        patch("api.load_metrics", return_value={"metric_source": "walk_forward_out_of_fold"}),
    ):
        result = api._price_prediction_pipeline("AAPL", 3)
    assert id(snapshot) == original_id
    assert fetch.call_count == 1
    assert observed == ["immutable-one"]
    assert load.call_args.args[7].attrs["snapshot"] == "immutable-one"
    assert result["metadata"]["data_snapshot"]["snapshot_id"] == "one"


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
