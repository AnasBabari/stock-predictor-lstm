import importlib
import sys
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import config
from server_models.compatibility import CompatibilityReport
from server_models.contracts import (
    ReproducibilityMetadata,
    RobustScalerParams,
    ServerArtifactKey,
    ServerForecastBundle,
    ServerModelRecord,
)


def reload_api_with_flag(enabled: bool):
    orig = config.settings.server_forecast_serving_enabled
    config.settings.server_forecast_serving_enabled = enabled
    if "api" in sys.modules:
        importlib.reload(sys.modules["api"])
    from api import app

    return app, orig


def test_api_docs_unchanged_when_disabled():
    app, orig = reload_api_with_flag(False)
    client = TestClient(app)
    response = client.get("/openapi.json")
    config.settings.server_forecast_serving_enabled = orig
    assert "/api/v1/server-forecasts/availability" not in response.json()["paths"]


def test_api_docs_includes_paths_when_enabled():
    app, orig = reload_api_with_flag(True)
    client = TestClient(app)
    response = client.get("/openapi.json")
    config.settings.server_forecast_serving_enabled = orig
    assert "/api/v1/server-forecasts/availability" in response.json()["paths"]


@pytest.fixture
def test_app():
    app, orig_enabled = reload_api_with_flag(True)
    orig_allowlist = config.settings.server_forecast_allowlist
    config.settings.server_forecast_allowlist = ["AAPL"]
    yield app
    config.settings.server_forecast_allowlist = orig_allowlist
    config.settings.server_forecast_serving_enabled = orig_enabled


@pytest.fixture
def client(test_app):
    return TestClient(test_app)


def create_mock_record(ticker="AAPL"):
    key = ServerArtifactKey.create(ticker=ticker, snapshot_id="1234567")
    repro = ReproducibilityMetadata(
        feature_names=["f1"] * 28,  # doesn't matter for these tests unless we test compat
        scaler=RobustScalerParams(medians=[0.0] * 28, iqrs=[1.0] * 28),
        python_version="3.12",
        git_commit="unknown",
    )
    return ServerModelRecord(
        key=key,
        reproducibility=repro,
        sha256_digest="0" * 64,
        signature="fake",
        schema_version=2,
    )


def test_availability_endpoint(test_app, client):
    mock_registry = MagicMock()
    record = create_mock_record()
    mock_registry.get_promoted.return_value = record

    from server_models.api import get_registry

    test_app.dependency_overrides[get_registry] = lambda: mock_registry

    response = client.get("/api/v1/server-forecasts/availability")
    assert response.status_code == 200
    data = response.json()
    assert "AAPL" in data["allowlist"]
    assert len(data["tickers"]) == 1
    assert data["tickers"][0]["ticker"] == "AAPL"
    assert data["tickers"][0]["status"] == "fresh"


def test_forecast_missing_fallback(test_app, client):
    mock_registry = MagicMock()
    mock_registry.get_promoted.return_value = None
    from server_models.api import get_registry

    test_app.dependency_overrides[get_registry] = lambda: mock_registry

    response = client.get("/api/v1/server-forecasts/AAPL")
    assert response.status_code == 200
    data = response.json()
    assert data["available"] is False
    assert data["reason"] == "missing"


def test_forecast_stale_fallback(test_app, client, monkeypatch):
    mock_registry = MagicMock()
    mock_registry.get_promoted.return_value = create_mock_record()
    from server_models.api import get_registry

    test_app.dependency_overrides[get_registry] = lambda: mock_registry

    monkeypatch.setattr("server_models.api.is_fresh", lambda *args, **kwargs: False)

    response = client.get("/api/v1/server-forecasts/AAPL")
    assert response.status_code == 200
    data = response.json()
    assert data["available"] is False
    assert data["reason"] == "stale"


def test_forecast_incompatible_fallback(test_app, client, monkeypatch):
    mock_registry = MagicMock()
    mock_registry.get_promoted.return_value = create_mock_record()
    from server_models.api import get_registry

    test_app.dependency_overrides[get_registry] = lambda: mock_registry

    monkeypatch.setattr("server_models.api.is_fresh", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        "server_models.api.check_record_compatibility",
        lambda r: CompatibilityReport(compatible=False, reason="test"),
    )

    response = client.get("/api/v1/server-forecasts/AAPL")
    assert response.status_code == 200
    data = response.json()
    assert data["available"] is False
    assert data["reason"] == "incompatible"


def test_forecast_success(test_app, client, monkeypatch):
    mock_registry = MagicMock()
    record = create_mock_record()
    mock_registry.get_promoted.return_value = record
    from server_models.api import get_registry, get_storage

    test_app.dependency_overrides[get_registry] = lambda: mock_registry

    monkeypatch.setattr("server_models.api.is_fresh", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        "server_models.api.check_record_compatibility",
        lambda r: CompatibilityReport(compatible=True, reason="ok"),
    )

    mock_storage = MagicMock()
    bundle = ServerForecastBundle(
        version_id=record.key.version_id,
        ticker="AAPL",
        generated_at=record.key.trained_at,
        origin_close=150.0,
        origin_date="2026-08-01",
        evidence={},
        future_dates=[datetime.now(UTC).date() + timedelta(days=i) for i in range(30)],
        predicted_prices=[100.0] * 30,
        predicted_log_returns=[0.0] * 30,
    )
    mock_storage.get_bundle.return_value = bundle.model_dump_json().encode("utf-8")
    test_app.dependency_overrides[get_storage] = lambda: mock_storage

    response = client.get("/api/v1/server-forecasts/AAPL?days=7")
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "AAPL"
    assert data["forecast_days"] == 7
    assert len(data["predicted_prices"]) == 7
    assert data["metadata"]["engine"]["role"] == "server_pretrained"
