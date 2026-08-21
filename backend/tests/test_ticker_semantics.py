"""HTTP semantics for ticker/data failures and bounded snapshot queueing.

Status contract:
- 400: syntactically invalid ticker (validate_ticker)
- 404: well-formed symbol the provider knows nothing about (UnknownTickerError)
- 422: symbol exists but data is unusable/insufficient (MarketDataUnavailable)
- 503: provider transport/circuit/context failures, or a snapshot build that
       stayed queued longer than the configured wait
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import api
from data_pipeline import MarketDataUnavailable, UnknownTickerError


@pytest.fixture
def client():
    with TestClient(api.app) as test_client:
        yield test_client


def test_predict_unknown_ticker_returns_404(client):
    with patch(
        "api.fetch_data",
        side_effect=UnknownTickerError("No market data is available for ZZZZZZ."),
    ):
        response = client.get("/api/v1/predict?ticker=ZZZZZZ&days=3")
    assert response.status_code == 404
    assert "No market data" in response.json()["detail"]


def test_predict_unusable_market_data_returns_422(client):
    with patch(
        "api.fetch_data",
        side_effect=MarketDataUnavailable("Not enough historical data for ABC."),
    ):
        response = client.get("/api/v1/predict?ticker=AAPL&days=3")
    assert response.status_code == 422


def test_direction_unknown_ticker_returns_404(client):
    with patch(
        "api.fetch_data",
        side_effect=UnknownTickerError("No market data is available for NOPE."),
    ):
        response = client.get("/api/v1/predict/direction?ticker=NOPE&days=3")
    assert response.status_code == 404


def test_training_data_unknown_ticker_returns_404(client):
    with patch(
        "api.build_training_snapshot",
        side_effect=UnknownTickerError("No market data is available for ZZZZZZ."),
    ):
        response = client.get("/api/v1/training-data?ticker=ZZZZZZ")
    assert response.status_code == 404


def test_training_data_insufficient_history_returns_422(client):
    with patch(
        "api.build_training_snapshot",
        side_effect=MarketDataUnavailable(
            "Not enough historical data for ABC. Need at least 120 trading days."
        ),
    ):
        response = client.get("/api/v1/training-data?ticker=ABC")
    assert response.status_code == 422
    assert response.json()["detail"].startswith("Not enough historical data")


def test_snapshot_queue_bounded_wait_returns_503(client, monkeypatch):
    import asyncio

    async def slow_build(_ticker: str) -> dict:
        await asyncio.sleep(5)
        return {}

    monkeypatch.setattr(api.settings, "snapshot_build_wait_seconds", 1)
    monkeypatch.setattr(api, "_snapshot_cache", {})
    monkeypatch.setattr(api, "_in_flight_tasks", {})
    monkeypatch.setattr(api, "_execute_snapshot_build", slow_build)

    response = client.get("/api/v1/training-data?ticker=SLOW")
    assert response.status_code == 503
    assert "queued behind other requests" in response.json()["detail"]
