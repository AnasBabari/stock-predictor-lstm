from __future__ import annotations

from fastapi.testclient import TestClient

from api import app
from routes import health

CLIENT = TestClient(app)


def test_readiness_is_503_without_current_market_data_evidence(monkeypatch) -> None:
    monkeypatch.setattr(
        health.market_data_service,
        "readiness",
        lambda: (False, {"status": "unavailable", "fresh_cache_entries": 0}),
    )
    monkeypatch.setattr(
        health.market_circuit_breaker,
        "is_ready",
        lambda: (True, {"status": "available", "circuit": "closed"}),
    )
    response = CLIENT.get("/ready")
    assert response.status_code == 503
    assert response.json()["status"] == "degraded"


def test_readiness_is_200_with_fresh_market_data_evidence(monkeypatch) -> None:
    monkeypatch.setattr(
        health.market_data_service,
        "readiness",
        lambda: (True, {"status": "available", "fresh_cache_entries": 1}),
    )
    monkeypatch.setattr(
        health.market_circuit_breaker,
        "is_ready",
        lambda: (True, {"status": "available", "circuit": "closed"}),
    )
    response = CLIENT.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
