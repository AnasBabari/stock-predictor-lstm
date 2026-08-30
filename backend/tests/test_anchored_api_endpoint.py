"""Unit tests for anchored forecast API endpoint."""

from fastapi.testclient import TestClient

from api import app


def test_anchored_forecast_endpoint_returns_valid_p0_and_prices():
    client = TestClient(app)
    response = client.get("/api/v1/forecast/anchored?ticker=BP&days=7")
    assert response.status_code == 200
    data = response.json()

    assert data["ticker"] == "BP"
    assert data["base_price"] > 0
    assert "base_date" in data
    assert len(data["forecast"]["median_prices"]) == 7
    assert len(data["forecast"]["intervals_80pct"]) == 7
    assert data["evidence"]["contract_id"] == "price-return-distribution-v1"

    # Invariant: First day price is anchored close to P0
    p0 = data["base_price"]
    p1 = data["forecast"]["median_prices"][0]
    assert abs(p1 - p0) < 1.0, f"Day 1 price {p1} diverged dramatically from P0 {p0}"
