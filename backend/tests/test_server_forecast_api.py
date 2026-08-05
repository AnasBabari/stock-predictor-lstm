import hashlib
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi.testclient import TestClient

from api import app
from artifacts.signing import Ed25519ManifestSigner, Ed25519ManifestVerifier
from server_models.compatibility import CompatibilityReport
from server_models.contracts import (
    FORECAST_LENGTH,
    HISTORY_DISPLAY_WINDOW,
    ReproducibilityMetadata,
    RobustScalerParams,
    ServerArtifactKey,
    ServerForecastBundle,
    ServerModelRecord,
)

CLIENT = TestClient(app)


def make_record(ticker="AAPL", forecast_type="price", snapshot_id="snap-1"):
    key = ServerArtifactKey.create(
        ticker=ticker, snapshot_id=snapshot_id, forecast_type=forecast_type
    )
    repro = ReproducibilityMetadata(
        feature_names=["f1"] * 28,
        scaler=RobustScalerParams(medians=[0.0] * 28, iqrs=[1.0] * 28),
        python_version="3.12",
        git_commit="unknown",
    )
    return ServerModelRecord(key=key, reproducibility=repro, sha256_digest="0" * 64)


def make_bundle(record, *, signer=None):
    """Return (bundle, digest, payload, record-with-matched-digest)."""
    bundle = ServerForecastBundle(
        version_id=record.key.version_id,
        ticker=record.key.ticker,
        forecast_type=record.key.forecast_type,
        generated_at=record.key.trained_at,
        origin_close=150.0,
        origin_date="2026-07-31",
        evidence={
            "metric_source": "server_purged_walk_forward",
            "family": "elastic_net",
            "pooled": {"relative_rmse": 0.9},
        },
        future_dates=[datetime.now(UTC).date() + timedelta(days=i) for i in range(FORECAST_LENGTH)],
        predicted_prices=[100.0 * 1.001**i for i in range(FORECAST_LENGTH)],
        predicted_log_returns=[0.001] * FORECAST_LENGTH,
        historical_dates=[
            datetime.now(UTC).date() - timedelta(days=i) for i in range(HISTORY_DISPLAY_WINDOW)
        ][::-1],
        historical_prices=[150.0 * 0.999**i for i in range(HISTORY_DISPLAY_WINDOW)][::-1],
    )
    payload = bundle.model_dump_json().encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    record = record.model_copy(update={"sha256_digest": digest})
    if signer is not None:
        record = record.model_copy(update={"signature": signer(payload)})
    return bundle, digest, payload, record


def _enable(monkeypatch, *, enabled=True, mode="hybrid", allowlist=("AAPL",)):
    import config

    settings = config.settings
    monkeypatch.setattr(settings, "server_forecast_serving_enabled", enabled)
    monkeypatch.setattr(settings, "training_mode", mode)
    monkeypatch.setattr(settings, "server_forecast_allowlist", list(allowlist))
    monkeypatch.setattr(settings, "server_forecast_public_key_path", None)


def _stub_serve_deps(monkeypatch, *, registry=None, storage=None, verifier=None):

    monkeypatch.setattr(
        "server_models.api.get_registry",
        (lambda: registry) if registry is not None else (lambda: MagicMock()),
    )
    monkeypatch.setattr(
        "server_models.api.get_storage",
        (lambda: storage) if storage is not None else (lambda: MagicMock()),
    )
    monkeypatch.setattr(
        "server_models.api.get_verifier",
        (lambda: verifier) if verifier is not None else (lambda: None),
    )


class _BrokenRegistry:
    def get_promoted(self, *args, **kwargs):
        raise RuntimeError("postgres down")


def _happy_path(monkeypatch, record):
    _bundle, _digest, payload, final_record = make_bundle(record)
    registry = MagicMock()
    registry.get_promoted.return_value = final_record
    storage = MagicMock()
    storage.get_bundle.return_value = payload
    _stub_serve_deps(monkeypatch, registry=registry, storage=storage)
    monkeypatch.setattr(
        "server_models.api.check_record_compatibility",
        lambda r: CompatibilityReport(compatible=True, reason="ok"),
    )
    return final_record


@pytest.fixture(autouse=True)
def _clear_server_caches(monkeypatch):
    from server_models import api as server_api

    server_api._availability_cache.clear()
    server_api._bundle_cache.clear()
    _enable(monkeypatch)
    yield
    server_api._availability_cache.clear()
    server_api._bundle_cache.clear()


def test_openapi_includes_server_paths_unconditionally():
    response = CLIENT.get("/openapi.json")
    paths = response.json()["paths"]
    assert "/api/v1/server-forecasts/availability" in paths
    assert "/api/v1/server-forecasts/{ticker}" in paths


def test_availability_reports_disabled(monkeypatch):
    _enable(monkeypatch, enabled=False)
    response = CLIENT.get("/api/v1/server-forecasts/availability")
    assert response.status_code == 200
    data = response.json()
    assert data["enabled"] is False
    assert data["mode"] == "browser_only"
    assert data["tickers"] == []


def test_availability_reports_fresh_when_promoted(monkeypatch):
    record = make_record()
    registry = MagicMock()
    registry.get_promoted.return_value = record
    _stub_serve_deps(monkeypatch, registry=registry)

    response = CLIENT.get("/api/v1/server-forecasts/availability")
    assert response.status_code == 200
    data = response.json()
    assert data["enabled"] is True
    assert data["mode"] == "hybrid"
    assert len(data["tickers"]) == 1
    assert data["tickers"][0]["ticker"] == "AAPL"
    assert data["tickers"][0]["status"] == "fresh"
    assert data["tickers"][0]["version_id"] == record.key.version_id


def test_forecast_disabled_fallback(monkeypatch):
    _enable(monkeypatch, enabled=False)
    response = CLIENT.get("/api/v1/server-forecasts/AAPL")
    assert response.status_code == 200
    assert response.json()["reason"] == "disabled"


def test_forecast_not_in_allowlist_fallback(monkeypatch):
    _enable(monkeypatch, allowlist=("MSFT",))
    response = CLIENT.get("/api/v1/server-forecasts/AAPL")
    assert response.status_code == 200
    assert response.json()["reason"] == "disabled"


def test_forecast_direction_is_unsupported_and_browser_falls_back(monkeypatch):
    record = make_record(forecast_type="direction")
    _happy_path(monkeypatch, record)
    response = CLIENT.get("/api/v1/server-forecasts/AAPL?forecast_type=trend")
    assert response.status_code == 200
    data = response.json()
    assert data["available"] is False
    assert data["reason"] == "unsupported_forecast_type"
    assert "predicted_prices" not in data


def test_forecast_missing_fallback_in_hybrid(monkeypatch):
    _enable(monkeypatch, mode="hybrid")
    registry = MagicMock()
    registry.get_promoted.return_value = None
    _stub_serve_deps(monkeypatch, registry=registry)

    response = CLIENT.get("/api/v1/server-forecasts/AAPL")
    assert response.status_code == 200
    assert response.json()["reason"] == "missing"


def test_forecast_stale_fallback_in_hybrid(monkeypatch):
    record = make_record()
    registry = MagicMock()
    registry.get_promoted.return_value = record
    _stub_serve_deps(monkeypatch, registry=registry)
    monkeypatch.setattr("server_models.api.is_fresh", lambda *a, **k: False)

    response = CLIENT.get("/api/v1/server-forecasts/AAPL")
    assert response.status_code == 200
    assert response.json()["reason"] == "stale"


def test_forecast_incompatible_fallback_in_hybrid(monkeypatch):
    record = make_record()
    registry = MagicMock()
    registry.get_promoted.return_value = record
    _stub_serve_deps(monkeypatch, registry=registry)
    monkeypatch.setattr(
        "server_models.api.check_record_compatibility",
        lambda r: CompatibilityReport(compatible=False, reason="flagged"),
    )

    response = CLIENT.get("/api/v1/server-forecasts/AAPL")
    assert response.status_code == 200
    assert response.json()["reason"] == "incompatible"


def test_forecast_missing_is_503_in_server_pretrained_mode(monkeypatch):
    _enable(monkeypatch, mode="server_pretrained")
    registry = MagicMock()
    registry.get_promoted.return_value = None
    _stub_serve_deps(monkeypatch, registry=registry)

    response = CLIENT.get("/api/v1/server-forecasts/AAPL")
    assert response.status_code == 503


def test_forecast_registry_unavailable_is_503_in_any_mode(monkeypatch):
    _stub_serve_deps(monkeypatch, registry=_BrokenRegistry())

    response = CLIENT.get("/api/v1/server-forecasts/AAPL")
    assert response.status_code == 503


def _compat_ok(monkeypatch):
    monkeypatch.setattr(
        "server_models.api.check_record_compatibility",
        lambda r: CompatibilityReport(compatible=True, reason="ok"),
    )


def test_forecast_success_returns_canonical_bundle(monkeypatch):
    record = make_record()
    _bundle, _digest, payload, final_record = make_bundle(record)
    registry = MagicMock()
    registry.get_promoted.return_value = final_record
    storage = MagicMock()
    storage.get_bundle.return_value = payload
    _stub_serve_deps(monkeypatch, registry=registry, storage=storage)
    _compat_ok(monkeypatch)

    response = CLIENT.get("/api/v1/server-forecasts/AAPL?days=7")
    assert response.status_code == 200
    data = response.json()
    assert data["available"] is True
    assert data["ticker"] == "AAPL"
    assert data["forecast_days"] == 7
    assert len(data["predicted_prices"]) == 7
    assert len(data["future_dates"]) == 7
    assert len(data["historical_prices"]) == HISTORY_DISPLAY_WINDOW
    assert len(data["historical_dates"]) == HISTORY_DISPLAY_WINDOW
    assert data["metadata"]["engine"]["role"] == "server_pretrained"
    assert data["metadata"]["engine"]["version_id"] == final_record.key.version_id
    assert data["metadata"]["authenticity"] == "sha256_only"
    assert data["metadata"]["origin"]["date"] == "2026-07-31"
    assert data["metadata"]["origin"]["close"] == 150.0
    assert data["metrics"]["metric_source"] == "server_purged_walk_forward"


def test_forecast_success_verifies_ed25519_when_key_configured(monkeypatch):
    private_key = Ed25519PrivateKey.generate()
    signer = Ed25519ManifestSigner(private_key)
    record = make_record()
    _bundle, _digest, payload, final_record = make_bundle(record, signer=signer)
    registry = MagicMock()
    registry.get_promoted.return_value = final_record
    storage = MagicMock()
    storage.get_bundle.return_value = payload
    _stub_serve_deps(
        monkeypatch,
        registry=registry,
        storage=storage,
        verifier=Ed25519ManifestVerifier(private_key.public_key()),
    )
    _compat_ok(monkeypatch)

    response = CLIENT.get("/api/v1/server-forecasts/AAPL?days=7")
    assert response.status_code == 200
    assert response.json()["metadata"]["authenticity"] == "ed25519_verified"


def test_forecast_tampered_bundle_fails_closed(monkeypatch):
    private_key = Ed25519PrivateKey.generate()
    signer = Ed25519ManifestSigner(private_key)
    record = make_record()
    _bundle, _digest, payload, final_record = make_bundle(record, signer=signer)
    registry = MagicMock()
    registry.get_promoted.return_value = final_record
    storage = MagicMock()
    tampered = bytearray(payload)
    tampered[0] ^= 0xFF
    storage.get_bundle.return_value = bytes(tampered)
    _stub_serve_deps(
        monkeypatch,
        registry=registry,
        storage=storage,
        verifier=Ed25519ManifestVerifier(private_key.public_key()),
    )
    _compat_ok(monkeypatch)

    response = CLIENT.get("/api/v1/server-forecasts/AAPL")
    assert response.status_code == 503
    assert "verification" in response.json()["detail"]


def test_forecast_digest_mismatch_fails_closed(monkeypatch):
    record = make_record()
    _bundle, _digest, payload, _final_record = make_bundle(record)
    registry = MagicMock()
    registry.get_promoted.return_value = record.model_copy(update={"sha256_digest": "f" * 64})
    storage = MagicMock()
    storage.get_bundle.return_value = payload
    _stub_serve_deps(monkeypatch, registry=registry, storage=storage)
    _compat_ok(monkeypatch)

    response = CLIENT.get("/api/v1/server-forecasts/AAPL")
    assert response.status_code == 503
    assert "digest" in response.json()["detail"]
