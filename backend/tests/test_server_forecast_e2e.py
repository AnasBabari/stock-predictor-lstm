"""In-process E2E for the hybrid server path.

Covers the full lifecycle over real contracts and HTTP: deterministic snapshot
-> walk-forward gates -> train -> Ed25519 signature -> immutable bundle ->
promote -> serve (verified on the wire) -> replacement promotion -> rollback ->
tamper-fails-closed. Uses in-memory registry/storage so it runs in CI without
infrastructure, complementing tests/test_server_models_integration.py which
repeats the same lifecycle against real Postgres + MinIO.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi.testclient import TestClient

from api import app
from artifacts.signing import Ed25519ManifestSigner
from config import FEATURES_V4, MAX_FORECAST_DAYS
from server_models.contracts import FORECAST_LENGTH, HISTORY_DISPLAY_WINDOW, WINDOW_SIZE
from server_models.db import InMemoryRegistry
from server_models.storage import InMemoryObjectStore
from server_models.training import train_server_forecast

CLIENT = TestClient(app)

FIXTURE_LAST_DATE = date(2025, 1, 31)
ROWS = WINDOW_SIZE + HISTORY_DISPLAY_WINDOW + 40


class RecordingElasticNet:
    instances: list["RecordingElasticNet"] = []

    def __init__(self) -> None:
        RecordingElasticNet.instances.append(self)

    def fit(self, features, targets):
        return self

    def predict(self, features) -> np.ndarray:
        return np.full((features.shape[0], FORECAST_LENGTH), 0.002)


def _fixture(snapshot_suffix: str):
    rng = np.random.default_rng(int(snapshot_suffix, 36) + 1)
    df = pd.DataFrame(rng.standard_normal((ROWS, len(FEATURES_V4))), columns=FEATURES_V4)
    closes = 100.0 + rng.uniform(-1.5, 1.5, ROWS)
    dates = pd.date_range(start="2024-04-01", periods=ROWS, freq="B")
    assert dates[-1].date() == FIXTURE_LAST_DATE

    def fetch(ticker):
        return df, closes, dates, {"snapshot_id": f"sha256:{'a' * 60}{snapshot_suffix}"}

    return fetch


def _passing_run(*args, **kwargs):
    return {
        "models": {
            "elastic_net": {
                "promotion": {"promoted": True},
                "aggregate": {
                    "pooled": {"relative_rmse": 0.85, "relative_mae": 0.9},
                    "per_horizon": {
                        str(h): {"relative_rmse": 0.9} for h in range(1, MAX_FORECAST_DAYS + 1)
                    },
                },
            }
        }
    }


def _configure(monkeypatch, *, fetch, public_key_path):
    import config
    from server_models import api as server_api

    monkeypatch.setattr("server_models.training.fetch_browser_data", fetch)
    monkeypatch.setattr("server_models.training.run_baseline_experiment", _passing_run)
    monkeypatch.setattr("server_models.training.ElasticNetForecaster", RecordingElasticNet)

    monkeypatch.setattr(config.settings, "server_forecast_serving_enabled", True)
    monkeypatch.setattr(config.settings, "training_mode", "hybrid")
    monkeypatch.setattr(config.settings, "server_forecast_allowlist", ["AAPL"])
    monkeypatch.setattr(config.settings, "server_forecast_public_key_path", public_key_path)
    server_api._bundle_cache.clear()
    server_api._availability_cache.clear()


def _write_keys(tmp_path):
    private_key = Ed25519PrivateKey.generate()
    priv_path = tmp_path / "private.pem"
    pub_path = tmp_path / "public.pem"
    priv_path.write_bytes(
        private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    pub_path.write_bytes(
        private_key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
    )
    return priv_path, pub_path


@pytest.fixture(autouse=True)
def _clear_http_caches():
    from server_models import api as server_api

    server_api._bundle_cache.clear()
    server_api._availability_cache.clear()
    yield
    server_api._bundle_cache.clear()
    server_api._availability_cache.clear()


def test_full_lifecycle_train_promote_serve_replacement_rollback(monkeypatch, tmp_path):
    priv_path, pub_path = _write_keys(tmp_path)
    signer = Ed25519ManifestSigner.from_pem_file(priv_path)
    registry = InMemoryRegistry()
    storage = InMemoryObjectStore()
    _configure(monkeypatch, fetch=_fixture("01"), public_key_path=str(pub_path))
    monkeypatch.setattr("server_models.api.get_registry", lambda: registry)
    monkeypatch.setattr("server_models.api.get_storage", lambda: storage)

    # 1) First promotion: train with a real (ephemeral) signer.
    first = train_server_forecast("AAPL", registry, storage, signer)
    assert first is not None and first.key.forecast_type == "price"
    assert registry.get_promoted("AAPL").key.version_id == first.key.version_id

    # 2) Serve it over HTTP with ed25519 verification enabled.
    response = CLIENT.get("/api/v1/server-forecasts/AAPL?days=7")
    assert response.status_code == 200
    data = response.json()
    assert data["available"] is True
    assert data["metadata"]["engine"]["version_id"] == first.key.version_id
    assert data["metadata"]["authenticity"] == "ed25519_verified"
    assert data["metadata"]["origin"]["date"] == FIXTURE_LAST_DATE.isoformat()
    # The first future session must be strictly after the origin session.
    assert data["future_dates"][0] > data["metadata"]["origin"]["date"]
    assert len(data["predicted_prices"]) == 7
    assert len(data["historical_prices"]) == HISTORY_DISPLAY_WINDOW
    assert data["historical_prices"][-1] == pytest.approx(data["metadata"]["origin"]["close"])
    assert data["metrics"]["metric_source"] == "server_purged_walk_forward"
    assert data["metadata"]["engine"]["role"] == "server_pretrained"

    # 3) Replacement promotion on a new snapshot.
    _configure(monkeypatch, fetch=_fixture("two"), public_key_path=str(pub_path))
    monkeypatch.setattr("server_models.api.get_registry", lambda: registry)
    monkeypatch.setattr("server_models.api.get_storage", lambda: storage)
    second = train_server_forecast("AAPL", registry, storage, signer)
    assert second is not None and second.key.version_id != first.key.version_id
    assert registry.get_promoted("AAPL").key.version_id == second.key.version_id

    resp2 = CLIENT.get("/api/v1/server-forecasts/AAPL?days=7")
    assert resp2.status_code == 200
    assert resp2.json()["metadata"]["engine"]["version_id"] == second.key.version_id
    assert resp2.json()["metadata"]["origin"]["date"] == FIXTURE_LAST_DATE.isoformat()

    # 4) Rollback restores the previous champion.
    restored = registry.rollback("AAPL")
    assert restored.key.version_id == first.key.version_id
    assert registry.get_promoted("AAPL").key.version_id == first.key.version_id


def test_e2e_tampered_bundle_fails_closed(monkeypatch, tmp_path):
    priv_path, pub_path = _write_keys(tmp_path)
    signer = Ed25519ManifestSigner.from_pem_file(priv_path)
    registry = InMemoryRegistry()
    storage = InMemoryObjectStore()
    _configure(monkeypatch, fetch=_fixture("tamper"), public_key_path=str(pub_path))
    monkeypatch.setattr("server_models.api.get_registry", lambda: registry)
    monkeypatch.setattr("server_models.api.get_storage", lambda: storage)

    record = train_server_forecast("AAPL", registry, storage, signer)
    stored = bytearray(storage.get_bundle(record.key.version_id))
    stored[0] ^= 0xFF
    storage._objects[storage.bundle_key(record.key.version_id)] = bytes(stored)

    resp = CLIENT.get("/api/v1/server-forecasts/AAPL")
    assert resp.status_code == 503
    assert "verification" in resp.json()["detail"]
