"""Postgres + MinIO integration lifecycle for the hybrid server path.

Runs only in ``.github/workflows/server-models-e2e.yml`` (or local dev) when the
``SERVER_E2E_*`` environment variables are present. Mirrors the in-process
coverage of ``tests/test_server_forecast_e2e.py`` against real infrastructure:
schema bootstrap, SKIP LOCKED queue, immutable inserts, atomic promotion with
saved previous pointer, replacement promotion, rollback, block-on-reject,
bundle round-trip + immutability, and the full HTTP serve path including a
fail-closed tamper check.
"""

import hashlib
import os
from datetime import date, datetime, timedelta

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi.testclient import TestClient

from api import app
from artifacts.signing import Ed25519ManifestSigner
from server_models.contracts import (
    FORECAST_LENGTH,
    HISTORY_DISPLAY_WINDOW,
    ReproducibilityMetadata,
    RobustScalerParams,
    ServerArtifactKey,
    ServerForecastBundle,
    ServerModelRecord,
)

DATABASE_URL = os.environ.get("SERVER_E2E_DATABASE_URL")
S3_BUCKET = os.environ.get("SERVER_E2E_S3_BUCKET")
S3_ENDPOINT = os.environ.get("SERVER_E2E_S3_ENDPOINT")
S3_PREFIX = os.environ.get("SERVER_E2E_S3_PREFIX", "artifacts")

pytestmark = pytest.mark.skipif(
    not DATABASE_URL or not S3_BUCKET,
    reason="Postgres/MinIO E2E environment not configured",
)


def _require_env() -> tuple[str, str, str | None]:
    assert DATABASE_URL and S3_BUCKET
    return DATABASE_URL, S3_BUCKET, S3_ENDPOINT


CLIENT = TestClient(app)


def _make_record(ticker="AAPL", *, forecast_type="price", snapshot="sha256:e2e01"):
    from config import FEATURES_V4

    key = ServerArtifactKey.create(ticker=ticker, snapshot_id=snapshot, forecast_type=forecast_type)
    scaler = RobustScalerParams(medians=[0.0] * 28, iqrs=[1.0] * 28)
    repro = ReproducibilityMetadata(
        feature_names=FEATURES_V4,
        scaler=scaler,
        python_version="3.12",
        git_commit="unknown",
    )
    return ServerModelRecord(key=key, reproducibility=repro, sha256_digest="0" * 64)


def _make_bundle(version_id: str):
    return ServerForecastBundle(
        ticker="AAPL",
        version_id=version_id,
        origin_close=150.0,
        origin_date=date(2026, 7, 31),
        future_dates=[date(2026, 8, 1) + timedelta(days=i) for i in range(FORECAST_LENGTH)],
        predicted_log_returns=[0.001] * FORECAST_LENGTH,
        predicted_prices=[100.0 + i for i in range(FORECAST_LENGTH)],
        historical_dates=[
            date(2026, 7, 31) - timedelta(days=HISTORY_DISPLAY_WINDOW - 1 - i)
            for i in range(HISTORY_DISPLAY_WINDOW)
        ],
        historical_prices=[
            150.0 - (HISTORY_DISPLAY_WINDOW - 1 - i) * 0.25 for i in range(HISTORY_DISPLAY_WINDOW)
        ],
        evidence={"metric_source": "server_purged_walk_forward", "family": "elastic_net"},
        generated_at=datetime(2026, 7, 31, 12, 0, 0),
    )


def _sign(record, bundle_bytes, signer) -> ServerModelRecord:
    return record.model_copy(
        update={
            "signature": signer(bundle_bytes),
            "sha256_digest": hashlib.sha256(bundle_bytes).hexdigest(),
        }
    )


@pytest.fixture
def database():
    from server_models.db import PostgresRegistry

    url, _bucket, _endpoint = _require_env()
    registry = PostgresRegistry(database_url=url)
    registry.init_schema()
    with registry._conn.cursor() as cursor:
        cursor.execute(
            "TRUNCATE server_artifacts, server_promotions, training_jobs, audit_log "
            "RESTART IDENTITY CASCADE"
        )
    registry._conn.commit()
    return registry


@pytest.fixture
def object_store():
    from server_models.storage import S3ObjectStore

    _url, bucket, endpoint = _require_env()
    store = S3ObjectStore(bucket=bucket, prefix=S3_PREFIX, endpoint_url=endpoint)
    store.ensure_bucket()
    return store


def test_registry_queue_and_promotion_lifecycle(database):
    from server_models.db import ModelRegistryError

    v1 = _make_record("AAPL", snapshot="integration-v1")
    v2 = _make_record("AAPL", snapshot="integration-v2")
    v3 = _make_record("AAPL", snapshot="integration-v3")

    # Queue: FIFO claim, then completion.
    job_id = database.enqueue_job("AAPL", payload={"days": 7})
    claimed = database.dequeue_job()
    assert claimed["id"] == job_id and claimed["attempts"] == 1
    database.complete_job(job_id)
    assert database.dequeue_job() is None

    # Immutable insert + promotion.
    database.insert_artifact(v1)
    database.insert_artifact(v2)
    database.promote(v1.key.version_id)
    assert database.get_promoted("AAPL").key.version_id == v1.key.version_id

    # Replacement promotion saves the previous pointer.
    database.promote(v2.key.version_id)
    assert database.get_promoted("AAPL").key.version_id == v2.key.version_id

    audits = {e["event"] for e in database.read_audit_log()}
    assert "artifact_promoted" in audits
    promoted = [e for e in database.read_audit_log() if e["event"] == "artifact_promoted"][-1]
    assert promoted["details"]["previous_version"] == v1.key.version_id

    # Rollback restores the previous champion.
    restored = database.rollback("AAPL")
    assert restored.key.version_id == v1.key.version_id
    assert database.get_promoted("AAPL").key.version_id == v1.key.version_id

    # A rejected candidate can never be promoted.
    database.insert_artifact(v3)
    database.reject(v3.key.version_id, "failed gates")
    with pytest.raises(ModelRegistryError, match="rejected"):
        database.promote(v3.key.version_id)


def test_object_store_round_trip_and_immutability(object_store):
    from server_models.storage import ObjectStoreError

    version_id = _make_record("AAPL", snapshot="storage-v1").key.version_id
    key = object_store.put_bundle(version_id, b'{"v": 1}')
    assert object_store.bundle_exists(version_id)
    assert object_store.get_bundle(version_id) == b'{"v": 1}'
    with pytest.raises(ObjectStoreError, match="immutable"):
        object_store.put_bundle(version_id, b"tampered")
    assert object_store.get_bundle(version_id) == b'{"v": 1}'
    assert key.startswith(S3_PREFIX)


def test_full_http_path_with_real_postgres_minio_and_keys(
    database, object_store, tmp_path, monkeypatch
):
    from server_models import api as server_api

    private_key = Ed25519PrivateKey.generate()
    pub_path = tmp_path / "public.pem"
    pub_path.write_bytes(
        private_key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
    )

    record = _make_record("AAPL", snapshot="http-v1")
    bundle = _make_bundle(record.key.version_id)
    signer = Ed25519ManifestSigner(private_key)
    payload = bundle.model_dump_json().encode("utf-8")
    record = _sign(record, payload, signer)

    object_store.put_bundle(record.key.version_id, payload)
    database.insert_artifact(record)
    database.promote(record.key.version_id)

    import config

    url, bucket, endpoint = _require_env()
    monkeypatch.setattr(config.settings, "server_forecast_serving_enabled", True)
    monkeypatch.setattr(config.settings, "training_mode", "hybrid")
    monkeypatch.setattr(config.settings, "server_forecast_allowlist", ["AAPL"])
    monkeypatch.setattr(config.settings, "server_forecast_public_key_path", str(pub_path))
    monkeypatch.setattr(config.settings, "registry_database_url", url)
    monkeypatch.setattr(config.settings, "s3_bucket", bucket)
    monkeypatch.setattr(config.settings, "s3_endpoint_url", endpoint)
    monkeypatch.setattr(config.settings, "s3_key_prefix", S3_PREFIX)
    server_api._bundle_cache.clear()
    server_api._availability_cache.clear()

    response = CLIENT.get("/api/v1/server-forecasts/AAPL?days=7")
    assert response.status_code == 200
    data = response.json()
    assert data["available"] is True
    assert data["metadata"]["engine"]["version_id"] == record.key.version_id
    assert data["metadata"]["authenticity"] == "ed25519_verified"
    assert data["metadata"]["origin"]["date"] == "2026-07-31"
    assert len(data["predicted_prices"]) == 7
    assert len(data["historical_prices"]) == HISTORY_DISPLAY_WINDOW

    # Fail closed: tampering the bundle in MinIO yields 503, never a fallback.
    tampered = bytearray(payload)
    tampered[0] ^= 0xFF
    object_store.put(object_store.bundle_key(record.key.version_id), bytes(tampered))
    server_api._bundle_cache.clear()
    response = CLIENT.get("/api/v1/server-forecasts/AAPL?days=7")
    assert response.status_code == 503
