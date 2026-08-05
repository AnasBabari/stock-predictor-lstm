"""Foundation tests for backend/server_models (plan items 1-7)."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime, timedelta

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import ValidationError

from artifacts.signing import Ed25519ManifestSigner, Ed25519ManifestVerifier
from config import FEATURES_V4, TARGET_MODE, WINDOW_SIZE, Settings
from server_models.compatibility import check_record_compatibility, is_fresh
from server_models.contracts import (
    FORECAST_LENGTH,
    ReproducibilityMetadata,
    RobustScalerParams,
    ServerArtifactKey,
    ServerForecastBundle,
    ServerModelRecord,
    make_version_id,
)
from server_models.db import (
    SCHEMA_SQL,
    InMemoryRegistry,
    ModelRegistryError,
)
from server_models.signing_manifests import (
    ServerArtifactIntegrityError,
    sign_bundle,
    verify_bundle,
)
from server_models.storage import InMemoryObjectStore, ObjectStoreError, S3ObjectStore

GIT_SHA = "0123456789ab"
TRAINED_AT = datetime(2026, 1, 2, 22, 0, 0, tzinfo=UTC)
# New identity format: {ticker}-{forecast_type}-{utc-ts}-{gitsha12}-{snapshot8}
V1 = "AAPL-price-20260102T220000Z-0123456789ab-00000aaa"
V2 = "AAPL-price-20260103T220000Z-0123456789ab-00000aaa"
MSFT_V1 = "MSFT-price-20260102T220000Z-0123456789ab-000000af"


def _key(ticker: str = "AAPL", version_id: str | None = None) -> ServerArtifactKey:
    return (
        ServerArtifactKey.create(
            ticker=ticker,
            snapshot_id=f"snap-{ticker}",
            trained_at=TRAINED_AT,
            git_commit=GIT_SHA,
        )
        if version_id is None
        else ServerArtifactKey(
            ticker=ticker,
            snapshot_id=f"snap-{ticker}",
            trained_at=TRAINED_AT,
            version_id=version_id,
        )
    )


def _record(ticker: str = "AAPL", version_id: str | None = None) -> ServerModelRecord:
    key = _key(ticker, version_id)
    reproducibility = ReproducibilityMetadata(
        feature_names=list(FEATURES_V4),
        window_size=WINDOW_SIZE,
        target_mode=TARGET_MODE,
        scaler=RobustScalerParams(medians=[0.0] * len(FEATURES_V4), iqrs=[1.0] * len(FEATURES_V4)),
        python_version="3.11",
        library_versions={"scikit-learn": "1.5.0"},
        git_commit=GIT_SHA,
        metrics={"1": {"mae": 0.01}},
    )
    digest = hashlib.sha256(b"bundle").hexdigest()
    return ServerModelRecord(key=key, reproducibility=reproducibility, sha256_digest=digest)


# ── contracts: version_id immutability & format ──────────────────────


def test_version_id_format_is_ticker_type_ts_gitsha_snapshot():
    key = _key("AAPL")
    assert key.version_id == "AAPL-price-20260102T220000Z-0123456789ab-00000aaa"


def test_version_id_includes_forecast_type_and_snapshot_hash():
    """Distinct forecast types or snapshots must never collide on one key."""
    a = ServerArtifactKey.create(
        ticker="AAPL",
        snapshot_id="abcdefabcdefabcdefabcdefabcdefabcd_1",
        trained_at=TRAINED_AT,
        git_commit=GIT_SHA,
    )
    b = ServerArtifactKey.create(
        ticker="AAPL",
        snapshot_id="abcdefabcdefabcdefabcdefabcdefabcd_2",
        trained_at=TRAINED_AT,
        git_commit=GIT_SHA,
    )
    c = ServerArtifactKey.create(
        ticker="AAPL",
        snapshot_id=a.snapshot_id,
        trained_at=TRAINED_AT,
        git_commit=GIT_SHA,
        forecast_type="direction",
    )
    assert a.version_id != b.version_id
    assert a.version_id != c.version_id
    assert "price" in a.version_id.split("-")
    assert "direction" in c.version_id.split("-")


def test_make_version_id_falls_back_to_unknown_without_git(monkeypatch):
    def _boom(*args, **kwargs):
        raise OSError("no git")

    monkeypatch.setattr("server_models.contracts.subprocess.check_output", _boom)
    assert make_version_id("MSFT", trained_at=TRAINED_AT).endswith("-unknown")


def test_version_id_is_deterministic_and_immutable():
    first = _key("AAPL")
    second = _key("AAPL")
    assert first.version_id == second.version_id
    with pytest.raises(ValidationError):
        first.version_id = "AAPL-mutated"


def test_server_artifact_key_rejects_bad_schema_version():
    with pytest.raises(ValidationError):
        ServerArtifactKey(
            ticker="AAPL",
            snapshot_id="s",
            trained_at=TRAINED_AT,
            version_id=V1,
            schema_version=3,
        )


def test_server_artifact_key_rejects_malformed_version_id():
    with pytest.raises(ValidationError):
        ServerArtifactKey(
            ticker="AAPL", snapshot_id="s", trained_at=TRAINED_AT, version_id="nonsense"
        )


def test_record_rejects_bad_digest():
    with pytest.raises(ValidationError):
        ServerModelRecord(
            key=_key("AAPL"),
            reproducibility=_record().reproducibility,
            sha256_digest="nothex",
        )


def _bundle(**overrides) -> ServerForecastBundle:
    payload = {
        "ticker": "AAPL",
        "forecast_type": "price",
        "version_id": V1,
        "origin_close": 100.0,
        "origin_date": date(2026, 1, 2),
        "future_dates": [date(2026, 1, 5) + timedelta(days=i) for i in range(FORECAST_LENGTH)],
        "predicted_log_returns": [0.001] * FORECAST_LENGTH,
        "predicted_prices": [100.0 + i for i in range(FORECAST_LENGTH)],
        "historical_dates": [date(2026, 1, 2) - timedelta(days=119 - i) for i in range(120)],
        "historical_prices": [100.0 - (119 - i) * 0.5 for i in range(120)],
        "evidence": {"relative_mae": 0.02},
        "generated_at": TRAINED_AT,
    }
    payload.update(overrides)
    return ServerForecastBundle(**payload)


def test_bundle_requires_exactly_30_steps():
    bundle = _bundle()
    assert len(bundle.predicted_prices) == FORECAST_LENGTH == 30
    with pytest.raises(ValidationError):
        _bundle(predicted_prices=[100.0] * 29)


def test_bundle_rejects_non_positive_prices():
    with pytest.raises(ValidationError):
        _bundle(predicted_prices=[-1.0] * FORECAST_LENGTH)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_bundle_rejects_non_finite_prices(value):
    with pytest.raises(ValidationError):
        _bundle(predicted_prices=[value] * FORECAST_LENGTH)
    with pytest.raises(ValidationError):
        _bundle(predicted_log_returns=[value] * FORECAST_LENGTH)


def test_bundle_rejects_non_increasing_future_dates():
    with pytest.raises(ValidationError):
        _bundle(future_dates=[date(2026, 1, 10)] * FORECAST_LENGTH)


def test_bundle_rejects_future_before_origin():
    with pytest.raises(ValidationError):
        _bundle(future_dates=[date(2026, 1, 2) + timedelta(days=i) for i in range(FORECAST_LENGTH)])


def test_bundle_rejects_historical_dates_not_ending_at_origin():
    with pytest.raises(ValidationError):
        _bundle(historical_dates=[date(2026, 1, 1) - timedelta(days=119 - i) for i in range(120)])


def test_bundle_rejects_last_historical_price_mismatching_origin_close():
    with pytest.raises(ValidationError):
        _bundle(historical_prices=[99.0 - (119 - i) * 0.5 for i in range(120)])


# ── signing round-trip + tamper rejection ────────────────────────────


def _keypair():
    private = Ed25519PrivateKey.generate()
    signer = Ed25519ManifestSigner(private)
    verifier = Ed25519ManifestVerifier(private.public_key())
    return signer, verifier


def test_sign_and_verify_round_trip():
    signer, verifier = _keypair()
    payload = json.dumps({"ticker": "AAPL"}).encode()
    manifest = sign_bundle(payload, signer)
    verify_bundle(payload, manifest, verifier)  # must not raise


def test_verify_rejects_tampered_payload():
    signer, verifier = _keypair()
    manifest = sign_bundle(b'{"ticker": "AAPL"}', signer)
    with pytest.raises(ServerArtifactIntegrityError, match="digest"):
        verify_bundle(b'{"ticker": "EVIL"}', manifest, verifier)


def test_verify_rejects_wrong_key():
    signer, _ = _keypair()
    _, other_verifier = _keypair()
    payload = b"bundle-bytes"
    manifest = sign_bundle(payload, signer)
    with pytest.raises(ServerArtifactIntegrityError, match="signature"):
        verify_bundle(payload, manifest, other_verifier)


def test_verify_fails_closed_without_signature():
    signer, verifier = _keypair()
    payload = b"bundle-bytes"
    manifest = sign_bundle(payload, signer)
    manifest.pop("signature")
    with pytest.raises(ServerArtifactIntegrityError):
        verify_bundle(payload, manifest, verifier)


def test_verify_rejects_unknown_manifest_schema():
    signer, verifier = _keypair()
    payload = b"bundle"
    manifest = sign_bundle(payload, signer)
    manifest["schema_version"] = 99
    with pytest.raises(ServerArtifactIntegrityError, match="schema"):
        verify_bundle(payload, manifest, verifier)


def test_sign_rejects_empty_payload():
    signer, _ = _keypair()
    with pytest.raises(ServerArtifactIntegrityError):
        sign_bundle(b"", signer)


# ── storage round-trip ───────────────────────────────────────────────


def test_in_memory_store_round_trip():
    store = InMemoryObjectStore()
    assert not store.exists("artifacts/v1/bundle.json")
    store.put("artifacts/v1/bundle.json", b"{}")
    assert store.exists("artifacts/v1/bundle.json")
    assert store.get("artifacts/v1/bundle.json") == b"{}"


def test_in_memory_store_missing_key_raises():
    store = InMemoryObjectStore()
    with pytest.raises(ObjectStoreError):
        store.get("nope")


def test_bundle_put_is_immutable_in_memory():
    store = InMemoryObjectStore()
    store.put_bundle("AAPL-v1", b"{}")
    with pytest.raises(ObjectStoreError, match="immutable"):
        store.put_bundle("AAPL-v1", b"tampered")
    assert store.get_bundle("AAPL-v1") == b"{}"


def test_s3_store_bundle_key_layout():
    store = S3ObjectStore(bucket="b", prefix="artifacts", client=object())
    assert store.bundle_key("AAPL-v1") == "artifacts/AAPL-v1/bundle.json"


def test_s3_store_requires_bucket():
    with pytest.raises(ObjectStoreError):
        S3ObjectStore(bucket="", client=object())


class _FakeS3NotFound(Exception):
    response = {"Error": {"Code": "404"}}


class _FakeS3PreconditionFailed(Exception):
    response = {"Error": {"Code": "PreconditionFailed"}}


class _FakeS3Client:
    def __init__(self):
        self.objects: dict[tuple[str, str], bytes] = {}
        self.created_buckets: set[str] = set()

    def put_object(self, Bucket, Key, Body, IfNoneMatch=None):
        if IfNoneMatch == "*" and (Bucket, Key) in self.objects:
            raise _FakeS3PreconditionFailed()
        self.objects[(Bucket, Key)] = bytes(Body)

    class _Body:
        def __init__(self, data):
            self._data = data

        def read(self):
            return self._data

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise _FakeS3NotFound()
        return {"Body": self._Body(self.objects[(Bucket, Key)])}

    def head_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise _FakeS3NotFound()
        return {}

    def create_bucket(self, Bucket):
        if Bucket in self.created_buckets:
            raise _FakeS3BucketAlreadyOwned()
        self.created_buckets.add(Bucket)


class _FakeS3BucketAlreadyOwned(Exception):
    response = {"Error": {"Code": "BucketAlreadyOwnedByYou"}}


def test_s3_store_round_trip_with_fake_client():
    fake = _FakeS3Client()
    store = S3ObjectStore(bucket="b", prefix="artifacts", client=fake)
    key = store.put_bundle("AAPL-v1", b'{"x": 1}')
    assert key == "artifacts/AAPL-v1/bundle.json"
    assert store.bundle_exists("AAPL-v1")
    assert store.get_bundle("AAPL-v1") == b'{"x": 1}'
    assert not store.bundle_exists("missing")


def test_s3_exists_propagates_non_404_errors():
    class _ExplodingHead(_FakeS3Client):
        def head_object(self, Bucket, Key):
            raise RuntimeError("endpoint down")

    store = S3ObjectStore(bucket="b", client=_ExplodingHead())
    with pytest.raises(ObjectStoreError):
        store.exists("whatever")


def test_s3_put_bundle_conditional_write_rejects_existing():
    fake = _FakeS3Client()
    store = S3ObjectStore(bucket="b", prefix="artifacts", client=fake)
    store.put_bundle("AAPL-v1", b"{}")
    with pytest.raises(ObjectStoreError, match="immutable"):
        store.put_bundle("AAPL-v1", b"tampered")
    assert store.get_bundle("AAPL-v1") == b"{}"


def test_s3_ensure_bucket_is_idempotent():
    fake = _FakeS3Client()
    store = S3ObjectStore(bucket="b", prefix="artifacts", client=fake)
    store.ensure_bucket()
    store.ensure_bucket()
    assert "b" in fake.created_buckets


# ── registry promote/keep-previous/rollback semantics ────────────────


def test_registry_promote_sets_pointer():
    registry = InMemoryRegistry()
    record = _record("AAPL", V1)
    registry.insert_artifact(record)
    assert registry.get_promoted("AAPL") is None
    registry.promote(V1)
    promoted = registry.get_promoted("AAPL")
    assert promoted is not None and promoted.status == "promoted"


def test_registry_insert_is_immutable():
    registry = InMemoryRegistry()
    registry.insert_artifact(_record("AAPL", V1))
    with pytest.raises(ModelRegistryError, match="already exists"):
        registry.insert_artifact(_record("AAPL", V1))


def test_registry_promote_keeps_previous_pointer():
    registry = InMemoryRegistry()
    registry.insert_artifact(_record("AAPL", V1))
    registry.insert_artifact(_record("AAPL", V2))
    registry.promote(V1)
    registry.promote(V2)
    assert registry.get_promoted("AAPL").key.version_id == V2


def test_registry_rollback_restores_previous():
    registry = InMemoryRegistry()
    registry.insert_artifact(_record("AAPL", V1))
    registry.insert_artifact(_record("AAPL", V2))
    registry.promote(V1)
    registry.promote(V2)
    restored = registry.rollback("AAPL")
    assert restored.key.version_id == V1
    assert registry.get_promoted("AAPL").key.version_id == V1


def test_registry_rollback_without_previous_raises():
    registry = InMemoryRegistry()
    with pytest.raises(ModelRegistryError, match="No previous"):
        registry.rollback("AAPL")


def test_registry_reject_blocks_promotion():
    registry = InMemoryRegistry()
    registry.insert_artifact(_record("AAPL", V1))
    registry.reject(V1, "failed gates")
    with pytest.raises(ModelRegistryError, match="rejected"):
        registry.promote(V1)


def test_registry_cannot_reject_promoted_artifact():
    registry = InMemoryRegistry()
    registry.insert_artifact(_record("AAPL", V1))
    registry.promote(V1)
    with pytest.raises(ModelRegistryError, match="roll back"):
        registry.reject(V1, "too late")


def test_registry_reject_unknown_raises():
    registry = InMemoryRegistry()
    with pytest.raises(ModelRegistryError, match="Unknown"):
        registry.reject("ghost", "nope")


def test_registry_list_artifacts_filters_by_ticker():
    registry = InMemoryRegistry()
    registry.insert_artifact(_record("AAPL", V1))
    registry.insert_artifact(_record("MSFT", MSFT_V1))
    tickers = [record.key.ticker for record in registry.list_artifacts("AAPL")]
    assert tickers == ["AAPL"]


def test_registry_audit_log_records_events():
    registry = InMemoryRegistry()
    registry.insert_artifact(_record("AAPL", V1))
    registry.promote(V1)
    registry.append_audit("custom_event", {"k": "v"})
    events = [entry["event"] for entry in registry.read_audit_log()]
    assert "artifact_inserted" in events
    assert "artifact_promoted" in events
    assert "custom_event" in events


def test_registry_promote_audit_records_previous_version():
    registry = InMemoryRegistry()
    registry.insert_artifact(_record("AAPL", V1))
    registry.insert_artifact(_record("AAPL", V2))
    registry.promote(V1)
    registry.promote(V2)
    promoted_event = [e for e in registry.read_audit_log() if e["event"] == "artifact_promoted"][-1]
    assert promoted_event["details"]["previous_version"] == V1


def test_registry_job_queue_fifo_and_dequeue_empty():
    registry = InMemoryRegistry()
    assert registry.dequeue_job() is None
    registry.enqueue_job("AAPL")
    registry.enqueue_job("MSFT", payload={"days": 7})
    first = registry.dequeue_job()
    assert first["ticker"] == "AAPL" and first["attempts"] == 1
    second = registry.dequeue_job()
    assert second["ticker"] == "MSFT" and second["payload"] == {"days": 7}
    assert registry.dequeue_job() is None


def test_schema_sql_contains_required_tables_and_partial_index():
    assert "CREATE TABLE IF NOT EXISTS server_artifacts" in SCHEMA_SQL
    assert "CREATE TABLE IF NOT EXISTS server_promotions" in SCHEMA_SQL
    assert "CREATE TABLE IF NOT EXISTS training_jobs" in SCHEMA_SQL
    assert "CREATE TABLE IF NOT EXISTS audit_log" in SCHEMA_SQL
    assert "WHERE status = 'promoted'" in SCHEMA_SQL
    assert "UNIQUE INDEX" in SCHEMA_SQL
    assert "SKIP LOCKED" not in SCHEMA_SQL  # queue semantics live in dequeue SQL


# ── Postgres promote: single transaction, demote-before-promote ───────


class _FakePsycopgCursor:
    def __init__(self, script):
        self._script = list(script)
        self.executed: list[str] = []
        self.executed_params: list[object] = []
        self.rowcount = 1

    def execute(self, sql, params=None):
        self.executed.append(sql)
        self.executed_params.append(params)

    def fetchone(self):
        return self._script.pop(0) if self._script else None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _FakePsycopgConn:
    def __init__(self, script):
        self._cursor = _FakePsycopgCursor(script)
        self.committed = 0
        self.rolled_back = 0
        self.closed = 0

    def cursor(self):
        return self._cursor

    def commit(self):
        self.committed += 1

    def rollback(self):
        self.rolled_back += 1

    def close(self):
        self.closed += 1


def _promote_sql_statements() -> list[str]:
    from server_models.db import PostgresRegistry

    record_json = _record("AAPL", V1).model_dump_json()
    conn = _FakePsycopgConn(script=[("AAPL", "price", record_json, "candidate"), (None,)])
    registry = PostgresRegistry(conn=conn)
    registry.promote(V1)
    return conn._cursor.executed


def test_postgres_promote_locks_pointer_with_for_update():
    statements = _promote_sql_statements()
    assert any("FOR UPDATE" in s and "server_promotions" in s for s in statements)


def test_postgres_promote_demotes_before_promoting():
    statements = _promote_sql_statements()
    candidate_update = next(s for s in statements if "SET status = 'candidate'" in s)
    promoted_update = next(s for s in statements if "SET status = 'promoted'" in s)
    assert statements.index(candidate_update) < statements.index(promoted_update)


def test_postgres_promote_saves_previous_version():
    statements = _promote_sql_statements()
    pointer_update = next(s for s in statements if "current_version = %s" in s)
    assert "previous_version = %s" in pointer_update


def test_postgres_jsonb_payload_as_dict_parses():
    # psycopg returns JSONB columns as Python dicts, not JSON strings; promote
    # must accept both through the shared record-payload helper.
    from server_models.db import PostgresRegistry

    record = _record("AAPL", V1)
    record_dict = record.model_dump(mode="json")
    conn = _FakePsycopgConn(script=[("AAPL", "price", record_dict, "candidate"), (None,)])
    registry = PostgresRegistry(conn=conn)
    promoted = registry.promote(V1)
    assert promoted.key.version_id == V1
    assert promoted.status == "promoted"
    assert promoted.reproducibility == record.reproducibility


def test_postgres_promote_unknown_version_rolls_back():
    from server_models.db import PostgresRegistry

    conn = _FakePsycopgConn(script=[None])
    registry = PostgresRegistry(conn=conn)
    with pytest.raises(ModelRegistryError, match="Unknown"):
        registry.promote("ghost")
    assert conn.rolled_back == 1


# ── Postgres rollback: demote-then-promote, single transaction ────────


def _rollback_sql_statements() -> list[str]:
    from server_models.db import PostgresRegistry

    conn = _FakePsycopgConn(script=[("CUR", "PREV")])
    registry = PostgresRegistry(conn=conn)
    with pytest.raises(ModelRegistryError):
        registry.rollback("AAPL")
    return conn._cursor.executed


def _rollback_sql_params() -> list[object]:
    from server_models.db import PostgresRegistry

    conn = _FakePsycopgConn(script=[("CUR", "PREV")])
    registry = PostgresRegistry(conn=conn)
    with pytest.raises(ModelRegistryError):
        registry.rollback("AAPL")
    return conn._cursor.executed_params


def test_postgres_rollback_locks_pointer_with_for_update():
    statements = _rollback_sql_statements()
    assert any("FOR UPDATE" in s and "server_promotions" in s for s in statements)


def test_postgres_rollback_demotes_champion_before_promoting_previous():
    statements = _rollback_sql_statements()
    demote = next(s for s in statements if "SET status = 'candidate'" in s)
    promote = next(s for s in statements if "SET status = 'promoted'" in s)
    assert statements.index(demote) < statements.index(promote)


def test_postgres_rollback_clears_previous_version_and_audits():
    statements = _rollback_sql_statements()
    pointer = next(s for s in statements if "server_promotions" in s and "SET" in s)
    assert "previous_version = NULL" in pointer
    params = _rollback_sql_params()
    assert any("artifact_rollback" in p for p in params if p is not None)


def test_postgres_rollback_rolls_back_transaction_on_unexpected_error():
    from server_models.db import PostgresRegistry

    class _ExplodingCursor(_FakePsycopgCursor):
        def execute(self, sql, params=None):
            if sql.startswith("UPDATE server_artifacts SET status = 'candidate'"):
                raise RuntimeError("db exploded")
            return super().execute(sql, params)

    conn = _FakePsycopgConn(script=[("CUR", "PREV")])
    conn._cursor = _ExplodingCursor(conn._cursor._script)
    registry = PostgresRegistry(conn=conn)
    with pytest.raises(RuntimeError, match="db exploded"):
        registry.rollback("AAPL")
    assert conn.rolled_back == 1
    assert conn.committed == 0


def test_postgres_registry_close_closes_connection():
    from server_models.db import PostgresRegistry

    conn = _FakePsycopgConn(script=[])
    registry = PostgresRegistry(conn=conn)
    registry.close()
    assert conn.closed == 1


# ── compatibility: fresh/stale/incompatible ──────────────────────────


def test_compatible_record_passes():
    report = check_record_compatibility(_record("AAPL"))
    assert report.compatible is True


def test_feature_order_mismatch_is_incompatible():
    record = _record("AAPL")
    bad = list(FEATURES_V4)
    bad[0], bad[1] = bad[1], bad[0]
    record.reproducibility = record.reproducibility.model_copy(update={"feature_names": bad})
    report = check_record_compatibility(record)
    assert report.compatible is False and "FEATURES_V4" in report.reason


def test_wrong_schema_is_incompatible():
    record = _record("AAPL")
    object.__setattr__(record.key, "schema_version", 3)
    report = check_record_compatibility(record)
    assert report.compatible is False and "schema" in report.reason


def test_fresh_and_stale_thresholds():
    now = datetime.now(UTC)
    assert is_fresh(now - timedelta(hours=1), max_age_hours=36) is True
    assert is_fresh(now - timedelta(hours=40), max_age_hours=36) is False


def test_is_fresh_accepts_naive_utc():
    naive = datetime.now(UTC).replace(tzinfo=None)
    assert is_fresh(naive, max_age_hours=36) is True


# ── config flags default OFF / additive ──────────────────────────────


def test_new_settings_default_to_off():
    settings = Settings(
        _env_file=None,
        server_forecast_serving_enabled=False,
        server_training_enabled=False,
    )
    assert settings.server_forecast_serving_enabled is False
    assert settings.server_training_enabled is False
    assert settings.server_forecast_allowlist == []
    assert settings.server_forecast_max_age_hours == 36
    assert settings.server_forecast_cache_ttl == 900
    assert settings.registry_database_url is None
    assert settings.s3_bucket is None
    assert settings.s3_key_prefix == "artifacts"
    assert settings.training_mode == "browser_only"


def test_allowlist_parses_comma_separated_string():
    settings = Settings(_env_file=None, server_forecast_allowlist="aapl, MSFT,spy")
    assert settings.server_forecast_allowlist == ["AAPL", "MSFT", "SPY"]


def test_training_mode_rejects_invalid_value():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, training_mode="not_a_mode")
