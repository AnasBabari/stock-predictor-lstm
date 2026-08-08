"""Server-bundle retention preserves live/rollback pointers and registry history."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

import pytest

from config import FEATURES_V4, TARGET_MODE, WINDOW_SIZE
from server_models.contracts import (
    ReproducibilityMetadata,
    RobustScalerParams,
    ServerArtifactKey,
    ServerModelRecord,
)
from server_models.db import InMemoryRegistry, ModelRegistryError, PostgresRegistry
from server_models.retention import sweep_expired_bundles
from server_models.storage import InMemoryObjectStore


def _record(version_day: int) -> ServerModelRecord:
    trained_at = datetime(2026, 1, version_day, 12, tzinfo=UTC)
    key = ServerArtifactKey.create(
        ticker="AAPL",
        snapshot_id=f"sha256:{version_day:064x}",
        trained_at=trained_at,
        git_commit="0123456789ab",
    )
    reproducibility = ReproducibilityMetadata(
        feature_names=list(FEATURES_V4),
        window_size=WINDOW_SIZE,
        target_mode=TARGET_MODE,
        scaler=RobustScalerParams(medians=[0.0] * len(FEATURES_V4), iqrs=[1.0] * len(FEATURES_V4)),
        python_version="3.11",
        git_commit="0123456789ab",
    )
    return ServerModelRecord(
        key=key,
        reproducibility=reproducibility,
        sha256_digest=hashlib.sha256(key.version_id.encode()).hexdigest(),
    )


def test_retention_prunes_only_expired_non_pointer_objects_and_keeps_rows():
    registry = InMemoryRegistry()
    storage = InMemoryObjectStore()
    previous, current, expired_candidate = (_record(day) for day in (1, 2, 3))
    for record in (previous, current, expired_candidate):
        registry.insert_artifact(record)
        storage.put_bundle(record.key.version_id, b"{}")
    registry.promote(previous.key.version_id)
    registry.promote(current.key.version_id)

    pruned = sweep_expired_bundles(
        registry,
        storage,
        retention_days=30,
        now=datetime.now(UTC) + timedelta(days=31),
    )

    assert pruned == [expired_candidate.key.version_id]
    assert storage.bundle_exists(previous.key.version_id)
    assert storage.bundle_exists(current.key.version_id)
    assert not storage.bundle_exists(expired_candidate.key.version_id)
    assert len(registry.list_artifacts("AAPL")) == 3
    assert registry.read_audit_log()[-1]["event"] == "artifact_bundles_pruned"
    with pytest.raises(ModelRegistryError, match="pruned"):
        registry.promote(expired_candidate.key.version_id)


def test_retention_reclaims_demoted_champion_only_after_rollback_pointer_moves():
    registry = InMemoryRegistry()
    storage = InMemoryObjectStore()
    previous, current = (_record(day) for day in (1, 2))
    for record in (previous, current):
        registry.insert_artifact(record)
        storage.put_bundle(record.key.version_id, b"{}")
    registry.promote(previous.key.version_id)
    registry.promote(current.key.version_id)
    registry.rollback("AAPL")

    pruned = sweep_expired_bundles(
        registry,
        storage,
        retention_days=30,
        now=datetime.now(UTC) + timedelta(days=31),
    )

    assert pruned == [current.key.version_id]
    assert storage.bundle_exists(previous.key.version_id)
    assert not storage.bundle_exists(current.key.version_id)


class _RetentionCursor:
    def __init__(self, version_ids: list[str]):
        self.version_ids = version_ids
        self.executed: list[str] = []
        self.params: list[object] = []

    def execute(self, sql, params=None):
        self.executed.append(sql)
        self.params.append(params)

    def fetchall(self):
        return [(version_id,) for version_id in self.version_ids]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _RetentionConnection:
    def __init__(self, version_ids: list[str]):
        self._cursor = _RetentionCursor(version_ids)
        self.commits = 0
        self.rollbacks = 0

    def cursor(self):
        return self._cursor

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


def test_postgres_retention_locks_pointer_snapshot_before_deleting_objects():
    version_id = _record(1).key.version_id
    connection = _RetentionConnection([version_id])
    deleted: list[str] = []

    result = PostgresRegistry(conn=connection).prune_bundle_objects(
        datetime(2026, 2, 1, tzinfo=UTC), deleted.append
    )

    statements = connection._cursor.executed
    assert result == deleted == [version_id]
    assert "server_promotions" in statements[0] and "FOR UPDATE" in statements[0]
    assert "bundle_pruned_at IS NULL" in statements[1]
    assert "previous_version" in statements[1] and "current_version" in statements[1]
    assert (
        statements.index(next(sql for sql in statements if "bundle_pruned_at = now()" in sql)) > 1
    )
    assert connection.commits == 1
    assert connection.rollbacks == 0
