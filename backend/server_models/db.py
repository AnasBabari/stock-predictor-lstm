"""Server artifact registry with atomic promotion pointers and rollback.

Two implementations share the :class:`ModelRegistry` contract:

* :class:`InMemoryRegistry` — thread-safe dict-backed registry for unit tests.
* :class:`PostgresRegistry` — psycopg 3 sync registry for production, with a
  ``SKIP LOCKED`` training-job queue.  Channel semantics (candidate/promoted/
  rejected, atomic pointer swap, rollback) mirror ``backend/artifacts/registry.py``.
"""

from __future__ import annotations

import json
import threading
import uuid
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any

from server_models.contracts import ServerModelRecord


class ModelRegistryError(RuntimeError):
    """A registry transition or lookup failed."""


class ModelRegistry(ABC):
    """Contract for immutable server-artifact lifecycle management."""

    @abstractmethod
    def insert_artifact(self, record: ServerModelRecord) -> None: ...

    @abstractmethod
    def get_promoted(
        self, ticker: str, forecast_type: str = "price"
    ) -> ServerModelRecord | None: ...

    @abstractmethod
    def promote(self, version_id: str) -> ServerModelRecord: ...

    @abstractmethod
    def reject(self, version_id: str, reason: str) -> None: ...

    @abstractmethod
    def rollback(self, ticker: str, forecast_type: str = "price") -> ServerModelRecord: ...

    @abstractmethod
    def list_artifacts(self, ticker: str) -> list[ServerModelRecord]: ...

    @abstractmethod
    def append_audit(self, event: str, details: dict[str, Any] | None = None) -> None: ...

    @abstractmethod
    def read_audit_log(self) -> list[dict[str, Any]]: ...

    @abstractmethod
    def enqueue_job(
        self, ticker: str, forecast_type: str = "price", payload: dict[str, Any] | None = None
    ) -> str: ...

    @abstractmethod
    def dequeue_job(self) -> dict[str, Any] | None: ...

    @abstractmethod
    def complete_job(self, job_id: str) -> None: ...

    @abstractmethod
    def fail_job(self, job_id: str, reason: str) -> None: ...


def _now() -> datetime:
    return datetime.now(UTC)


class InMemoryRegistry(ModelRegistry):
    """Thread-safe in-memory registry mirroring the Postgres semantics."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._artifacts: dict[str, ServerModelRecord] = {}
        self._rejection_reasons: dict[str, str] = {}
        self._promotions: dict[tuple[str, str], dict[str, str | None]] = {}
        self._audit_log: list[dict[str, Any]] = []
        self._jobs: list[dict[str, Any]] = []

    def insert_artifact(self, record: ServerModelRecord) -> None:
        with self._lock:
            version_id = record.key.version_id
            if version_id in self._artifacts:
                raise ModelRegistryError(f"Artifact version already exists: {version_id}")
            self._artifacts[version_id] = record.model_copy(deep=True)
            self._audit("artifact_inserted", {"version_id": version_id})

    def _get(self, version_id: str) -> ServerModelRecord:
        record = self._artifacts.get(version_id)
        if record is None:
            raise ModelRegistryError(f"Unknown artifact version: {version_id}")
        return record

    def get_promoted(self, ticker: str, forecast_type: str = "price") -> ServerModelRecord | None:
        with self._lock:
            pointer = self._promotions.get((ticker, forecast_type))
            if pointer is None or pointer["current"] is None:
                return None
            current_id = pointer["current"]
            assert current_id is not None
            return self._get(current_id).model_copy(deep=True)

    def promote(self, version_id: str) -> ServerModelRecord:
        with self._lock:
            record = self._get(version_id)
            if record.status == "rejected":
                raise ModelRegistryError(f"Artifact version is rejected: {version_id}")
            if record.status == "promoted":
                return record.model_copy(deep=True)
            pair = (record.key.ticker, record.key.forecast_type)
            pointer = self._promotions.get(pair, {"current": None, "previous": None})
            previous_id = pointer["current"]
            if previous_id is not None and previous_id in self._artifacts:
                previous = self._artifacts[previous_id]
                self._artifacts[previous_id] = previous.model_copy(update={"status": "candidate"})
            self._artifacts[version_id] = record.model_copy(update={"status": "promoted"})
            self._promotions[pair] = {"current": version_id, "previous": previous_id}
            self._audit(
                "artifact_promoted",
                {"version_id": version_id, "previous_version": previous_id},
            )
            return self._artifacts[version_id].model_copy(deep=True)

    def reject(self, version_id: str, reason: str) -> None:
        with self._lock:
            record = self._get(version_id)
            if record.status == "promoted":
                raise ModelRegistryError(
                    f"Promoted artifact cannot be rejected; roll back first: {version_id}"
                )
            self._artifacts[version_id] = record.model_copy(update={"status": "rejected"})
            self._rejection_reasons[version_id] = reason
            self._audit("artifact_rejected", {"version_id": version_id, "reason": reason})

    def rejection_reason(self, version_id: str) -> str | None:
        with self._lock:
            return self._rejection_reasons.get(version_id)

    def rollback(self, ticker: str, forecast_type: str = "price") -> ServerModelRecord:
        with self._lock:
            pointer = self._promotions.get((ticker, forecast_type))
            if pointer is None or pointer.get("previous") is None:
                raise ModelRegistryError("No previous artifact is available for rollback.")
            previous_id = pointer["previous"]
            current_id = pointer["current"]
            if current_id is not None and current_id in self._artifacts:
                current = self._artifacts[current_id]
                self._artifacts[current_id] = current.model_copy(update={"status": "candidate"})
            assert previous_id is not None
            previous = self._get(previous_id)
            self._artifacts[previous_id] = previous.model_copy(update={"status": "promoted"})
            self._promotions[(ticker, forecast_type)] = {
                "current": previous_id,
                "previous": None,
            }
            self._audit(
                "artifact_rollback",
                {"version_id": previous_id, "demoted_version": current_id},
            )
            return self._artifacts[previous_id].model_copy(deep=True)

    def list_artifacts(self, ticker: str) -> list[ServerModelRecord]:
        with self._lock:
            records = [
                record.model_copy(deep=True)
                for record in self._artifacts.values()
                if record.key.ticker == ticker
            ]
        return sorted(records, key=lambda record: record.key.trained_at, reverse=True)

    def append_audit(self, event: str, details: dict[str, Any] | None = None) -> None:
        with self._lock:
            self._audit(event, details)

    def read_audit_log(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._audit_log)

    def _audit(self, event: str, details: dict[str, Any] | None) -> None:
        self._audit_log.append(
            {"event": event, "details": details or {}, "created_at": _now().isoformat()}
        )

    def enqueue_job(
        self, ticker: str, forecast_type: str = "price", payload: dict[str, Any] | None = None
    ) -> str:
        with self._lock:
            job_id = uuid.uuid4().hex
            self._jobs.append(
                {
                    "id": job_id,
                    "ticker": ticker,
                    "forecast_type": forecast_type,
                    "payload": payload or {},
                    "status": "queued",
                    "attempts": 0,
                    "enqueued_at": _now().isoformat(),
                }
            )
            return job_id

    def dequeue_job(self) -> dict[str, Any] | None:
        """Claim the oldest queued job; the lock mirrors SKIP LOCKED semantics."""

        with self._lock:
            for job in self._jobs:
                if job["status"] == "queued":
                    job["status"] = "processing"
                    job["attempts"] += 1
                    return dict(job)
            return None

    def complete_job(self, job_id: str) -> None:
        pass

    def fail_job(self, job_id: str, reason: str) -> None:
        pass


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS server_artifacts (
    version_id TEXT PRIMARY KEY,
    ticker TEXT NOT NULL,
    forecast_type TEXT NOT NULL,
    profile TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    snapshot_id TEXT NOT NULL,
    trained_at TIMESTAMPTZ NOT NULL,
    record_json JSONB NOT NULL,
    status TEXT NOT NULL DEFAULT 'candidate',
    rejection_reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_server_artifacts_promoted
    ON server_artifacts (ticker, forecast_type) WHERE status = 'promoted';
CREATE TABLE IF NOT EXISTS server_promotions (
    ticker TEXT NOT NULL,
    forecast_type TEXT NOT NULL,
    current_version TEXT REFERENCES server_artifacts (version_id),
    previous_version TEXT REFERENCES server_artifacts (version_id),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (ticker, forecast_type)
);
CREATE TABLE IF NOT EXISTS training_jobs (
    id BIGSERIAL PRIMARY KEY,
    ticker TEXT NOT NULL,
    forecast_type TEXT NOT NULL DEFAULT 'price',
    payload JSONB,
    status TEXT NOT NULL DEFAULT 'queued',
    attempts INTEGER NOT NULL DEFAULT 0,
    enqueued_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS audit_log (
    id BIGSERIAL PRIMARY KEY,
    event TEXT NOT NULL,
    details JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


class PostgresRegistry(ModelRegistry):
    """psycopg 3 sync registry; rows are immutable and swaps happen in one statement."""

    def __init__(
        self,
        database_url: str | None = None,
        *,
        conn: Any | None = None,
    ) -> None:
        if conn is not None:
            self._conn = conn
        else:
            if not database_url:
                raise ModelRegistryError("registry_database_url is required.")
            import psycopg

            self._conn = psycopg.connect(database_url, autocommit=False)

    def init_schema(self) -> None:
        with self._conn.cursor() as cursor:
            cursor.execute(SCHEMA_SQL)
        self._conn.commit()

    def insert_artifact(self, record: ServerModelRecord) -> None:
        key = record.key
        with self._conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO server_artifacts (version_id, ticker, forecast_type, profile, "
                "schema_version, snapshot_id, trained_at, record_json, status) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    key.version_id,
                    key.ticker,
                    key.forecast_type,
                    key.profile,
                    key.schema_version,
                    key.snapshot_id,
                    key.trained_at,
                    json.dumps(record.model_dump(mode="json")),
                    record.status,
                ),
            )
            cursor.execute(
                "INSERT INTO audit_log (event, details) VALUES (%s, %s)",
                ("artifact_inserted", json.dumps({"version_id": key.version_id})),
            )
        self._conn.commit()

    def _record_from_row(self, row: Any) -> ServerModelRecord:
        record = ServerModelRecord.model_validate_json(row[0])
        return record.model_copy(update={"status": row[1]})

    def get_promoted(self, ticker: str, forecast_type: str = "price") -> ServerModelRecord | None:
        with self._conn.cursor() as cursor:
            cursor.execute(
                "SELECT a.record_json, a.status FROM server_promotions p "
                "JOIN server_artifacts a ON a.version_id = p.current_version "
                "WHERE p.ticker = %s AND p.forecast_type = %s AND a.status = 'promoted'",
                (ticker, forecast_type),
            )
            row = cursor.fetchone()
        return None if row is None else self._record_from_row(row)

    def promote(self, version_id: str) -> ServerModelRecord:
        with self._conn.cursor() as cursor:
            cursor.execute(
                "UPDATE server_artifacts AS incoming SET status = 'promoted' "
                "FROM (SELECT ticker, forecast_type FROM server_artifacts WHERE version_id = %s) t "
                "WHERE incoming.version_id = %s AND incoming.status != 'rejected' "
                "RETURNING incoming.record_json",
                (version_id, version_id),
            )
            row = cursor.fetchone()
            if row is None:
                self._conn.rollback()
                raise ModelRegistryError(f"Unknown or rejected artifact version: {version_id}")
            cursor.execute(
                "UPDATE server_artifacts SET status = 'candidate' "
                "WHERE version_id != %s AND status = 'promoted' "
                "AND (ticker, forecast_type) = (SELECT ticker, forecast_type FROM server_artifacts "
                "WHERE version_id = %s)",
                (version_id, version_id),
            )
            cursor.execute(
                "INSERT INTO server_promotions (ticker, forecast_type, current_version, "
                "previous_version, updated_at) "
                "SELECT ticker, forecast_type, %s, current_version, now() "
                "FROM server_artifacts WHERE version_id = %s "
                "ON CONFLICT (ticker, forecast_type) DO UPDATE SET "
                "previous_version = server_promotions.current_version, "
                "current_version = EXCLUDED.current_version, updated_at = now()",
                (version_id, version_id),
            )
            cursor.execute(
                "INSERT INTO audit_log (event, details) VALUES (%s, %s)",
                ("artifact_promoted", json.dumps({"version_id": version_id})),
            )
        self._conn.commit()
        record = ServerModelRecord.model_validate_json(row[0])
        return record.model_copy(update={"status": "promoted"})

    def reject(self, version_id: str, reason: str) -> None:
        with self._conn.cursor() as cursor:
            cursor.execute(
                "UPDATE server_artifacts SET status = 'rejected', rejection_reason = %s "
                "WHERE version_id = %s AND status != 'promoted'",
                (reason, version_id),
            )
            if cursor.rowcount == 0:
                self._conn.rollback()
                raise ModelRegistryError(f"Artifact is unknown or already promoted: {version_id}")
            cursor.execute(
                "INSERT INTO audit_log (event, details) VALUES (%s, %s)",
                (
                    "artifact_rejected",
                    json.dumps({"version_id": version_id, "reason": reason}),
                ),
            )
        self._conn.commit()

    def rollback(self, ticker: str, forecast_type: str = "price") -> ServerModelRecord:
        with self._conn.cursor() as cursor:
            cursor.execute(
                "UPDATE server_promotions SET "
                "current_version = previous_version, previous_version = NULL, updated_at = now() "
                "WHERE ticker = %s AND forecast_type = %s AND previous_version IS NOT NULL "
                "RETURNING current_version",
                (ticker, forecast_type),
            )
            row = cursor.fetchone()
            if row is None:
                self._conn.rollback()
                raise ModelRegistryError("No previous artifact is available for rollback.")
            version_id = row[0]
            cursor.execute(
                "UPDATE server_artifacts SET status = 'promoted' WHERE version_id = %s",
                (version_id,),
            )
            cursor.execute(
                "UPDATE server_artifacts SET status = 'candidate' "
                "WHERE ticker = %s AND forecast_type = %s AND version_id != %s "
                "AND status = 'promoted'",
                (ticker, forecast_type, version_id),
            )
            cursor.execute(
                "INSERT INTO audit_log (event, details) VALUES (%s, %s)",
                ("artifact_rollback", json.dumps({"version_id": version_id})),
            )
        self._conn.commit()
        promoted = self.get_promoted(ticker, forecast_type)
        if promoted is None:
            raise ModelRegistryError("Rollback left the promotion pointer in an invalid state.")
        return promoted

    def list_artifacts(self, ticker: str) -> list[ServerModelRecord]:
        with self._conn.cursor() as cursor:
            cursor.execute(
                "SELECT record_json, status FROM server_artifacts "
                "WHERE ticker = %s ORDER BY trained_at DESC",
                (ticker,),
            )
            rows = cursor.fetchall()
        return [self._record_from_row(row) for row in rows]

    def append_audit(self, event: str, details: dict[str, Any] | None = None) -> None:
        with self._conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO audit_log (event, details) VALUES (%s, %s)",
                (event, json.dumps(details or {})),
            )
        self._conn.commit()

    def read_audit_log(self) -> list[dict[str, Any]]:
        with self._conn.cursor() as cursor:
            cursor.execute("SELECT event, details, created_at FROM audit_log ORDER BY id")
            rows = cursor.fetchall()
        return [
            {"event": row[0], "details": row[1] or {}, "created_at": row[2].isoformat()}
            for row in rows
        ]

    def enqueue_job(
        self, ticker: str, forecast_type: str = "price", payload: dict[str, Any] | None = None
    ) -> str:
        with self._conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO training_jobs (ticker, forecast_type, payload) "
                "VALUES (%s, %s, %s) RETURNING id",
                (ticker, forecast_type, json.dumps(payload or {})),
            )
            row = cursor.fetchone()
        self._conn.commit()
        return str(row[0])

    def dequeue_job(self) -> dict[str, Any] | None:
        """Claim one queued job using FOR UPDATE SKIP LOCKED."""

        with self._conn.cursor() as cursor:
            cursor.execute(
                "SELECT id FROM training_jobs WHERE status = 'queued' "
                "ORDER BY id LIMIT 1 FOR UPDATE SKIP LOCKED"
            )
            row = cursor.fetchone()
            if row is None:
                self._conn.rollback()
                return None
            cursor.execute(
                "UPDATE training_jobs SET status = 'processing', attempts = attempts + 1 "
                "WHERE id = %s RETURNING id, ticker, forecast_type, payload, attempts",
                (row[0],),
            )
            claimed = cursor.fetchone()
        self._conn.commit()
        return {
            "id": str(claimed[0]),
            "ticker": claimed[1],
            "forecast_type": claimed[2],
            "payload": claimed[3] or {},
            "attempts": claimed[4],
        }

    def complete_job(self, job_id: str) -> None:
        with self._conn.cursor() as cursor:
            cursor.execute("UPDATE training_jobs SET status = 'completed' WHERE id = %s", (job_id,))
        self._conn.commit()

    def fail_job(self, job_id: str, reason: str) -> None:
        with self._conn.cursor() as cursor:
            # Maybe append reason to payload for debugging?
            cursor.execute(
                "UPDATE training_jobs SET status = 'failed', payload = payload || jsonb_build_object('error', %s::text) WHERE id = %s",
                (
                    reason,
                    job_id,
                ),
            )
        self._conn.commit()
