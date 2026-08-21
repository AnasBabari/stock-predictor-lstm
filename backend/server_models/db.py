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
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from server_models.contracts import ServerModelRecord

# A claimed job whose worker dies stays 'processing' until its lease expires;
# expiry returns it to 'queued' unless it has burned MAX_JOB_ATTEMPTS.
DEFAULT_LEASE_SECONDS = 1800
MAX_JOB_ATTEMPTS = 3


def _validate_record_payload(payload: Any) -> ServerModelRecord:
    """Parse a registry row payload whether psycopg returned the JSONB column as
    a Python object (dict) or a JSON string/bytes."""
    if isinstance(payload, (str, bytes, bytearray)):
        return ServerModelRecord.model_validate_json(payload)
    return ServerModelRecord.model_validate(payload)


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
    def prune_bundle_objects(
        self, before: datetime, delete_bundle: Callable[[str], None]
    ) -> list[str]: ...

    @abstractmethod
    def append_audit(self, event: str, details: dict[str, Any] | None = None) -> None: ...

    @abstractmethod
    def read_audit_log(self) -> list[dict[str, Any]]: ...

    @abstractmethod
    def enqueue_job(
        self, ticker: str, forecast_type: str = "price", payload: dict[str, Any] | None = None
    ) -> str: ...

    @abstractmethod
    def dequeue_job(self, lease_seconds: int = DEFAULT_LEASE_SECONDS) -> dict[str, Any] | None: ...

    @abstractmethod
    def complete_job(self, job_id: str) -> None: ...

    @abstractmethod
    def fail_job(self, job_id: str, reason: str) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


def _now() -> datetime:
    return datetime.now(UTC)


class InMemoryRegistry(ModelRegistry):
    """Thread-safe in-memory registry mirroring the Postgres semantics."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._artifacts: dict[str, ServerModelRecord] = {}
        self._rejection_reasons: dict[str, str] = {}
        self._promotions: dict[tuple[str, str], dict[str, str | None]] = {}
        self._created_at: dict[str, datetime] = {}
        self._pruned: set[str] = set()
        self._audit_log: list[dict[str, Any]] = []
        self._jobs: list[dict[str, Any]] = []
        self._clock: Callable[[], datetime] = _now

    def insert_artifact(self, record: ServerModelRecord) -> None:
        with self._lock:
            version_id = record.key.version_id
            if version_id in self._artifacts:
                raise ModelRegistryError(f"Artifact version already exists: {version_id}")
            self._artifacts[version_id] = record.model_copy(deep=True)
            self._created_at[version_id] = _now()
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
            if version_id in self._pruned:
                raise ModelRegistryError(f"Artifact bundle was pruned: {version_id}")
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

    def prune_bundle_objects(
        self, before: datetime, delete_bundle: Callable[[str], None]
    ) -> list[str]:
        """Delete expired non-pointer bundles while holding the promotion snapshot."""
        with self._lock:
            protected = {
                version_id
                for pointer in self._promotions.values()
                for version_id in (pointer.get("current"), pointer.get("previous"))
                if version_id is not None
            }
            candidates = sorted(
                version_id
                for version_id, created_at in self._created_at.items()
                if created_at < before
                and version_id not in protected
                and version_id not in self._pruned
                and self._artifacts[version_id].status != "promoted"
            )
            for version_id in candidates:
                delete_bundle(version_id)
                self._pruned.add(version_id)
            if candidates:
                self._audit(
                    "artifact_bundles_pruned",
                    {"version_ids": candidates, "count": len(candidates)},
                )
            return candidates

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
                    "leased_at": None,
                    "lease_expires_at": None,
                    "last_error": None,
                }
            )
            return job_id

    def dequeue_job(self, lease_seconds: int = DEFAULT_LEASE_SECONDS) -> dict[str, Any] | None:
        """Claim the oldest queued job; mirrors the Postgres lease semantics.

        Expired leases are reclaimed first (status back to 'queued', or
        'failed' once attempts exceed MAX_JOB_ATTEMPTS), then the oldest
        queued job is claimed under the lock (SKIP LOCKED analogue).
        """
        with self._lock:
            now = self._clock()
            for job in self._jobs:
                if (
                    job["status"] == "processing"
                    and job.get("lease_expires_at") is not None
                    and job["lease_expires_at"] < now
                ):
                    if job["attempts"] >= MAX_JOB_ATTEMPTS:
                        job["status"] = "failed"
                        job["last_error"] = "lease expired after max attempts"
                    else:
                        job["status"] = "queued"
                    job["leased_at"] = None
                    job["lease_expires_at"] = None
            for job in self._jobs:
                if job["status"] == "queued":
                    job["status"] = "processing"
                    job["attempts"] += 1
                    job["leased_at"] = now
                    job["lease_expires_at"] = now + timedelta(seconds=lease_seconds)
                    return dict(job)
            return None

    def complete_job(self, job_id: str) -> None:
        with self._lock:
            for job in self._jobs:
                if str(job["id"]) == str(job_id):
                    job["status"] = "completed"
                    job["leased_at"] = None
                    job["lease_expires_at"] = None

    def fail_job(self, job_id: str, reason: str) -> None:
        with self._lock:
            for job in self._jobs:
                if str(job["id"]) == str(job_id):
                    job["status"] = "failed"
                    job["last_error"] = reason
                    job["leased_at"] = None
                    job["lease_expires_at"] = None

    def close(self) -> None:
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
    bundle_pruned_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
ALTER TABLE server_artifacts
    ADD COLUMN IF NOT EXISTS bundle_pruned_at TIMESTAMPTZ;
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
    enqueued_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    leased_at TIMESTAMPTZ,
    lease_expires_at TIMESTAMPTZ,
    last_error TEXT
);
CREATE TABLE IF NOT EXISTS audit_log (
    id BIGSERIAL PRIMARY KEY,
    event TEXT NOT NULL,
    details JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

# Idempotent upgrades for registries created before job leasing existed.
SCHEMA_MIGRATIONS_SQL = """
ALTER TABLE training_jobs ADD COLUMN IF NOT EXISTS leased_at TIMESTAMPTZ;
ALTER TABLE training_jobs ADD COLUMN IF NOT EXISTS lease_expires_at TIMESTAMPTZ;
ALTER TABLE training_jobs ADD COLUMN IF NOT EXISTS last_error TEXT;
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
            cursor.execute(SCHEMA_MIGRATIONS_SQL)
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
        record = _validate_record_payload(row[0])
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
            # 1. Resolve the pointer identity without locking the artifact yet.
            #    Retention and promotion both lock pointer rows before artifact
            #    rows, preventing a deadlock or a prune-while-promoting race.
            cursor.execute(
                "SELECT ticker, forecast_type FROM server_artifacts WHERE version_id = %s",
                (version_id,),
            )
            row = cursor.fetchone()
            if row is None:
                self._conn.rollback()
                raise ModelRegistryError(f"Unknown artifact version: {version_id}")
            ticker, forecast_type = row

            # 2. Lock the promotion pointer row (created on first promotion) so two
            #    concurrent promote calls serialize instead of clobbering each other.
            cursor.execute(
                "INSERT INTO server_promotions (ticker, forecast_type) VALUES (%s, %s) "
                "ON CONFLICT (ticker, forecast_type) DO NOTHING",
                (ticker, forecast_type),
            )
            cursor.execute(
                "SELECT current_version FROM server_promotions "
                "WHERE ticker = %s AND forecast_type = %s FOR UPDATE",
                (ticker, forecast_type),
            )
            previous_version = cursor.fetchone()[0]

            # 3. Read and lock the artifact only after the pointer. A retention
            #    sweep that selected this row must finish first; a completed sweep
            #    leaves bundle_pruned_at set and promotion fails closed.
            cursor.execute(
                "SELECT record_json, status, bundle_pruned_at FROM server_artifacts "
                "WHERE version_id = %s FOR UPDATE",
                (version_id,),
            )
            record_json, status, bundle_pruned_at = cursor.fetchone()
            if status == "rejected":
                self._conn.rollback()
                raise ModelRegistryError(f"Artifact version is rejected: {version_id}")
            if bundle_pruned_at is not None:
                self._conn.rollback()
                raise ModelRegistryError(f"Artifact bundle was pruned: {version_id}")

            # Re-promoting the live champion is a no-op that preserves the pointer.
            if status == "promoted" and previous_version == version_id:
                self._conn.rollback()
                return _validate_record_payload(record_json)

            # 4. Demote any existing champion *before* promoting the incoming row so
            #    the unique partial index on (ticker, forecast_type) can never observe
            #    two 'promoted' rows at the same instant.
            cursor.execute(
                "UPDATE server_artifacts SET status = 'candidate' "
                "WHERE ticker = %s AND forecast_type = %s AND status = 'promoted' "
                "AND version_id != %s",
                (ticker, forecast_type, version_id),
            )
            cursor.execute(
                "UPDATE server_artifacts SET status = 'promoted' WHERE version_id = %s",
                (version_id,),
            )
            cursor.execute(
                "UPDATE server_promotions SET current_version = %s, previous_version = %s, "
                "updated_at = now() WHERE ticker = %s AND forecast_type = %s",
                (version_id, previous_version, ticker, forecast_type),
            )
            cursor.execute(
                "INSERT INTO audit_log (event, details) VALUES (%s, %s)",
                (
                    "artifact_promoted",
                    json.dumps(
                        {
                            "version_id": version_id,
                            "previous_version": previous_version,
                        }
                    ),
                ),
            )
        self._conn.commit()
        return _validate_record_payload(record_json).model_copy(update={"status": "promoted"})

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
        try:
            with self._conn.cursor() as cursor:
                # 1. Lock the promotion pointer row and read both versions so the
                #    whole rollback serializes against concurrent promote calls.
                cursor.execute(
                    "SELECT current_version, previous_version FROM server_promotions "
                    "WHERE ticker = %s AND forecast_type = %s FOR UPDATE",
                    (ticker, forecast_type),
                )
                row = cursor.fetchone()
                if row is None or row[1] is None:
                    self._conn.rollback()
                    raise ModelRegistryError("No previous artifact is available for rollback.")
                current_version, previous_version = row

                # 2. Demote the live champion *first*. The partial unique index on
                #    (ticker, forecast_type) allows exactly one 'promoted' row, so
                #    promoting the previous version before demoting the champion
                #    would violate it. Demote-then-promote never observes two
                #    promoted rows at the same instant.
                cursor.execute(
                    "UPDATE server_artifacts SET status = 'candidate' "
                    "WHERE version_id = %s AND status = 'promoted'",
                    (current_version,),
                )
                # 3. Promote the previous version, then repoint the pointer.
                cursor.execute(
                    "UPDATE server_artifacts SET status = 'promoted' WHERE version_id = %s",
                    (previous_version,),
                )
                cursor.execute(
                    "UPDATE server_promotions SET current_version = %s, previous_version = NULL, "
                    "updated_at = now() WHERE ticker = %s AND forecast_type = %s",
                    (previous_version, ticker, forecast_type),
                )
                cursor.execute(
                    "INSERT INTO audit_log (event, details) VALUES (%s, %s)",
                    (
                        "artifact_rollback",
                        json.dumps(
                            {
                                "version_id": previous_version,
                                "demoted_version": current_version,
                            }
                        ),
                    ),
                )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        promoted = self.get_promoted(ticker, forecast_type)
        if promoted is None:
            raise ModelRegistryError("Rollback left the promotion pointer in an invalid state.")
        return promoted

    def close(self) -> None:
        self._conn.close()

    def list_artifacts(self, ticker: str) -> list[ServerModelRecord]:
        with self._conn.cursor() as cursor:
            cursor.execute(
                "SELECT record_json, status FROM server_artifacts "
                "WHERE ticker = %s ORDER BY trained_at DESC",
                (ticker,),
            )
            rows = cursor.fetchall()
        return [self._record_from_row(row) for row in rows]

    def prune_bundle_objects(
        self, before: datetime, delete_bundle: Callable[[str], None]
    ) -> list[str]:
        """Prune expired blobs under the same pointer-first lock order as promotion."""
        try:
            with self._conn.cursor() as cursor:
                # Lock the complete pointer snapshot in a deterministic order.
                # Current and previous versions remain protected for serving and
                # rollback throughout the object-store deletes.
                cursor.execute(
                    "SELECT ticker, forecast_type FROM server_promotions "
                    "ORDER BY ticker, forecast_type FOR UPDATE"
                )
                cursor.execute(
                    "SELECT a.version_id FROM server_artifacts a "
                    "WHERE a.created_at < %s AND a.bundle_pruned_at IS NULL "
                    "AND a.status != 'promoted' "
                    "AND NOT EXISTS ("
                    "SELECT 1 FROM server_promotions p "
                    "WHERE a.version_id = p.current_version "
                    "OR a.version_id = p.previous_version"
                    ") ORDER BY a.created_at, a.version_id FOR UPDATE OF a",
                    (before,),
                )
                version_ids = [str(row[0]) for row in cursor.fetchall()]
                for version_id in version_ids:
                    delete_bundle(version_id)
                    cursor.execute(
                        "UPDATE server_artifacts SET bundle_pruned_at = now() "
                        "WHERE version_id = %s AND bundle_pruned_at IS NULL",
                        (version_id,),
                    )
                if version_ids:
                    cursor.execute(
                        "INSERT INTO audit_log (event, details) VALUES (%s, %s)",
                        (
                            "artifact_bundles_pruned",
                            json.dumps({"version_ids": version_ids, "count": len(version_ids)}),
                        ),
                    )
            self._conn.commit()
            return version_ids
        except Exception:
            self._conn.rollback()
            raise

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

    def dequeue_job(self, lease_seconds: int = DEFAULT_LEASE_SECONDS) -> dict[str, Any] | None:
        """Claim one queued job using FOR UPDATE SKIP LOCKED.

        Expired leases are reclaimed first: a worker that died mid-run must
        not strand its job in 'processing' forever. Reclaiming past
        MAX_JOB_ATTEMPTS fails the job instead of looping forever.
        """
        with self._conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE training_jobs SET
                    status = CASE WHEN attempts >= %s THEN 'failed' ELSE 'queued' END,
                    last_error = CASE WHEN attempts >= %s
                        THEN 'lease expired after max attempts' ELSE last_error END,
                    leased_at = NULL,
                    lease_expires_at = NULL
                WHERE status = 'processing' AND lease_expires_at < now()
                """,
                (MAX_JOB_ATTEMPTS, MAX_JOB_ATTEMPTS),
            )
            cursor.execute(
                "SELECT id FROM training_jobs WHERE status = 'queued' "
                "ORDER BY id LIMIT 1 FOR UPDATE SKIP LOCKED"
            )
            row = cursor.fetchone()
            if row is None:
                self._conn.rollback()
                return None
            cursor.execute(
                """
                UPDATE training_jobs SET
                    status = 'processing',
                    attempts = attempts + 1,
                    leased_at = now(),
                    lease_expires_at = now() + make_interval(secs => %s)
                WHERE id = %s
                RETURNING id, ticker, forecast_type, payload, attempts
                """,
                (lease_seconds, row[0]),
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
            cursor.execute(
                "UPDATE training_jobs SET status = 'completed', "
                "leased_at = NULL, lease_expires_at = NULL WHERE id = %s",
                (job_id,),
            )
        self._conn.commit()

    def fail_job(self, job_id: str, reason: str) -> None:
        with self._conn.cursor() as cursor:
            cursor.execute(
                "UPDATE training_jobs SET status = 'failed', "
                "leased_at = NULL, lease_expires_at = NULL, "
                "payload = payload || jsonb_build_object('error', %s::text), "
                "last_error = %s WHERE id = %s",
                (
                    reason,
                    reason,
                    job_id,
                ),
            )
        self._conn.commit()
