"""Persistent forecast ledger service with strict immutability, fingerprinting, and provenance tracking.

Records volatility forecasts and scores them against subsequently realized
market volatility once future trading sessions resolve.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sqlite3
import subprocess
import sys
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from services.volatility_contract import validate_volatility_horizon

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_DEFAULT_DB_PATH = _REPO_ROOT / "data" / "forecast_ledger.db"
_EPS = 1e-6

VALID_RECORD_SOURCES = frozenset({"live", "historical_replay"})
SUPPORTED_REPLAY_MODELS = ("rolling_mean", "har_rv", "ewma", "persistence", "garch_11")


class LedgerUnavailableError(RuntimeError):
    """Raised when the configured forecast-ledger store cannot be used."""


class LedgerConflictError(ValueError):
    """Raised when an immutable logical ledger key has different evidence."""


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class _LedgerConnection:
    """Small DB-API adapter allowing the ledger SQL to run on SQLite or psycopg.

    Existing ledger queries deliberately use SQLite's ``?`` placeholder.  The
    adapter translates only placeholders for PostgreSQL, keeping the two
    backends on one tested code path and avoiding string interpolation of data.
    """

    def __init__(self, raw: Any, backend: str) -> None:
        self.raw = raw
        self.backend = backend

    def _translate(self, query: str) -> str:
        return query.replace("?", "%s") if self.backend == "postgres" else query

    def execute(self, query: str, params: Iterable[Any] = ()) -> Any:
        return self.raw.execute(self._translate(query), tuple(params))

    def executemany(self, query: str, params: Iterable[Iterable[Any]]) -> Any:
        return self.raw.executemany(self._translate(query), params)

    def commit(self) -> None:
        self.raw.commit()

    def rollback(self) -> None:
        self.raw.rollback()

    def close(self) -> None:
        self.raw.close()

    def __enter__(self) -> _LedgerConnection:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        try:
            if exc_type is not None:
                self.rollback()
        finally:
            self.close()


_POSTGRES_REQUIRED_COLUMNS = frozenset(
    {
        "id",
        "forecast_date",
        "ticker",
        "horizon",
        "target_date",
        "model_name",
        "predicted_volatility",
        "recent_realized_volatility",
        "origin_price",
        "lower_scenario_price",
        "upper_scenario_price",
        "actual_realized_volatility",
        "forecast_error",
        "abs_error",
        "qlike_loss",
        "status",
        "record_source",
        "model_version",
        "feature_set_version",
        "code_commit",
        "data_as_of",
        "data_provider",
        "forecast_fingerprint",
        "created_at",
    }
)


def get_current_code_commit() -> str:
    """Resolve the current git commit SHA or explicit environment variable; fallback to 'dev-local'."""
    env_commit = os.environ.get("APP_COMMIT_SHA") or os.environ.get("GIT_COMMIT")
    if env_commit:
        return str(env_commit).strip()
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short=8", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    return "dev-local"


def compute_forecast_fingerprint(
    *,
    ticker: str,
    forecast_date: str,
    horizon: int,
    target_date: str,
    model_name: str,
    model_version: str,
    feature_set_version: str,
    code_commit: str,
    data_as_of: str,
    data_provider: str,
    record_source: str,
    predicted_volatility: float,
    recent_realized_volatility: float,
    origin_price: float,
    lower_scenario_price: float,
    upper_scenario_price: float,
) -> str:
    """Compute a canonical SHA-256 fingerprint over all immutable forecast and provenance fields."""
    payload = {
        "code_commit": str(code_commit).strip(),
        "data_as_of": str(data_as_of).strip(),
        "data_provider": str(data_provider).strip().lower(),
        "feature_set_version": str(feature_set_version).strip(),
        "forecast_date": str(forecast_date).strip(),
        "horizon": int(horizon),
        "lower_scenario_price": round(float(lower_scenario_price), 4),
        "model_name": str(model_name).strip(),
        "model_version": str(model_version).strip(),
        "origin_price": round(float(origin_price), 4),
        "predicted_volatility": round(float(predicted_volatility), 6),
        "recent_realized_volatility": round(float(recent_realized_volatility), 6),
        "record_source": str(record_source).strip(),
        "target_date": str(target_date).strip(),
        "ticker": str(ticker).strip().upper(),
        "upper_scenario_price": round(float(upper_scenario_price), 4),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class ForecastRecord:
    id: int | None
    forecast_date: str
    ticker: str
    horizon: int
    target_date: str
    model_name: str
    predicted_volatility: float
    recent_realized_volatility: float
    origin_price: float
    lower_scenario_price: float
    upper_scenario_price: float
    actual_realized_volatility: float | None
    forecast_error: float | None
    abs_error: float | None
    qlike_loss: float | None
    status: str  # 'pending' | 'scored'
    record_source: str  # 'live' | 'historical_replay'
    model_version: str
    feature_set_version: str
    code_commit: str
    data_as_of: str
    data_provider: str
    forecast_fingerprint: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ForecastLedger:
    """Immutable forecast ledger backed by SQLite or PostgreSQL.

    A caller that supplies ``db_path`` always gets SQLite.  The default
    application ledger selects PostgreSQL when ``DATABASE_URL`` (or its
    explicit ``FORECAST_LEDGER_DATABASE_URL`` alias) is configured.  This
    precedence is intentional: tests and offline replay can use an explicit
    SQLite path without inheriting a deployment database environment.
    """

    def __init__(
        self,
        db_path: Path | str | None = None,
        *,
        database_url: str | None = None,
        database_required: bool | None = None,
    ) -> None:
        self.db_path: Path | None = Path(db_path) if db_path is not None else None
        self.database_url = self._resolve_database_url(database_url)
        self.connection_timeout_seconds = self._resolve_connection_timeout()
        self.database_required = (
            bool(database_required)
            if database_required is not None
            else self._resolve_database_required()
        )
        if self.db_path is not None:
            self.backend = "sqlite"
        elif self.database_url:
            self.backend = "postgres"
        else:
            self.backend = "sqlite"
            if self.database_required:
                raise LedgerUnavailableError(
                    "A durable PostgreSQL forecast ledger is required but DATABASE_URL is not configured"
                )
            self.db_path = _DEFAULT_DB_PATH

        if self.backend == "sqlite":
            assert self.db_path is not None
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self._validate_database_url()
        try:
            self._init_db()
        except LedgerUnavailableError:
            raise
        except Exception as exc:
            raise LedgerUnavailableError("Forecast ledger schema is unavailable") from exc

    @staticmethod
    def _resolve_database_url(database_url: str | None) -> str | None:
        if database_url is not None:
            value = str(database_url).strip()
            return value or None
        # Read the settings object when available, then fall back to direct
        # environment lookup for migration scripts and lightweight imports.
        try:
            from config import settings

            configured = getattr(settings, "forecast_ledger_database_url", None)
            if configured:
                return str(configured).strip()
        except Exception:
            pass
        value = os.environ.get("DATABASE_URL") or os.environ.get("FORECAST_LEDGER_DATABASE_URL")
        return str(value).strip() if value and str(value).strip() else None

    @staticmethod
    def _resolve_database_required() -> bool:
        try:
            from config import settings

            if bool(getattr(settings, "forecast_ledger_database_required", False)):
                return True
        except Exception:
            pass
        return _truthy(os.environ.get("FORECAST_LEDGER_DATABASE_REQUIRED", "false"))

    @staticmethod
    def _resolve_connection_timeout() -> int:
        try:
            from config import settings

            value = int(getattr(settings, "forecast_ledger_connection_timeout_seconds", 10))
        except Exception:
            try:
                value = int(os.environ.get("FORECAST_LEDGER_CONNECTION_TIMEOUT_SECONDS", "10"))
            except ValueError:
                value = 10
        return max(1, min(value, 60))

    def _validate_database_url(self) -> None:
        if not self.database_url or not self.database_url.lower().startswith(
            ("postgresql://", "postgres://")
        ):
            raise LedgerUnavailableError(
                "Forecast ledger DATABASE_URL must use a PostgreSQL connection URL"
            )

    def _get_connection(self) -> _LedgerConnection:
        if self.backend == "sqlite":
            assert self.db_path is not None
            try:
                raw = sqlite3.connect(
                    str(self.db_path), timeout=float(self.connection_timeout_seconds)
                )
                raw.row_factory = sqlite3.Row
                return _LedgerConnection(raw, "sqlite")
            except Exception as exc:
                raise LedgerUnavailableError("SQLite forecast ledger is unavailable") from exc

        try:
            import psycopg
            from psycopg.rows import dict_row

            raw = psycopg.connect(
                self.database_url,
                connect_timeout=self.connection_timeout_seconds,
                row_factory=dict_row,
            )
            return _LedgerConnection(raw, "postgres")
        except Exception as exc:
            # Do not include the URL: it can contain a password and is never
            # safe to put in logs or an HTTP response.
            raise LedgerUnavailableError("PostgreSQL forecast ledger is unavailable") from exc

    def _init_db(self) -> None:
        if self.backend == "postgres":
            self._init_postgres_db()
            return

        with self._get_connection() as conn:
            table_info = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='forecast_ledger'"
            ).fetchone()

            if table_info is None:
                conn.execute(
                    """
                    CREATE TABLE forecast_ledger (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        forecast_date TEXT NOT NULL,
                        ticker TEXT NOT NULL,
                        horizon INTEGER NOT NULL,
                        target_date TEXT NOT NULL,
                        model_name TEXT NOT NULL,
                        predicted_volatility REAL NOT NULL,
                        recent_realized_volatility REAL NOT NULL,
                        origin_price REAL NOT NULL,
                        lower_scenario_price REAL NOT NULL,
                        upper_scenario_price REAL NOT NULL,
                        actual_realized_volatility REAL,
                        forecast_error REAL,
                        abs_error REAL,
                        qlike_loss REAL,
                        status TEXT NOT NULL,
                        record_source TEXT NOT NULL DEFAULT 'live',
                        model_version TEXT NOT NULL DEFAULT 'deployable_v5',
                        feature_set_version TEXT NOT NULL DEFAULT 'deployable_feature_columns_v5',
                        code_commit TEXT NOT NULL DEFAULT 'dev-local',
                        data_as_of TEXT NOT NULL DEFAULT '',
                        data_provider TEXT NOT NULL DEFAULT 'unknown',
                        forecast_fingerprint TEXT NOT NULL DEFAULT '',
                        created_at TEXT NOT NULL,
                        UNIQUE(forecast_date, ticker, horizon, model_name, record_source)
                    )
                    """
                )
            else:
                existing_sql = str(table_info["sql"])
                existing_cols = {
                    row["name"]
                    for row in conn.execute("PRAGMA table_info(forecast_ledger)").fetchall()
                }

                sql_compact = "".join(existing_sql.split())
                expected_unique = "UNIQUE(forecast_date,ticker,horizon,model_name,record_source)"
                needs_migration = (
                    "forecast_fingerprint" not in existing_cols
                    or "data_provider" not in existing_cols
                    or "record_source" not in existing_cols
                    or expected_unique not in sql_compact
                )

                if needs_migration:
                    logger.info(
                        "Migrating forecast_ledger table to Phase 5.2 schema with fingerprinting..."
                    )
                    conn.execute(
                        """
                        CREATE TABLE forecast_ledger_migrating (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            forecast_date TEXT NOT NULL,
                            ticker TEXT NOT NULL,
                            horizon INTEGER NOT NULL,
                            target_date TEXT NOT NULL,
                            model_name TEXT NOT NULL,
                            predicted_volatility REAL NOT NULL,
                            recent_realized_volatility REAL NOT NULL,
                            origin_price REAL NOT NULL,
                            lower_scenario_price REAL NOT NULL,
                            upper_scenario_price REAL NOT NULL,
                            actual_realized_volatility REAL,
                            forecast_error REAL,
                            abs_error REAL,
                            qlike_loss REAL,
                            status TEXT NOT NULL,
                            record_source TEXT NOT NULL DEFAULT 'live',
                            model_version TEXT NOT NULL DEFAULT 'deployable_v5',
                            feature_set_version TEXT NOT NULL DEFAULT 'deployable_feature_columns_v5',
                            code_commit TEXT NOT NULL DEFAULT 'dev-local',
                            data_as_of TEXT NOT NULL DEFAULT '',
                            data_provider TEXT NOT NULL DEFAULT 'unknown',
                            forecast_fingerprint TEXT NOT NULL DEFAULT '',
                            created_at TEXT NOT NULL,
                            UNIQUE(forecast_date, ticker, horizon, model_name, record_source)
                        )
                        """
                    )
                    old_rows = conn.execute("SELECT * FROM forecast_ledger").fetchall()
                    for r in old_rows:
                        r_keys = r.keys()
                        f_date = str(r["forecast_date"])
                        ticker = str(r["ticker"]).upper()
                        horizon = int(r["horizon"])
                        target_date = str(r["target_date"])
                        m_name = str(r["model_name"])
                        pred_vol = float(r["predicted_volatility"])
                        recent_rv = float(r["recent_realized_volatility"])
                        origin_p = float(r["origin_price"])
                        lower_p = float(r["lower_scenario_price"])
                        upper_p = float(r["upper_scenario_price"])
                        act_rv = (
                            float(r["actual_realized_volatility"])
                            if r["actual_realized_volatility"] is not None
                            else None
                        )
                        f_err = (
                            float(r["forecast_error"]) if r["forecast_error"] is not None else None
                        )
                        abs_err = float(r["abs_error"]) if r["abs_error"] is not None else None
                        qlike = float(r["qlike_loss"]) if r["qlike_loss"] is not None else None
                        status = str(r["status"])
                        rec_src = str(r["record_source"]) if "record_source" in r_keys else "live"
                        m_ver = (
                            str(r["model_version"])
                            if "model_version" in r_keys
                            else "deployable_v5"
                        )
                        f_ver = (
                            str(r["feature_set_version"])
                            if "feature_set_version" in r_keys
                            else "deployable_feature_columns_v5"
                        )
                        c_commit = str(r["code_commit"]) if "code_commit" in r_keys else "dev-local"
                        d_as_of = str(r["data_as_of"]) if "data_as_of" in r_keys else f_date
                        d_provider = (
                            str(r["data_provider"]) if "data_provider" in r_keys else "unknown"
                        )
                        created_at = str(r["created_at"])

                        fp = compute_forecast_fingerprint(
                            ticker=ticker,
                            forecast_date=f_date,
                            horizon=horizon,
                            target_date=target_date,
                            model_name=m_name,
                            model_version=m_ver,
                            feature_set_version=f_ver,
                            code_commit=c_commit,
                            data_as_of=d_as_of,
                            data_provider=d_provider,
                            record_source=rec_src,
                            predicted_volatility=pred_vol,
                            recent_realized_volatility=recent_rv,
                            origin_price=origin_p,
                            lower_scenario_price=lower_p,
                            upper_scenario_price=upper_p,
                        )

                        conn.execute(
                            """
                            INSERT INTO forecast_ledger_migrating (
                                id, forecast_date, ticker, horizon, target_date, model_name,
                                predicted_volatility, recent_realized_volatility, origin_price,
                                lower_scenario_price, upper_scenario_price, actual_realized_volatility,
                                forecast_error, abs_error, qlike_loss, status,
                                record_source, model_version, feature_set_version, code_commit,
                                data_as_of, data_provider, forecast_fingerprint, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                r["id"],
                                f_date,
                                ticker,
                                horizon,
                                target_date,
                                m_name,
                                pred_vol,
                                recent_rv,
                                origin_p,
                                lower_p,
                                upper_p,
                                act_rv,
                                f_err,
                                abs_err,
                                qlike,
                                status,
                                rec_src,
                                m_ver,
                                f_ver,
                                c_commit,
                                d_as_of,
                                d_provider,
                                fp,
                                created_at,
                            ),
                        )

                    conn.execute("DROP TABLE forecast_ledger")
                    conn.execute("ALTER TABLE forecast_ledger_migrating RENAME TO forecast_ledger")

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ledger_ticker_source ON forecast_ledger(ticker, horizon, record_source)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ledger_status_source ON forecast_ledger(status, record_source)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ledger_fingerprint ON forecast_ledger(forecast_fingerprint)"
            )
            conn.commit()

    def _init_postgres_db(self) -> None:
        """Create or validate the durable PostgreSQL ledger schema.

        Existing incompatible schemas fail closed.  The SQLite migration
        command is deliberately explicit so a deployment cannot accidentally
        drop or reinterpret production history during application startup.
        """
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS forecast_ledger (
                        id BIGSERIAL PRIMARY KEY,
                        forecast_date TEXT NOT NULL,
                        ticker TEXT NOT NULL,
                        horizon INTEGER NOT NULL CHECK (horizon > 0),
                        target_date TEXT NOT NULL,
                        model_name TEXT NOT NULL,
                        predicted_volatility DOUBLE PRECISION NOT NULL,
                        recent_realized_volatility DOUBLE PRECISION NOT NULL,
                        origin_price DOUBLE PRECISION NOT NULL,
                        lower_scenario_price DOUBLE PRECISION NOT NULL,
                        upper_scenario_price DOUBLE PRECISION NOT NULL,
                        actual_realized_volatility DOUBLE PRECISION,
                        forecast_error DOUBLE PRECISION,
                        abs_error DOUBLE PRECISION,
                        qlike_loss DOUBLE PRECISION,
                        status TEXT NOT NULL CHECK (status IN ('pending', 'scored')),
                        record_source TEXT NOT NULL DEFAULT 'live'
                            CHECK (record_source IN ('live', 'historical_replay')),
                        model_version TEXT NOT NULL DEFAULT 'deployable_v5',
                        feature_set_version TEXT NOT NULL DEFAULT 'deployable_feature_columns_v5',
                        code_commit TEXT NOT NULL DEFAULT 'dev-local',
                        data_as_of TEXT NOT NULL DEFAULT '',
                        data_provider TEXT NOT NULL DEFAULT 'unknown',
                        forecast_fingerprint TEXT NOT NULL DEFAULT '',
                        created_at TEXT NOT NULL,
                        UNIQUE (forecast_date, ticker, horizon, model_name, record_source)
                    )
                    """
                )
                columns = {
                    str(row["column_name"])
                    for row in conn.execute(
                        """
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_schema = current_schema() AND table_name = 'forecast_ledger'
                        """
                    ).fetchall()
                }
                missing = _POSTGRES_REQUIRED_COLUMNS - columns
                if missing:
                    raise LedgerUnavailableError(
                        "PostgreSQL forecast ledger schema is incompatible; "
                        f"missing columns: {', '.join(sorted(missing))}"
                    )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_ledger_ticker_source "
                    "ON forecast_ledger(ticker, horizon, record_source)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_ledger_status_source "
                    "ON forecast_ledger(status, record_source)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_ledger_fingerprint "
                    "ON forecast_ledger(forecast_fingerprint)"
                )
                conn.commit()
        except LedgerUnavailableError:
            raise
        except Exception as exc:
            raise LedgerUnavailableError(
                "PostgreSQL forecast ledger schema is unavailable"
            ) from exc

    @property
    def storage_kind(self) -> str:
        """Return the selected storage backend without exposing credentials."""
        return "postgresql" if self.backend == "postgres" else "sqlite"

    @property
    def durable(self) -> bool:
        """Whether records survive replacement of an application instance."""
        return self.backend == "postgres"

    def check_connection(self) -> dict[str, Any]:
        """Run a bounded, read-only connectivity check for readiness probes."""
        try:
            with self._get_connection() as conn:
                conn.execute("SELECT 1").fetchone()
            return {
                "status": "available",
                "backend": self.storage_kind,
                "durable": self.durable,
                "required": self.database_required,
            }
        except LedgerUnavailableError:
            return {
                "status": "unavailable",
                "backend": self.storage_kind,
                "durable": self.durable,
                "required": self.database_required,
            }
        except Exception:
            return {
                "status": "unavailable",
                "backend": self.storage_kind,
                "durable": self.durable,
                "required": self.database_required,
            }

    def record_forecast(
        self,
        *,
        forecast_date: str,
        ticker: str,
        horizon: int,
        target_date: str,
        model_name: str,
        predicted_volatility: float,
        recent_realized_volatility: float,
        origin_price: float,
        lower_scenario_price: float,
        upper_scenario_price: float,
        record_source: str = "live",
        model_version: str = "deployable_v5",
        feature_set_version: str = "deployable_feature_columns_v5",
        code_commit: str | None = None,
        data_as_of: str = "",
        data_provider: str = "unknown",
    ) -> ForecastRecord:
        """Insert a forecast entry into the ledger with strict immutability and fingerprint verification.

        - If no record exists: inserts a new pending forecast record with a deterministic SHA-256 fingerprint.
        - If an identical forecast record exists (matching fingerprint): returns existing record idempotently.
        - If a conflicting forecast record exists (differing prediction or provenance): raises ValueError.
        - If an existing record has already settled (status='scored'): strictly forbids mutation.
        """
        source_str = str(record_source).strip()
        if source_str not in VALID_RECORD_SOURCES:
            raise ValueError(
                f"record_source must be one of {sorted(VALID_RECORD_SOURCES)}, got '{record_source}'"
            )

        now_iso = datetime.now(UTC).isoformat()
        f_date_str = str(forecast_date).strip()
        ticker_str = str(ticker).strip().upper()
        # Validate before coercion so a direct caller cannot smuggle a
        # fractional value such as 5.5 into the live ledger as horizon 5.
        h_int = validate_volatility_horizon(horizon) if source_str == "live" else int(horizon)
        t_date_str = str(target_date).strip()
        m_name_str = str(model_name).strip()
        m_version_str = str(model_version).strip()
        f_version_str = str(feature_set_version).strip()
        commit_str = str(code_commit).strip() if code_commit else get_current_code_commit()
        as_of_str = str(data_as_of).strip() or f_date_str
        provider_str = str(data_provider).strip().lower() or "unknown"

        pred_vol_f = float(predicted_volatility)
        recent_rv_f = float(recent_realized_volatility)
        origin_p_f = float(origin_price)
        lower_p_f = float(lower_scenario_price)
        upper_p_f = float(upper_scenario_price)

        calculated_fp = compute_forecast_fingerprint(
            ticker=ticker_str,
            forecast_date=f_date_str,
            horizon=h_int,
            target_date=t_date_str,
            model_name=m_name_str,
            model_version=m_version_str,
            feature_set_version=f_version_str,
            code_commit=commit_str,
            data_as_of=as_of_str,
            data_provider=provider_str,
            record_source=source_str,
            predicted_volatility=pred_vol_f,
            recent_realized_volatility=recent_rv_f,
            origin_price=origin_p_f,
            lower_scenario_price=lower_p_f,
            upper_scenario_price=upper_p_f,
        )

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM forecast_ledger
                WHERE forecast_date=? AND ticker=? AND horizon=? AND model_name=? AND record_source=?
                """,
                (f_date_str, ticker_str, h_int, m_name_str, source_str),
            )
            existing = cursor.fetchone()

            if existing is not None:
                existing_keys = set(existing.keys())
                existing_fp = (
                    existing["forecast_fingerprint"]
                    if "forecast_fingerprint" in existing_keys and existing["forecast_fingerprint"]
                    else compute_forecast_fingerprint(
                        ticker=existing["ticker"],
                        forecast_date=existing["forecast_date"],
                        horizon=existing["horizon"],
                        target_date=existing["target_date"],
                        model_name=existing["model_name"],
                        model_version=existing["model_version"],
                        feature_set_version=existing["feature_set_version"],
                        code_commit=existing["code_commit"],
                        data_as_of=existing["data_as_of"],
                        data_provider=(
                            existing["data_provider"]
                            if "data_provider" in existing_keys
                            else "unknown"
                        ),
                        record_source=existing["record_source"],
                        predicted_volatility=existing["predicted_volatility"],
                        recent_realized_volatility=existing["recent_realized_volatility"],
                        origin_price=existing["origin_price"],
                        lower_scenario_price=existing["lower_scenario_price"],
                        upper_scenario_price=existing["upper_scenario_price"],
                    )
                )

                if existing["status"] == "scored":
                    if existing_fp != calculated_fp:
                        raise LedgerConflictError(
                            f"Cannot modify or overwrite a settled/scored forecast for {ticker_str} "
                            f"on {f_date_str} (horizon={h_int}d, model={m_name_str}, source={source_str}). "
                            f"Settled fingerprint={existing_fp[:12]}..., incoming fingerprint={calculated_fp[:12]}... "
                            f"Settled forecast records are strictly immutable."
                        )
                    return self._row_to_record(existing)

                if existing_fp != calculated_fp:
                    raise LedgerConflictError(
                        f"Conflicting forecast fingerprint for {ticker_str} on {f_date_str} "
                        f"(horizon={h_int}d, model={m_name_str}, source={source_str}). "
                        f"Existing fingerprint={existing_fp[:12]}..., incoming fingerprint={calculated_fp[:12]}... "
                        f"Forecast records are immutable."
                    )
                return self._row_to_record(existing)

            conn.execute(
                """
                INSERT INTO forecast_ledger (
                    forecast_date, ticker, horizon, target_date, model_name,
                    predicted_volatility, recent_realized_volatility, origin_price,
                    lower_scenario_price, upper_scenario_price, status,
                    record_source, model_version, feature_set_version,
                    code_commit, data_as_of, data_provider, forecast_fingerprint, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (forecast_date, ticker, horizon, model_name, record_source) DO NOTHING
                """,
                (
                    f_date_str,
                    ticker_str,
                    h_int,
                    t_date_str,
                    m_name_str,
                    pred_vol_f,
                    recent_rv_f,
                    origin_p_f,
                    lower_p_f,
                    upper_p_f,
                    source_str,
                    m_version_str,
                    f_version_str,
                    commit_str,
                    as_of_str,
                    provider_str,
                    calculated_fp,
                    now_iso,
                ),
            )
            conn.commit()

            cursor = conn.execute(
                """
                SELECT * FROM forecast_ledger
                WHERE forecast_date=? AND ticker=? AND horizon=? AND model_name=? AND record_source=?
                """,
                (f_date_str, ticker_str, h_int, m_name_str, source_str),
            )
            row = cursor.fetchone()
            if row is None:
                raise LedgerUnavailableError("Forecast ledger insert was not durable")
            stored_fp = row["forecast_fingerprint"]
            if stored_fp != calculated_fp:
                raise LedgerConflictError(
                    f"Concurrent forecast conflict for {ticker_str} on {f_date_str} "
                    f"(horizon={h_int}d, model={m_name_str}, source={source_str})"
                )
            return self._row_to_record(row)

    def score_pending_forecasts(
        self,
        ticker: str,
        ohlcv_df: pd.DataFrame,
        *,
        record_source: str | None = None,
        annualization: float = 252.0,
    ) -> int:
        """Score pending forecasts where future realized market returns are now available."""
        if ohlcv_df.empty or "Close" not in ohlcv_df.columns:
            return 0

        df = ohlcv_df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            if "Date" in df.columns:
                df.index = pd.to_datetime(df["Date"])
            else:
                return 0
        df = df.sort_index()
        close_series = df["Close"].astype(float)
        date_str_list = df.index.strftime("%Y-%m-%d").to_list()

        scored_count = 0
        with self._get_connection() as conn:
            query = "SELECT * FROM forecast_ledger WHERE ticker=? AND status='pending'"
            params: list[Any] = [str(ticker).upper()]
            if record_source is not None:
                query += " AND record_source=?"
                params.append(str(record_source))
            if self.backend == "postgres":
                # Prevent two settlement workers from scoring the same row;
                # SQLite serializes the write transaction for local runs.
                query += " FOR UPDATE"

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

            for row in rows:
                f_date = row["forecast_date"]
                horizon = row["horizon"]
                pred_vol = row["predicted_volatility"]

                if f_date not in date_str_list:
                    continue
                origin_idx = date_str_list.index(f_date)

                if origin_idx + horizon >= len(close_series):
                    continue

                price_slice = close_series.iloc[origin_idx : origin_idx + horizon + 1].to_numpy()
                log_returns = np.log(price_slice[1:] / price_slice[:-1])
                squared_sum = float(np.sum(log_returns**2))
                actual_rv = math.sqrt(max(squared_sum * annualization / horizon, _EPS))

                error = float(pred_vol - actual_rv)
                abs_error = abs(error)

                pred_var = max(pred_vol**2, _EPS)
                actual_var = max(actual_rv**2, _EPS)
                ratio = actual_var / pred_var
                qlike = float(max(ratio - math.log(max(ratio, _EPS)) - 1.0, 0.0))

                actual_target_date = date_str_list[origin_idx + horizon]

                update_cursor = conn.execute(
                    """
                    UPDATE forecast_ledger
                    SET target_date=?,
                        actual_realized_volatility=?,
                        forecast_error=?,
                        abs_error=?,
                        qlike_loss=?,
                        status='scored'
                    WHERE id=?
                      AND status='pending'
                    """,
                    (actual_target_date, actual_rv, error, abs_error, qlike, row["id"]),
                )
                if update_cursor.rowcount == 1:
                    scored_count += 1

            conn.commit()

        return scored_count

    def get_ledger_entries(
        self,
        *,
        ticker: str | None = None,
        horizon: int | None = None,
        status: str | None = None,
        record_source: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Retrieve recent forecast ledger entries."""
        query = "SELECT * FROM forecast_ledger WHERE 1=1"
        params: list[Any] = []

        if ticker:
            query += " AND ticker=?"
            params.append(str(ticker).upper())
        if horizon:
            query += " AND horizon=?"
            params.append(int(horizon))
        if status:
            query += " AND status=?"
            params.append(str(status))
        if record_source:
            query += " AND record_source=?"
            params.append(str(record_source))

        query += " ORDER BY forecast_date DESC, id DESC LIMIT ?"
        params.append(int(limit))

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [self._row_to_record(row).to_dict() for row in cursor.fetchall()]

    def get_track_record_metrics(
        self,
        *,
        ticker: str | None = None,
        horizon: int | None = None,
        record_source: str = "live",
    ) -> dict[str, Any]:
        """Compute aggregate empirical track record accuracy metrics for a specific record source."""
        query = "SELECT * FROM forecast_ledger WHERE status='scored' AND record_source=?"
        params: list[Any] = [str(record_source)]

        if ticker:
            query += " AND ticker=?"
            params.append(str(ticker).upper())
        if horizon:
            query += " AND horizon=?"
            params.append(int(horizon))

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        if not rows:
            return {
                "record_source": record_source,
                "total_forecasts": 0,
                "scored_forecasts": 0,
                "mean_mae": None,
                "mean_rmse": None,
                "mean_qlike": None,
                "median_qlike": None,
                "direction_accuracy_pct": None,
            }

        abs_errors = [row["abs_error"] for row in rows if row["abs_error"] is not None]
        sq_errors = [
            row["forecast_error"] ** 2 for row in rows if row["forecast_error"] is not None
        ]
        qlikes = [row["qlike_loss"] for row in rows if row["qlike_loss"] is not None]

        direction_hits = 0
        total_direction = 0
        for r in rows:
            if r["actual_realized_volatility"] is not None:
                pred_delta = r["predicted_volatility"] - r["recent_realized_volatility"]
                actual_delta = r["actual_realized_volatility"] - r["recent_realized_volatility"]
                if (pred_delta >= 0 and actual_delta >= 0) or (pred_delta < 0 and actual_delta < 0):
                    direction_hits += 1
                total_direction += 1

        return {
            "record_source": record_source,
            "total_forecasts": len(rows),
            "scored_forecasts": len(rows),
            "mean_mae": float(np.mean(abs_errors)) if abs_errors else None,
            "mean_rmse": float(np.sqrt(np.mean(sq_errors))) if sq_errors else None,
            "mean_qlike": float(np.mean(qlikes)) if qlikes else None,
            "median_qlike": float(np.median(qlikes)) if qlikes else None,
            "direction_accuracy_pct": (
                float(direction_hits / total_direction * 100.0) if total_direction > 0 else None
            ),
        }

    def export_records(self, *, record_source: str | None = None) -> list[ForecastRecord]:
        """Return a deterministic snapshot ordered by database id.

        This is used by the explicit SQLite→PostgreSQL migration and export
        tools.  It never merges live and historical-replay tracks unless the
        caller intentionally omits ``record_source``.
        """
        query = "SELECT * FROM forecast_ledger"
        params: list[Any] = []
        if record_source is not None:
            if str(record_source) not in VALID_RECORD_SOURCES:
                raise ValueError(f"Unsupported record_source: {record_source}")
            query += " WHERE record_source=?"
            params.append(str(record_source))
        query += " ORDER BY id ASC"
        with self._get_connection() as conn:
            return [self._row_to_record(row) for row in conn.execute(query, params).fetchall()]

    def import_records(self, records: Iterable[ForecastRecord]) -> int:
        """Idempotently import records while preserving immutable evidence.

        The method is intentionally separate from ``record_forecast``: a
        migration must preserve historical status, settlement values, ids, and
        creation timestamps exactly rather than recomputing a new forecast.
        A logical-key collision is accepted only when every immutable field
        matches; otherwise the whole transaction fails closed.
        """
        columns = (
            "id, forecast_date, ticker, horizon, target_date, model_name, "
            "predicted_volatility, recent_realized_volatility, origin_price, "
            "lower_scenario_price, upper_scenario_price, actual_realized_volatility, "
            "forecast_error, abs_error, qlike_loss, status, record_source, model_version, "
            "feature_set_version, code_commit, data_as_of, data_provider, "
            "forecast_fingerprint, created_at"
        )
        records_list = list(records)
        if not records_list:
            return 0

        inserted = 0
        with self._get_connection() as conn:
            for record in records_list:
                if record.record_source not in VALID_RECORD_SOURCES:
                    raise ValueError(f"Unsupported record_source: {record.record_source}")
                values = (
                    record.id,
                    record.forecast_date,
                    record.ticker,
                    int(record.horizon),
                    record.target_date,
                    record.model_name,
                    record.predicted_volatility,
                    record.recent_realized_volatility,
                    record.origin_price,
                    record.lower_scenario_price,
                    record.upper_scenario_price,
                    record.actual_realized_volatility,
                    record.forecast_error,
                    record.abs_error,
                    record.qlike_loss,
                    record.status,
                    record.record_source,
                    record.model_version,
                    record.feature_set_version,
                    record.code_commit,
                    record.data_as_of,
                    record.data_provider,
                    record.forecast_fingerprint,
                    record.created_at,
                )
                insert_cursor = conn.execute(
                    f"INSERT INTO forecast_ledger ({columns}) VALUES "
                    f"({', '.join('?' for _ in values)}) "
                    "ON CONFLICT (forecast_date, ticker, horizon, model_name, record_source) "
                    "DO NOTHING",
                    values,
                )
                inserted_now = insert_cursor.rowcount == 1
                if inserted_now:
                    inserted += 1
                existing = conn.execute(
                    """
                    SELECT * FROM forecast_ledger
                    WHERE forecast_date=? AND ticker=? AND horizon=? AND model_name=? AND record_source=?
                    """,
                    (
                        record.forecast_date,
                        record.ticker,
                        int(record.horizon),
                        record.model_name,
                        record.record_source,
                    ),
                ).fetchone()
                if existing is None:
                    raise LedgerUnavailableError("Imported forecast record could not be verified")

                existing_record = self._row_to_record(existing)
                for field in (
                    "forecast_date",
                    "ticker",
                    "horizon",
                    "target_date",
                    "model_name",
                    "predicted_volatility",
                    "recent_realized_volatility",
                    "origin_price",
                    "lower_scenario_price",
                    "upper_scenario_price",
                    "actual_realized_volatility",
                    "forecast_error",
                    "abs_error",
                    "qlike_loss",
                    "status",
                    "record_source",
                    "model_version",
                    "feature_set_version",
                    "code_commit",
                    "data_as_of",
                    "data_provider",
                    "forecast_fingerprint",
                    "created_at",
                ):
                    if getattr(existing_record, field) != getattr(record, field):
                        raise LedgerConflictError(
                            "Imported forecast conflicts with an existing immutable logical key"
                        )
            if self.backend == "postgres":
                conn.execute(
                    """
                    SELECT setval(
                        pg_get_serial_sequence('forecast_ledger', 'id'),
                        COALESCE((SELECT MAX(id) FROM forecast_ledger), 1),
                        EXISTS (SELECT 1 FROM forecast_ledger)
                    )
                    """
                ).fetchone()
            conn.commit()
        return inserted

    def generate_historical_replay_ledger(
        self,
        ticker: str,
        ohlcv_df: pd.DataFrame,
        *,
        horizon: int = 5,
        model_name: str = "rolling_mean",
        lookback_sessions: int = 40,
    ) -> int:
        """Generate historical replay forecast entries from past data using the named model.

        Tagged explicitly with record_source='historical_replay' to ensure historical
        replays are never mixed with genuine live-emitted forecasts.
        Raises ValueError if an unsupported model_name is requested.
        """
        if model_name not in SUPPORTED_REPLAY_MODELS:
            raise ValueError(
                f"Unsupported replay model '{model_name}'. Must be one of {SUPPORTED_REPLAY_MODELS}"
            )

        if len(ohlcv_df) < 120:
            return 0

        df = ohlcv_df.sort_index()
        close = df["Close"].to_numpy(dtype=float)
        dates = df.index.strftime("%Y-%m-%d").to_list()
        n = len(df)

        seeded = 0
        commit_id = get_current_code_commit()
        start_idx = max(65, n - lookback_sessions - horizon)

        for i in range(start_idx, n - horizon):
            f_date = dates[i]
            t_date = dates[i + horizon]
            p_origin = float(close[i])

            ret_22 = np.log(close[i - 21 : i + 1] / close[i - 22 : i])
            recent_rv = float(np.std(ret_22, ddof=1) * math.sqrt(252))

            if model_name == "rolling_mean":
                ret_60 = np.log(close[i - 59 : i + 1] / close[i - 60 : i])
                pred_vol = float(np.std(ret_60, ddof=1) * math.sqrt(252))
            elif model_name == "har_rv":
                ret_1 = abs(float(np.log(close[i] / close[i - 1]))) * math.sqrt(252)
                ret_5 = float(
                    np.std(np.log(close[i - 4 : i + 1] / close[i - 5 : i]), ddof=1) * math.sqrt(252)
                )
                pred_vol = float(max(0.38 * ret_1 + 0.35 * ret_5 + 0.27 * recent_rv, _EPS))
            elif model_name == "ewma":
                lam = 0.94
                weights = (1 - lam) * (lam ** np.arange(len(ret_22) - 1, -1, -1))
                weights /= np.sum(weights)
                var_ewma = float(np.sum(weights * (ret_22**2)))
                pred_vol = float(math.sqrt(max(var_ewma * 252.0, _EPS)))
            elif model_name == "persistence":
                pred_vol = recent_rv
            elif model_name == "garch_11":
                from research.volatility_forecasting.simple_pipeline import (
                    fit_garch11_mle_from_returns,
                )

                pred_vol = fit_garch11_mle_from_returns(ret_22, horizon=horizon)
            else:
                raise ValueError(f"Unhandled replay model: {model_name}")

            sigma_h = pred_vol * math.sqrt(horizon / 252.0)
            p_lower = float(p_origin * math.exp(-1.64485 * sigma_h))
            p_upper = float(p_origin * math.exp(1.64485 * sigma_h))

            self.record_forecast(
                forecast_date=f_date,
                ticker=ticker,
                horizon=horizon,
                target_date=t_date,
                model_name=model_name,
                predicted_volatility=pred_vol,
                recent_realized_volatility=recent_rv,
                origin_price=p_origin,
                lower_scenario_price=p_lower,
                upper_scenario_price=p_upper,
                record_source="historical_replay",
                model_version="deployable_v5",
                feature_set_version="deployable_feature_columns_v5",
                code_commit=commit_id,
                data_as_of=f_date,
                data_provider="historical_replay",
            )
            seeded += 1

        self.score_pending_forecasts(ticker, ohlcv_df, record_source="historical_replay")
        return seeded

    @staticmethod
    def _row_to_record(row: Mapping[str, Any]) -> ForecastRecord:
        keys = row.keys()
        return ForecastRecord(
            id=row["id"],
            forecast_date=row["forecast_date"],
            ticker=row["ticker"],
            horizon=row["horizon"],
            target_date=row["target_date"],
            model_name=row["model_name"],
            predicted_volatility=float(row["predicted_volatility"]),
            recent_realized_volatility=float(row["recent_realized_volatility"]),
            origin_price=float(row["origin_price"]),
            lower_scenario_price=float(row["lower_scenario_price"]),
            upper_scenario_price=float(row["upper_scenario_price"]),
            actual_realized_volatility=(
                float(row["actual_realized_volatility"])
                if row["actual_realized_volatility"] is not None
                else None
            ),
            forecast_error=(
                float(row["forecast_error"]) if row["forecast_error"] is not None else None
            ),
            abs_error=float(row["abs_error"]) if row["abs_error"] is not None else None,
            qlike_loss=float(row["qlike_loss"]) if row["qlike_loss"] is not None else None,
            status=row["status"],
            record_source=row["record_source"] if "record_source" in keys else "live",
            model_version=row["model_version"] if "model_version" in keys else "deployable_v5",
            feature_set_version=(
                row["feature_set_version"]
                if "feature_set_version" in keys
                else "deployable_feature_columns_v5"
            ),
            code_commit=row["code_commit"] if "code_commit" in keys else "dev-local",
            data_as_of=row["data_as_of"] if "data_as_of" in keys else "",
            data_provider=row["data_provider"] if "data_provider" in keys else "unknown",
            forecast_fingerprint=(
                row["forecast_fingerprint"] if "forecast_fingerprint" in keys else ""
            ),
            created_at=row["created_at"],
        )


_DEFAULT_LEDGER: ForecastLedger | None = None
_DEFAULT_LEDGER_CONFIG: tuple[str | None, bool] | None = None


def get_forecast_ledger(
    db_path: Path | str | None = None,
    *,
    database_url: str | None = None,
    database_required: bool | None = None,
) -> ForecastLedger:
    """Return the process-shared default ledger or an explicit backend.

    Explicit paths/URLs are never cached, which keeps tests and migration
    commands isolated.  The default object is recreated if deployment
    configuration changes during a process lifetime (useful for test harnesses
    and controlled reloads).
    """
    global _DEFAULT_LEDGER, _DEFAULT_LEDGER_CONFIG
    if db_path is not None or database_url is not None or database_required is not None:
        return ForecastLedger(
            db_path,
            database_url=database_url,
            database_required=database_required,
        )

    resolved_url = ForecastLedger._resolve_database_url(None)
    resolved_required = ForecastLedger._resolve_database_required()
    config_signature = (resolved_url, resolved_required)
    if _DEFAULT_LEDGER is None or config_signature != _DEFAULT_LEDGER_CONFIG:
        _DEFAULT_LEDGER = ForecastLedger(
            database_url=resolved_url,
            database_required=resolved_required,
        )
        _DEFAULT_LEDGER_CONFIG = config_signature
    return _DEFAULT_LEDGER
