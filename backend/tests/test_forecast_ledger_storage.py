from __future__ import annotations

import pytest

from services.forecast_ledger import ForecastLedger, LedgerUnavailableError, _LedgerConnection


class _Cursor:
    rowcount = 0

    def fetchone(self):
        return None

    def fetchall(self):
        return []


class _RawConnection:
    def __init__(self) -> None:
        self.queries: list[tuple[str, tuple[object, ...]]] = []
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def execute(self, query: str, params=()):
        self.queries.append((query, tuple(params)))
        return _Cursor()

    def executemany(self, query: str, params):
        self.queries.append((query, tuple()))
        return _Cursor()

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = True


def test_postgres_adapter_translates_parameters_without_interpolation() -> None:
    raw = _RawConnection()
    adapter = _LedgerConnection(raw, "postgres")

    adapter.execute("SELECT * FROM forecast_ledger WHERE ticker=? AND horizon=?", ["MSFT", 5])

    assert raw.queries == [
        ("SELECT * FROM forecast_ledger WHERE ticker=%s AND horizon=%s", ("MSFT", 5))
    ]


def test_postgres_url_is_validated_before_connection_attempt() -> None:
    with pytest.raises(LedgerUnavailableError, match="PostgreSQL connection URL"):
        ForecastLedger(database_url="mysql://user:password@example/ledger")


def test_configured_postgres_never_falls_back_to_sqlite(monkeypatch) -> None:
    import psycopg

    def fail_connect(*args, **kwargs):
        raise OSError("database is offline")

    monkeypatch.setattr(psycopg, "connect", fail_connect)
    with pytest.raises(LedgerUnavailableError, match="PostgreSQL forecast ledger is unavailable"):
        ForecastLedger(database_url="postgresql://user:password@example/ledger")
