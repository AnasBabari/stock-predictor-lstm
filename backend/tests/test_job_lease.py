"""Job lease lifecycle: a dead worker must not strand jobs in 'processing'."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from server_models.db import MAX_JOB_ATTEMPTS, InMemoryRegistry


class _FakeClock:
    def __init__(self) -> None:
        self._now = datetime(2026, 1, 1, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)


def test_dequeue_claims_with_lease_and_attempts() -> None:
    registry = InMemoryRegistry()
    job_id = registry.enqueue_job("AAPL")
    clock = _FakeClock()
    registry._clock = clock

    job = registry.dequeue_job(lease_seconds=600)
    assert job is not None
    assert str(job["id"]) == job_id
    assert job["status"] == "processing"
    assert job["attempts"] == 1
    assert job["lease_expires_at"] == clock() + timedelta(seconds=600)
    # No second claim while the lease is live.
    assert registry.dequeue_job() is None


def test_expired_lease_is_reclaimed_and_retried() -> None:
    registry = InMemoryRegistry()
    job_id = registry.enqueue_job("MSFT")
    clock = _FakeClock()
    registry._clock = clock

    first = registry.dequeue_job(lease_seconds=100)
    assert first is not None and first["attempts"] == 1

    # Worker dies; time passes beyond the lease.
    clock.advance(101)
    second = registry.dequeue_job(lease_seconds=100)
    assert second is not None
    assert str(second["id"]) == job_id
    assert second["status"] == "processing"
    assert second["attempts"] == 2


def test_lease_reclaim_fails_job_after_max_attempts() -> None:
    registry = InMemoryRegistry()
    job_id = registry.enqueue_job("TSLA")
    clock = _FakeClock()
    registry._clock = clock

    for expected_attempt in range(1, MAX_JOB_ATTEMPTS + 1):
        clock.advance(1000)
        job = registry.dequeue_job(lease_seconds=100)
        assert job is not None, f"attempt {expected_attempt} should have been reclaimable"
        assert job["attempts"] == expected_attempt

    # Next dequeue after expiry must fail the job instead of looping forever.
    clock.advance(1000)
    assert registry.dequeue_job(lease_seconds=100) is None
    stored = next(j for j in registry._jobs if str(j["id"]) == job_id)
    assert stored["status"] == "failed"
    assert stored["last_error"] == "lease expired after max attempts"


def test_complete_and_fail_clear_the_lease() -> None:
    registry = InMemoryRegistry()
    clock = _FakeClock()
    registry._clock = clock

    done_id = registry.enqueue_job("SPY")
    fail_id = registry.enqueue_job("QQQ")

    done = registry.dequeue_job(lease_seconds=100)
    assert done is not None
    registry.complete_job(done_id)
    stored_done = next(j for j in registry._jobs if str(j["id"]) == done_id)
    assert stored_done["status"] == "completed"
    assert stored_done["lease_expires_at"] is None

    failed = registry.dequeue_job(lease_seconds=100)
    assert failed is not None
    registry.fail_job(fail_id, "boom")
    stored_failed = next(j for j in registry._jobs if str(j["id"]) == fail_id)
    assert stored_failed["status"] == "failed"
    assert stored_failed["last_error"] == "boom"
    assert stored_failed["lease_expires_at"] is None
