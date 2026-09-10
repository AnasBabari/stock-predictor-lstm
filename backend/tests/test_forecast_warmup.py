"""Guards for the background forecast warm-up.

Warm-up exists to move a one-time cost (PyTorch import plus model fitting,
~9s measured) off the first request path. These tests pin the properties that
make it safe: it must never run inside tests, never start twice, and must
respect its configuration bounds.
"""

from __future__ import annotations

import threading

import pytest

from services import forecast_warmup


@pytest.fixture(autouse=True)
def _reset_started_flag():
    """Each test starts from a clean module state."""
    forecast_warmup._started = False
    yield
    forecast_warmup._started = False


def test_start_is_suppressed_under_pytest():
    """Tests build the app with TestClient, which fires startup events.

    Training models there would be slow and flaky, so warm-up must refuse.
    """
    assert forecast_warmup._running_under_pytest() is True
    assert forecast_warmup.start_forecast_warmup() is False


def test_start_respects_the_enabled_flag(monkeypatch):
    monkeypatch.setattr(forecast_warmup, "_running_under_pytest", lambda: False)
    monkeypatch.setattr("config.settings.forecast_warmup_enabled", False, raising=False)
    assert forecast_warmup.start_forecast_warmup() is False


def test_start_respects_a_zero_ticker_budget(monkeypatch):
    monkeypatch.setattr(forecast_warmup, "_running_under_pytest", lambda: False)
    monkeypatch.setattr("config.settings.forecast_warmup_enabled", True, raising=False)
    monkeypatch.setattr("config.settings.forecast_warmup_max_tickers", 0, raising=False)
    assert forecast_warmup.start_forecast_warmup() is False


def test_start_launches_one_daemon_thread_and_is_idempotent(monkeypatch):
    monkeypatch.setattr(forecast_warmup, "_running_under_pytest", lambda: False)
    monkeypatch.setattr("config.settings.forecast_warmup_enabled", True, raising=False)
    monkeypatch.setattr("config.settings.forecast_warmup_max_tickers", 2, raising=False)

    warmed: list[str] = []
    monkeypatch.setattr(
        forecast_warmup, "_warm_one", lambda ticker: (warmed.append(ticker), 0.0)[1]
    )
    monkeypatch.setattr(forecast_warmup, "_warm_volatility", lambda _ticker: 0.0)

    assert forecast_warmup.start_forecast_warmup() is True
    # Second call must not spawn a duplicate thread.
    assert forecast_warmup.start_forecast_warmup() is False

    for thread in threading.enumerate():
        if thread.name == "forecast-warmup":
            thread.join(timeout=5)

    assert warmed == list(forecast_warmup.DEFAULT_WARMUP_TICKERS[:2])


def test_warm_loop_survives_a_failing_ticker(monkeypatch):
    """A warm-up failure must never propagate: the endpoint still trains
    on demand, so a broken warm-up may only cost latency."""
    seen: list[str] = []

    def flaky(ticker: str) -> float:
        seen.append(ticker)
        if ticker == "MSFT":
            raise RuntimeError("upstream unavailable")
        return 0.0

    monkeypatch.setattr(forecast_warmup, "_warm_one", flaky)
    monkeypatch.setattr(forecast_warmup, "_warm_volatility", lambda _ticker: 0.0)
    # Must not raise.
    forecast_warmup._warm_loop(("AAPL", "MSFT", "NVDA"))
    assert seen == ["AAPL", "MSFT", "NVDA"]


def test_warm_loop_also_prepares_volatility(monkeypatch):
    """The volatility snapshot costs ~2-4s and is shared by horizons 5/10/20,
    so the first visitor should not pay it."""
    warmed: list[str] = []
    monkeypatch.setattr(forecast_warmup, "_warm_one", lambda _ticker: 0.0)
    monkeypatch.setattr(
        forecast_warmup, "_warm_volatility", lambda ticker: (warmed.append(ticker), 0.0)[1]
    )
    forecast_warmup._warm_loop(("AAPL", "MSFT"))
    assert warmed == ["AAPL", "MSFT"]


def test_volatility_warmup_can_be_disabled(monkeypatch):
    monkeypatch.setattr(forecast_warmup, "_warm_one", lambda _ticker: 0.0)
    monkeypatch.setattr(forecast_warmup, "_warm_volatility", lambda _ticker: 0.0)
    monkeypatch.setattr("config.settings.forecast_warmup_volatility", False, raising=False)
    # Must not raise; the volatility step is simply skipped.
    forecast_warmup._warm_loop(("AAPL",))


def test_volatility_warmup_failure_does_not_stop_the_loop(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(forecast_warmup, "_warm_one", lambda _ticker: 0.0)

    def flaky(ticker: str) -> float:
        seen.append(ticker)
        raise RuntimeError("snapshot unavailable")

    monkeypatch.setattr(forecast_warmup, "_warm_volatility", flaky)
    forecast_warmup._warm_loop(("AAPL", "MSFT"))
    assert seen == ["AAPL", "MSFT"]


def test_default_tickers_are_bounded_and_real():
    """Warm-up is deliberately bounded: the supported universe is ~286
    tickers and fitting all of them would waste minutes of CPU."""
    defaults = forecast_warmup.DEFAULT_WARMUP_TICKERS
    assert 1 <= len(defaults) <= 16
    assert len(set(defaults)) == len(defaults)
