"""Background warm-up for the learned price-forecast path.

The first forecast for a ticker in a fresh process pays two one-time costs that
have nothing to do with that ticker:

1. importing PyTorch, which alone measured ~2s cold and pulled in ~4s of
   ``io.open_code`` module loading across the whole first request;
2. fitting and selecting the sklearn candidates plus the GPU-LSTM validation
   pass, roughly 2-3s.

Afterwards the artifact cache answers in ~0.2s, but the cache key embeds the
last data date, so the cost returns on the first request of every new trading
day and after every restart.

Warming this path in a daemon thread moves the cost off the request path. It
never trains inside a request and never delays readiness: the thread is a
daemon, every failure is logged and swallowed, and the endpoint still trains on
demand if warm-up has not finished or was disabled.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections.abc import Iterable

logger = logging.getLogger(__name__)

# The most frequently requested tickers. Warm-up is bounded by design: the
# supported universe is ~286 tickers, and fitting all of them would burn
# minutes of CPU for requests that may never arrive.
DEFAULT_WARMUP_TICKERS: tuple[str, ...] = (
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA",
    "AMD",
)

_started = False
_start_lock = threading.Lock()


def _running_under_pytest() -> bool:
    """Tests build the app with TestClient, which fires startup events.

    Training models inside every test would be slow and flaky, so warm-up is
    suppressed whenever pytest is the host process.
    """
    return "pytest" in sys.modules


def _warm_one(ticker: str) -> float:
    """Fit and cache one ticker. Returns elapsed seconds."""
    # Imported here so this module stays cheap to import and so the heavy
    # route/service graph is only touched from the worker thread.
    from routes.simple_forecast import _download_ohlcv
    from services.simple_forecast import train_and_forecast

    started = time.perf_counter()
    frame = _download_ohlcv(ticker)
    train_and_forecast(ticker, frame)
    return time.perf_counter() - started


def _warm_volatility(ticker: str) -> float:
    """Build and cache the volatility snapshot for one ticker.

    The serving UI asks for horizons 5/10/20, which share one snapshot. That
    snapshot costs ~2-4s to derive, so the first visitor would otherwise wait
    for it. Building it here moves that cost off the request path; the cache is
    keyed on frame content, so a later data refresh still rebuilds.
    """
    from services.volatility_snapshot import build_volatility_inference_snapshot

    started = time.perf_counter()
    build_volatility_inference_snapshot(ticker)
    return time.perf_counter() - started


def _warm_loop(tickers: Iterable[str]) -> None:
    # Eager, guarded import: this is the single largest fixed cost, and doing it
    # here means no user request ever pays it. Guarded because torch is an
    # optional dependency in some CI configurations.
    try:
        import torch  # noqa: F401
    except Exception as err:  # pragma: no cover - environment dependent
        logger.info("forecast_warmup: torch unavailable (%s); continuing", err)

    from config import settings

    warm_volatility = bool(getattr(settings, "forecast_warmup_volatility", True))

    for ticker in tickers:
        try:
            elapsed = _warm_one(ticker)
            logger.info("forecast_warmup: warmed %s in %.2fs", ticker, elapsed)
        except Exception as err:  # noqa: BLE001 - warm-up must never crash the app
            logger.info("forecast_warmup: skipped %s (%s)", ticker, err)
        if not warm_volatility:
            continue
        try:
            elapsed = _warm_volatility(ticker)
            logger.info("forecast_warmup: warmed volatility %s in %.2fs", ticker, elapsed)
        except Exception as err:  # noqa: BLE001 - warm-up must never crash the app
            logger.info("forecast_warmup: skipped volatility %s (%s)", ticker, err)
    logger.info("forecast_warmup: complete")


def start_forecast_warmup() -> bool:
    """Start the warm-up thread. Returns True when it was started.

    Idempotent: repeated calls (e.g. several TestClient instances, or a reload)
    never start a second thread.
    """
    global _started
    from config import settings

    if not getattr(settings, "forecast_warmup_enabled", True):
        logger.info("forecast_warmup: disabled by configuration")
        return False
    if _running_under_pytest():
        return False

    limit = int(getattr(settings, "forecast_warmup_max_tickers", 8) or 0)
    if limit <= 0:
        return False

    configured = getattr(settings, "forecast_warmup_tickers", None)
    candidates = tuple(configured) if configured else DEFAULT_WARMUP_TICKERS

    with _start_lock:
        if _started:
            return False
        _started = True

    thread = threading.Thread(
        target=_warm_loop,
        args=(candidates[:limit],),
        name="forecast-warmup",
        daemon=True,
    )
    thread.start()
    logger.info("forecast_warmup: started for %s", ", ".join(candidates[:limit]))
    return True
