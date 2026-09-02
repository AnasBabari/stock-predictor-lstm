"""Market-data provider chain, cache policy, and readiness state."""

from __future__ import annotations

import threading
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from calendars import latest_completed_trading_session
from market_data.alpaca import AlpacaProvider
from market_data.base import (
    MarketDataProvider,
    MarketDataProviderError,
    MarketDataResult,
    MarketDataServiceError,
    MarketDataSymbolNotFound,
)
from market_data.cache import MarketDataCache
from market_data.yahoo import YahooProvider


class MarketDataService:
    """Resolve bars from fresh cache or a bounded provider chain."""

    def __init__(
        self,
        providers: Iterable[MarketDataProvider],
        *,
        cache: MarketDataCache,
    ) -> None:
        self.providers = tuple(providers)
        self.cache = cache
        self._lock = threading.RLock()
        self._last_success_session: str | None = None
        self._last_success_at: str | None = None
        self._last_provider: str | None = None
        self._last_error: str | None = None

    def fetch_daily_bars(self, symbol: str, *, years: int) -> MarketDataResult:
        required_session = latest_completed_trading_session().date().isoformat()
        failures: list[MarketDataProviderError] = []
        unknowns: list[MarketDataSymbolNotFound] = []
        for provider in self.providers:
            cached = self.cache.load(provider.name, symbol, required_session=required_session)
            if cached is not None:
                self._record_success(cached)
                return cached
            if not provider.configured:
                continue
            try:
                result = provider.fetch_daily_bars(symbol, years=years)
            except MarketDataSymbolNotFound as err:
                unknowns.append(err)
                continue
            except MarketDataProviderError as err:
                failures.append(err)
                continue
            if result.data_as_of < required_session:
                failures.append(
                    MarketDataServiceError(
                        f"{provider.name} did not provide the latest completed session"
                    )
                )
                continue
            self.cache.save(symbol, result)
            self._record_success(result)
            return result
        if failures:
            error = MarketDataServiceError("All configured market-data providers are unavailable")
            self._record_failure(error)
            raise error from failures[-1]
        if unknowns:
            self._record_failure(unknowns[-1])
            raise MarketDataSymbolNotFound(
                f"No configured provider has data for {symbol}"
            ) from unknowns[-1]
        error = MarketDataServiceError("No market-data provider is configured")
        self._record_failure(error)
        raise error

    def _record_success(self, result: MarketDataResult) -> None:
        with self._lock:
            self._last_success_session = result.data_as_of
            self._last_success_at = datetime.now(UTC).isoformat()
            self._last_provider = result.provider
            self._last_error = None

    def _record_failure(self, error: Exception) -> None:
        with self._lock:
            self._last_error = type(error).__name__
            if isinstance(error, MarketDataServiceError):
                self._last_success_session = None

    def readiness(self) -> tuple[bool, dict[str, Any]]:
        required_session = latest_completed_trading_session().date().isoformat()
        fresh_entries = self.cache.fresh_entry_count(required_session=required_session)
        configured = [provider.name for provider in self.providers if provider.configured]
        with self._lock:
            recent_success = bool(
                self._last_success_session and self._last_success_session >= required_session
            )
            ready = bool(recent_success or fresh_entries > 0)
            return ready, {
                "status": "available" if ready else "unavailable",
                "configured_providers": configured,
                "last_provider": self._last_provider,
                "last_success_at": self._last_success_at,
                "last_success_session": self._last_success_session,
                "required_session": required_session,
                "fresh_cache_entries": fresh_entries,
                "cache_persistence": "ephemeral" if self.cache.enabled else "disabled",
                "last_error": self._last_error,
            }


def build_market_data_service(settings: Any) -> MarketDataService:
    """Build the configured provider chain without exposing credentials."""
    requested = str(settings.market_data_provider).strip().lower()
    providers: list[MarketDataProvider] = []
    if requested in {"alpaca", "auto"}:
        providers.append(
            AlpacaProvider(
                key_id=settings.alpaca_api_key_id,
                secret_key=settings.alpaca_api_secret_key,
                base_url=settings.alpaca_data_base_url,
                feed=settings.alpaca_data_feed,
                adjustment=settings.alpaca_adjustment,
                timeout_seconds=settings.market_data_timeout_seconds,
            )
        )
    if requested in {"yahoo", "auto"} or settings.market_data_yahoo_fallback_enabled:
        providers.append(YahooProvider())
    cache_dir = Path(settings.market_data_cache_dir) if settings.market_data_cache_enabled else None
    return MarketDataService(
        providers,
        cache=MarketDataCache(cache_dir, enabled=settings.market_data_cache_enabled),
    )
