"""Provider-neutral market-data contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


class MarketDataProviderError(RuntimeError):
    """Base class for provider failures."""


class MarketDataServiceError(MarketDataProviderError):
    """The provider is temporarily unable to serve data."""


class MarketDataRateLimitError(MarketDataServiceError):
    """The provider rejected the request because a quota was exceeded."""


class MarketDataTimeoutError(MarketDataServiceError):
    """The provider did not answer within the configured deadline."""


class MarketDataConfigurationError(MarketDataServiceError):
    """Required server-side provider configuration is absent or invalid."""


class MarketDataSymbolNotFound(MarketDataProviderError):
    """A provider authoritatively reported that a symbol has no data."""


@dataclass(frozen=True)
class MarketDataResult:
    """Normalized daily OHLCV bars and their acquisition provenance."""

    frame: pd.DataFrame
    provider: str
    data_as_of: str
    cache_status: str = "miss"


class MarketDataProvider(Protocol):
    """Interface implemented by daily-bar providers."""

    name: str

    @property
    def configured(self) -> bool: ...

    def fetch_daily_bars(self, symbol: str, *, years: int) -> MarketDataResult: ...
