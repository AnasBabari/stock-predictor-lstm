"""Bounded Yahoo Finance adapter for local development and optional fallback."""

from __future__ import annotations

import pandas as pd
import yfinance as yf  # type: ignore[import-untyped]

from market_data.base import (
    MarketDataProviderError,
    MarketDataRateLimitError,
    MarketDataResult,
    MarketDataServiceError,
    MarketDataSymbolNotFound,
    MarketDataTimeoutError,
)
from market_data.normalization import normalize_daily_bars

try:
    from yfinance.exceptions import (  # type: ignore[import-untyped]
        YFPricesMissingError,
        YFRateLimitError,
        YFTickerMissingError,
        YFTzMissingError,
    )
except ImportError:  # pragma: no cover - compatibility with older yfinance
    YFRateLimitError = type("YFRateLimitError", (Exception,), {})
    YFPricesMissingError = type("YFPricesMissingError", (Exception,), {})
    YFTickerMissingError = type("YFTickerMissingError", (Exception,), {})
    YFTzMissingError = type("YFTzMissingError", (Exception,), {})


def _classify_yahoo_error(err: Exception) -> MarketDataProviderError:
    message = str(err).lower()
    if isinstance(err, (YFPricesMissingError, YFTickerMissingError, YFTzMissingError)):
        return MarketDataSymbolNotFound("Yahoo has no data for this symbol")
    if (
        isinstance(err, YFRateLimitError)
        or "rate limit" in message
        or "too many requests" in message
    ):
        return MarketDataRateLimitError("Yahoo market-data rate limit exceeded")
    if isinstance(err, TimeoutError) or "timeout" in message or "timed out" in message:
        return MarketDataTimeoutError("Yahoo market-data request timed out")
    return MarketDataServiceError("Yahoo market-data request failed")


class YahooProvider:
    """Fetch adjusted Yahoo bars with one bounded fallback request."""

    name = "yahoo"

    @property
    def configured(self) -> bool:
        return True

    def fetch_daily_bars(self, symbol: str, *, years: int) -> MarketDataResult:
        first_error: Exception | None = None
        try:
            data = yf.download(
                symbol,
                period=f"{years}y",
                progress=False,
                auto_adjust=True,
                timeout=15,
                threads=False,
            )
        except Exception as err:
            first_error = err
            classified = _classify_yahoo_error(err)
            if isinstance(classified, (MarketDataRateLimitError, MarketDataTimeoutError)):
                raise classified from err
            data = None
        if data is None or not isinstance(data, pd.DataFrame) or data.empty:
            try:
                data = yf.Ticker(symbol).history(
                    period=f"{years}y", auto_adjust=True, timeout=15, raise_errors=True
                )
            except Exception as err:
                classified = _classify_yahoo_error(err)
                raise classified from err
        if data is None or not isinstance(data, pd.DataFrame) or data.empty:
            if first_error is not None:
                raise _classify_yahoo_error(first_error) from first_error
            raise MarketDataSymbolNotFound(f"Yahoo has no daily bars for {symbol}")
        try:
            normalized = normalize_daily_bars(data, provider=self.name, symbol=symbol)
        except MarketDataProviderError as err:
            raise MarketDataServiceError("Yahoo returned unusable OHLCV data") from err
        return MarketDataResult(
            frame=normalized,
            provider=self.name,
            data_as_of=normalized.index[-1].date().isoformat(),
        )
