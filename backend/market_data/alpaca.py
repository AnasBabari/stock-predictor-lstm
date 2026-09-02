"""Alpaca historical daily-bar provider."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pandas as pd

from market_data.base import (
    MarketDataConfigurationError,
    MarketDataProviderError,
    MarketDataRateLimitError,
    MarketDataResult,
    MarketDataServiceError,
    MarketDataSymbolNotFound,
    MarketDataTimeoutError,
)
from market_data.normalization import normalize_daily_bars


class AlpacaProvider:
    """Fetch split/dividend-adjusted US equity bars from Alpaca."""

    name = "alpaca"

    def __init__(
        self,
        *,
        key_id: str | None,
        secret_key: str | None,
        base_url: str = "https://data.alpaca.markets",
        feed: str = "iex",
        adjustment: str = "all",
        timeout_seconds: float = 15.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.key_id = (key_id or "").strip()
        self.secret_key = (secret_key or "").strip()
        self.base_url = base_url.rstrip("/")
        self.feed = feed
        self.adjustment = adjustment
        self.timeout_seconds = timeout_seconds
        self._client = client

    @property
    def configured(self) -> bool:
        return bool(self.key_id and self.secret_key)

    @staticmethod
    def _response_reports_unknown_symbol(response: httpx.Response) -> bool:
        """Recognize Alpaca's symbol-level 400/422 errors without leaking text."""
        if response.status_code == 404:
            return True
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        text = " ".join(
            str(payload.get(key, ""))
            for key in ("code", "message", "error")
            if isinstance(payload, dict)
        ).lower()
        return any(
            phrase in text
            for phrase in ("symbol not found", "unknown symbol", "invalid symbol", "no data")
        )

    def fetch_daily_bars(self, symbol: str, *, years: int) -> MarketDataResult:
        if not self.configured:
            raise MarketDataConfigurationError("Alpaca market-data credentials are not configured")
        end = datetime.now(UTC)
        start = end - timedelta(days=max(1, years) * 366)
        params: dict[str, Any] = {
            "timeframe": "1Day",
            "start": start.isoformat().replace("+00:00", "Z"),
            "end": end.isoformat().replace("+00:00", "Z"),
            "adjustment": self.adjustment,
            "feed": self.feed,
            "sort": "asc",
            "limit": 10000,
        }
        headers = {
            "APCA-API-KEY-ID": self.key_id,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Accept": "application/json",
        }
        bars: list[dict[str, Any]] = []
        seen_tokens: set[str] = set()
        pages = 0
        client = self._client or httpx.Client(timeout=self.timeout_seconds)
        owns_client = self._client is None
        try:
            while True:
                pages += 1
                if pages > 10:
                    raise MarketDataProviderError("Alpaca pagination exceeded the safety limit")
                try:
                    response = client.get(
                        f"{self.base_url}/v2/stocks/{symbol}/bars",
                        params=params,
                        headers=headers,
                    )
                except httpx.TimeoutException as err:
                    raise MarketDataTimeoutError("Alpaca market-data request timed out") from err
                except httpx.HTTPError as err:
                    raise MarketDataServiceError("Alpaca market-data transport failed") from err
                if response.status_code == 429:
                    raise MarketDataRateLimitError("Alpaca market-data rate limit exceeded")
                if response.status_code in {
                    400,
                    404,
                    422,
                } and self._response_reports_unknown_symbol(response):
                    raise MarketDataSymbolNotFound(f"Alpaca has no data for {symbol}")
                if response.status_code in {401, 403}:
                    raise MarketDataConfigurationError(
                        "Alpaca market-data credentials were rejected"
                    )
                if response.status_code >= 500:
                    raise MarketDataServiceError("Alpaca market-data service is unavailable")
                if response.status_code >= 400:
                    raise MarketDataServiceError(
                        "Alpaca rejected the server-generated bars request"
                    )
                try:
                    payload = response.json()
                except ValueError as err:
                    raise MarketDataProviderError("Alpaca returned malformed JSON") from err
                page = payload.get("bars") if isinstance(payload, dict) else None
                if not isinstance(page, list):
                    raise MarketDataProviderError("Alpaca returned a malformed bars payload")
                bars.extend(item for item in page if isinstance(item, dict))
                if len(bars) > 100_000:
                    raise MarketDataProviderError("Alpaca bars payload exceeded the safety limit")
                token = payload.get("next_page_token")
                if not token:
                    break
                token_str = str(token)
                if token_str in seen_tokens:
                    raise MarketDataProviderError("Alpaca returned a repeated pagination token")
                seen_tokens.add(token_str)
                params["page_token"] = token_str
        finally:
            if owns_client:
                client.close()
        if not bars:
            raise MarketDataSymbolNotFound(f"Alpaca has no daily bars for {symbol}")
        try:
            frame = pd.DataFrame(
                {
                    "Open": [bar["o"] for bar in bars],
                    "High": [bar["h"] for bar in bars],
                    "Low": [bar["l"] for bar in bars],
                    "Close": [bar["c"] for bar in bars],
                    "Volume": [bar["v"] for bar in bars],
                },
                index=[bar["t"] for bar in bars],
            )
        except (KeyError, TypeError, ValueError) as err:
            raise MarketDataProviderError("Alpaca returned incomplete daily bars") from err
        normalized = normalize_daily_bars(frame, provider=self.name, symbol=symbol)
        return MarketDataResult(
            frame=normalized,
            provider=self.name,
            data_as_of=normalized.index[-1].date().isoformat(),
        )
