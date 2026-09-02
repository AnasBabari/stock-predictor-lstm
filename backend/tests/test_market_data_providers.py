from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import httpx
import pandas as pd
import pytest

from calendars import latest_completed_trading_session
from market_data.alpaca import AlpacaProvider
from market_data.base import (
    MarketDataProviderError,
    MarketDataRateLimitError,
    MarketDataResult,
    MarketDataServiceError,
    MarketDataSymbolNotFound,
    MarketDataTimeoutError,
)
from market_data.cache import MarketDataCache
from market_data.service import MarketDataService
from market_data.yahoo import YahooProvider


def _bars(last_date: str = "2026-08-31", rows: int = 100) -> pd.DataFrame:
    index = pd.bdate_range(end=last_date, periods=rows)
    return pd.DataFrame(
        {
            "Open": range(100, 100 + rows),
            "High": range(101, 101 + rows),
            "Low": range(99, 99 + rows),
            "Close": range(100, 100 + rows),
            "Volume": [1000] * rows,
        },
        index=index,
    )


def _alpaca(handler) -> AlpacaProvider:
    return AlpacaProvider(
        key_id="key",
        secret_key="secret",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_alpaca_success_uses_adjusted_iex_daily_bars() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["APCA-API-KEY-ID"] == "key"
        assert request.url.params["adjustment"] == "all"
        assert request.url.params["feed"] == "iex"
        assert request.url.params["timeframe"] == "1Day"
        return httpx.Response(
            200,
            json={
                "bars": [
                    {"t": "2026-08-31T04:00:00Z", "o": 100, "h": 102, "l": 99, "c": 101, "v": 5000}
                ],
                "next_page_token": None,
            },
        )

    result = _alpaca(handler).fetch_daily_bars("MSFT", years=8)
    assert result.provider == "alpaca"
    assert result.data_as_of == "2026-08-31"
    assert list(result.frame.columns) == ["Open", "High", "Low", "Close", "Volume"]


@pytest.mark.parametrize(
    ("status", "error_type"),
    [(429, MarketDataRateLimitError), (500, MarketDataServiceError), (503, MarketDataServiceError)],
)
def test_alpaca_transient_statuses_are_service_failures(status, error_type) -> None:
    provider = _alpaca(lambda _request: httpx.Response(status, json={"message": "safe"}))
    with pytest.raises(error_type):
        provider.fetch_daily_bars("MSFT", years=8)


def test_alpaca_timeout_is_not_unknown_symbol() -> None:
    def handler(_request):
        raise httpx.ReadTimeout("secret upstream detail")

    with pytest.raises(MarketDataTimeoutError):
        _alpaca(handler).fetch_daily_bars("MSFT", years=8)


@pytest.mark.parametrize("payload", [{}, {"bars": "bad"}, {"bars": [{"t": "2026-01-01"}]}])
def test_alpaca_malformed_payload_fails_closed(payload) -> None:
    provider = _alpaca(lambda _request: httpx.Response(200, json=payload))
    with pytest.raises(MarketDataProviderError):
        provider.fetch_daily_bars("MSFT", years=8)


def test_alpaca_empty_bars_are_authoritative_unknown() -> None:
    provider = _alpaca(lambda _request: httpx.Response(200, json={"bars": []}))
    with pytest.raises(MarketDataSymbolNotFound):
        provider.fetch_daily_bars("NOTREAL", years=8)


def test_alpaca_symbol_error_in_422_is_authoritative_unknown() -> None:
    provider = _alpaca(
        lambda _request: httpx.Response(422, json={"code": 40010001, "message": "symbol not found"})
    )
    with pytest.raises(MarketDataSymbolNotFound):
        provider.fetch_daily_bars("NOTREAL", years=8)


@pytest.mark.parametrize("bars", [[], {}])
def test_alpaca_empty_success_page_is_authoritative_unknown(bars) -> None:
    provider = _alpaca(
        lambda _request: httpx.Response(200, json={"bars": bars, "next_page_token": "stale"})
    )
    with pytest.raises(MarketDataSymbolNotFound):
        provider.fetch_daily_bars("NOTREAL", years=8)


def test_cache_hit_survives_provider_failure(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "market_data.service.latest_completed_trading_session",
        lambda: pd.Timestamp("2026-08-31"),
    )
    cache = MarketDataCache(tmp_path)
    cached = MarketDataResult(_bars(), "alpaca", "2026-08-31")
    cache.save("MSFT", cached)
    failing = SimpleNamespace(
        name="alpaca",
        configured=True,
        fetch_daily_bars=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            MarketDataServiceError("unavailable")
        ),
    )
    result = MarketDataService([failing], cache=cache).fetch_daily_bars("MSFT", years=8)
    assert result.cache_status == "hit"
    assert result.data_as_of == "2026-08-31"


def test_stale_cache_is_not_served_when_provider_fails(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "market_data.service.latest_completed_trading_session",
        lambda: pd.Timestamp("2026-09-01"),
    )
    cache = MarketDataCache(tmp_path)
    cache.save("MSFT", MarketDataResult(_bars(), "alpaca", "2026-08-31"))
    failing = SimpleNamespace(
        name="alpaca",
        configured=True,
        fetch_daily_bars=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            MarketDataServiceError("unavailable")
        ),
    )
    with pytest.raises(MarketDataServiceError):
        MarketDataService([failing], cache=cache).fetch_daily_bars("MSFT", years=8)


def test_stale_provider_result_is_not_served(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "market_data.service.latest_completed_trading_session",
        lambda: pd.Timestamp("2026-09-01"),
    )
    provider = SimpleNamespace(
        name="alpaca",
        configured=True,
        fetch_daily_bars=lambda *_args, **_kwargs: MarketDataResult(
            _bars(), "alpaca", "2026-08-31"
        ),
    )
    with pytest.raises(MarketDataServiceError):
        MarketDataService([provider], cache=MarketDataCache(tmp_path)).fetch_daily_bars(
            "MSFT", years=8
        )


def test_latest_completed_session_handles_weekend_and_before_close() -> None:
    friday = latest_completed_trading_session(datetime(2026, 8, 29, 12, tzinfo=UTC))
    before_close = latest_completed_trading_session(datetime(2026, 8, 31, 18, tzinfo=UTC))
    after_close = latest_completed_trading_session(datetime(2026, 8, 31, 21, tzinfo=UTC))
    assert friday == pd.Timestamp("2026-08-28")
    assert before_close == pd.Timestamp("2026-08-28")
    assert after_close == pd.Timestamp("2026-08-31")


def test_yahoo_rate_limit_is_not_misclassified_as_unknown(monkeypatch) -> None:
    from yfinance.exceptions import YFRateLimitError

    monkeypatch.setattr(
        "market_data.yahoo.yf.download",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(YFRateLimitError()),
    )
    with pytest.raises(MarketDataRateLimitError):
        YahooProvider().fetch_daily_bars("MSFT", years=8)


def test_service_readiness_requires_current_session_evidence(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "market_data.service.latest_completed_trading_session",
        lambda: pd.Timestamp("2026-08-31"),
    )
    provider = SimpleNamespace(name="alpaca", configured=True)
    service = MarketDataService([provider], cache=MarketDataCache(tmp_path))
    ready, details = service.readiness()
    assert ready is False
    assert details["configured_providers"] == ["alpaca"]
    service._record_success(MarketDataResult(_bars(), "alpaca", "2026-08-31"))
    ready, details = service.readiness()
    assert ready is True
    assert details["last_provider"] == "alpaca"


def test_data_pipeline_maps_provider_rate_limit_to_transport_error(monkeypatch) -> None:
    import data_pipeline

    data_pipeline.market_circuit_breaker.record_success("MSFT")
    monkeypatch.setattr(
        data_pipeline.market_data_service,
        "fetch_daily_bars",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            MarketDataRateLimitError("upstream private detail")
        ),
    )
    with pytest.raises(data_pipeline.MarketTransportError, match="temporarily unavailable"):
        data_pipeline._download_ohlcv("MSFT")


def test_data_pipeline_preserves_authoritative_unknown_symbol(monkeypatch) -> None:
    import data_pipeline

    data_pipeline.market_circuit_breaker.record_success("NOTREAL")
    monkeypatch.setattr(
        data_pipeline.market_data_service,
        "fetch_daily_bars",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            MarketDataSymbolNotFound("provider private detail")
        ),
    )
    with pytest.raises(data_pipeline.UnknownTickerError):
        data_pipeline._download_ohlcv("NOTREAL")
    # The negative cache must preserve 404 semantics on a repeat request.
    with pytest.raises(data_pipeline.UnknownTickerError):
        data_pipeline._download_ohlcv("NOTREAL")
