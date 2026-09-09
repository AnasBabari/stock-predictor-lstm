from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from api import app
from market_data.base import MarketDataResult, MarketDataServiceError, MarketDataSymbolNotFound
from routes import market

CLIENT = TestClient(app)


def _bars(n: int, start: str = "2015-01-02") -> pd.DataFrame:
    index = pd.bdate_range(start, periods=n)
    close = 100.0 + np.arange(n) * 0.1
    return pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 0.5,
            "Low": close - 1.0,
            "Close": close,
            "Volume": 1_000_000,
        },
        index=index,
    )


def _result(n: int, provider: str = "alpaca") -> MarketDataResult:
    frame = _bars(n)
    return MarketDataResult(
        frame=frame, provider=provider, data_as_of=frame.index[-1].date().isoformat()
    )


def test_history_returns_daily_and_intraday(monkeypatch):
    monkeypatch.setattr(market, "_fetch_history_daily", lambda symbol: _result(300))
    monkeypatch.setattr(
        market,
        "_fetch_history_intraday",
        lambda symbol: [{"t": "2026-09-04T14:30:00+00:00", "c": 500.0}],
    )
    response = CLIENT.get("/api/v1/history", params={"ticker": "msft"})
    assert response.status_code == 200
    body = response.json()
    assert body["ticker"] == "MSFT"
    assert body["provider"] == "alpaca"
    assert len(body["daily"]) == 300
    assert body["daily"][-1]["c"] == 100.0 + 299 * 0.1
    assert body["first_date"] == body["daily"][0]["d"]
    assert body["intraday"] == [{"t": "2026-09-04T14:30:00+00:00", "c": 500.0}]
    assert body["intraday_session"] == body["as_of"]
    assert body["market_data_cache"] in ("hit", "miss")


def test_history_downsamples_long_series_but_keeps_latest_bar(monkeypatch):
    monkeypatch.setattr(market, "_fetch_history_daily", lambda symbol: _result(3000))
    monkeypatch.setattr(market, "_fetch_history_intraday", lambda symbol: None)
    body = CLIENT.get("/api/v1/history", params={"ticker": "MSFT"}).json()
    assert len(body["daily"]) <= 1500
    assert body["daily"][-1]["c"] == 100.0 + 2999 * 0.1
    assert body["intraday"] is None
    assert body["intraday_session"] is None


def test_history_routes_lse_tickers_to_yahoo(monkeypatch):
    calls: list[str] = []

    def fake_yahoo(self, symbol: str, *, years: int):
        calls.append(f"yahoo:{symbol}:{years}")
        return _result(60, provider="yahoo")

    def fail_service(symbol: str, *, years: int):
        raise AssertionError("shared forecast service must not serve LSE history")

    monkeypatch.setattr("market_data.yahoo.YahooProvider.fetch_daily_bars", fake_yahoo)
    monkeypatch.setattr("data_pipeline.market_data_service.fetch_daily_bars", fail_service)
    body = CLIENT.get("/api/v1/history", params={"ticker": "SHEL.L"}).json()
    assert body["provider"] == "yahoo"
    assert calls == ["yahoo:SHEL.L:10"]
    assert len(body["daily"]) == 60


def test_history_unknown_ticker_is_404(monkeypatch):
    def missing(symbol: str):
        raise MarketDataSymbolNotFound("no data")

    monkeypatch.setattr(market, "_fetch_history_daily", missing)
    response = CLIENT.get("/api/v1/history", params={"ticker": "NOTREAL"})
    assert response.status_code == 404


def test_history_provider_failure_is_503(monkeypatch):
    def failing(symbol: str):
        raise MarketDataServiceError("upstream down")

    monkeypatch.setattr(market, "_fetch_history_daily", failing)
    response = CLIENT.get("/api/v1/history", params={"ticker": "MSFT"})
    assert response.status_code == 503


def test_history_warms_shared_cache_for_learned_forecast(monkeypatch, tmp_path):
    """Same invariant through the training endpoint: chart load first, then
    GET /api/v1/forecast must not refetch bars. The learned forecast keeps
    its exact values; only the redundant upstream fetch disappears."""
    import data_pipeline
    from calendars import latest_completed_trading_session
    from market_data.cache import MarketDataCache
    from market_data.service import MarketDataService

    required = latest_completed_trading_session().date().isoformat()
    calls = {"upstream_fetches": 0}

    class CountingProvider:
        name = "alpaca"

        @property
        def configured(self) -> bool:
            return True

        def fetch_daily_bars(self, symbol: str, *, years: int):
            calls["upstream_fetches"] += 1
            rng = np.random.default_rng(7)
            index = pd.bdate_range(end=required, periods=900)
            index = index[index <= required][-800:]
            close = 500 * np.exp(np.cumsum(0.0003 + rng.normal(0, 0.01, len(index))))
            frame = pd.DataFrame(
                {
                    "Open": close * np.exp(rng.normal(0, 0.002, len(index))),
                    "High": close * 1.01,
                    "Low": close * 0.99,
                    "Close": close,
                    "Volume": rng.integers(100_000, 1_000_000, len(index)),
                },
                index=index,
            )
            return MarketDataResult(frame=frame, provider="alpaca", data_as_of=required)

    service = MarketDataService([CountingProvider()], cache=MarketDataCache(tmp_path / "mdcache"))
    monkeypatch.setattr(data_pipeline, "market_data_service", service)
    monkeypatch.setattr(market, "_fetch_history_intraday", lambda symbol: None)

    history = CLIENT.get("/api/v1/history", params={"ticker": "MSFT"})
    assert history.status_code == 200
    assert calls["upstream_fetches"] == 1

    forecast = CLIENT.get("/api/v1/forecast", params={"ticker": "MSFT", "days": 7})
    assert forecast.status_code == 200
    assert forecast.json()["ticker"] == "MSFT"
    assert calls["upstream_fetches"] == 1
    assert "server-timing" in forecast.headers


def test_history_warms_shared_cache_for_volatility_forecast(monkeypatch, tmp_path):
    """Performance invariant: one upstream fetch serves both chart and forecast.

    GET /history must populate the exact cache entries the volatility
    forecast reads, so a forecast issued right after a chart load performs
    zero additional upstream fetches. If a refactor splits the caches, this
    test fails before users pay for the second cold fetch.
    """
    import data_pipeline
    from calendars import latest_completed_trading_session
    from market_data.cache import MarketDataCache
    from market_data.service import MarketDataService

    required = latest_completed_trading_session().date().isoformat()
    calls = {"upstream_fetches": 0}

    class CountingProvider:
        name = "alpaca"

        @property
        def configured(self) -> bool:
            return True

        def fetch_daily_bars(self, symbol: str, *, years: int):
            calls["upstream_fetches"] += 1
            index = pd.bdate_range(end=required, periods=600)
            index = index[index <= required][-500:]
            close = 500.0 + np.arange(len(index)) * 0.05
            frame = pd.DataFrame(
                {
                    "Open": close - 0.2,
                    "High": close + 0.3,
                    "Low": close - 0.4,
                    "Close": close,
                    "Volume": 2_000_000,
                },
                index=index,
            )
            return MarketDataResult(frame=frame, provider="alpaca", data_as_of=required)

    service = MarketDataService([CountingProvider()], cache=MarketDataCache(tmp_path / "mdcache"))
    monkeypatch.setattr(data_pipeline, "market_data_service", service)
    monkeypatch.setattr(market, "_fetch_history_intraday", lambda symbol: None)

    history = CLIENT.get("/api/v1/history", params={"ticker": "MSFT"})
    assert history.status_code == 200
    assert history.json()["provider"] == "alpaca"
    assert calls["upstream_fetches"] == 1

    forecast = CLIENT.get(
        "/api/v1/volatility/forecast",
        params={"ticker": "MSFT", "horizon": 5, "model": "auto"},
    )
    assert forecast.status_code == 200
    assert forecast.json()["evidence"]["data_as_of"] == required
    assert calls["upstream_fetches"] == 1
