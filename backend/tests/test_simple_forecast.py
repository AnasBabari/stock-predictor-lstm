from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from api import app
from services.simple_forecast import FORECAST_DAYS, build_dataset, chronological_masks

CLIENT = TestClient(app)


def _frame(rows: int = 900) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    index = pd.bdate_range("2022-01-03", periods=rows)
    close = 100 * np.exp(np.cumsum(0.0003 + rng.normal(0, 0.01, rows)))
    return pd.DataFrame(
        {
            "Open": close * np.exp(rng.normal(0, 0.002, rows)),
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": rng.integers(100_000, 1_000_000, rows),
        },
        index=index,
    )


def test_chronological_split_purges_every_crossing_target() -> None:
    frame = _frame()
    dataset = build_dataset(frame)
    train, validation, test = chronological_masks(dataset, len(frame))
    train_boundary = int(len(frame) * 0.70)
    test_boundary = int(len(frame) * 0.85)

    assert np.all(dataset.origin_positions[train] + FORECAST_DAYS < train_boundary)
    assert np.all(dataset.origin_positions[validation] >= train_boundary)
    assert np.all(dataset.origin_positions[validation] + FORECAST_DAYS < test_boundary)
    assert np.all(dataset.origin_positions[test] >= test_boundary)
    assert not np.any(train & validation)
    assert not np.any(validation & test)


def test_forecast_cache_identity_tracks_bars_and_version(monkeypatch):
    from services import simple_forecast as sf

    frame = _frame(700)
    original = sf._forecast_cache_key("MSFT", frame, "ridge")
    revised = frame.copy()
    revised.iloc[-1, revised.columns.get_loc("Close")] += 1
    assert sf._forecast_cache_key("MSFT", revised, "ridge") != original
    assert sf._forecast_cache_key("MSFT", frame.iloc[:-1], "ridge") != original
    monkeypatch.setattr(sf, "MODEL_CACHE_VERSION", "test-new-version")
    assert sf._forecast_cache_key("MSFT", frame, "ridge") != original


def test_simultaneous_forecasts_train_once_and_results_are_isolated(monkeypatch, tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from services import simple_forecast as sf

    monkeypatch.setattr(sf.settings, "forecast_model_cache_dir", str(tmp_path))
    monkeypatch.setattr(sf, "_load_gpu_lstm_model", lambda: None)
    with sf._cache_lock:
        sf._cache.clear()
    original = sf.build_dataset
    calls = []

    def build(frame):
        calls.append(1)
        return original(frame)

    monkeypatch.setattr(sf, "build_dataset", build)
    frame = _frame(700)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: sf.train_and_forecast("MSFT", frame, "ridge"), range(2)))
    assert len(calls) == 1
    assert results[0] == results[1]
    results[0]["predicted_prices"][0] = -1
    assert sf.train_and_forecast("MSFT", frame, "ridge")["predicted_prices"][0] > 0


def test_history_then_forecast_reuses_daily_bars(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import data_pipeline
    from market_data.base import MarketDataResult
    from market_data.cache import MarketDataCache
    from market_data.service import MarketDataService
    from routes import market, simple_forecast

    frame = _frame(700)
    calls = []

    def fetch(symbol, *, years):
        calls.append((symbol, years))
        return MarketDataResult(frame, "fixture", str(frame.index[-1].date()), "miss")

    service = MarketDataService(
        [SimpleNamespace(name="fixture", configured=True, fetch_daily_bars=fetch)],
        cache=MarketDataCache(tmp_path),
    )
    monkeypatch.setattr(
        "market_data.service.latest_completed_trading_session", lambda **kwargs: frame.index[-1]
    )
    monkeypatch.setattr(data_pipeline, "market_data_service", service)
    monkeypatch.setattr(
        data_pipeline, "market_circuit_breaker", data_pipeline.MarketCircuitBreaker()
    )
    monkeypatch.setattr(market, "_fetch_history_intraday", lambda symbol: None)
    monkeypatch.setattr(
        simple_forecast,
        "train_and_forecast",
        lambda symbol, bars: {
            "ticker": symbol,
            "predicted_prices": [float(bars.Close.iloc[-1])] * 7,
        },
    )
    history = CLIENT.get("/api/v1/history?ticker=MSFT")
    assert history.status_code == 200
    forecast = CLIENT.get("/api/v1/forecast?ticker=MSFT")
    assert forecast.status_code == 200
    assert calls == [("MSFT", data_pipeline.HISTORICAL_YEARS)]


def test_forecast_route_supports_only_the_frozen_five_tickers() -> None:
    response = CLIENT.get("/api/v1/forecast?ticker=NMM&days=7")
    assert response.status_code == 400
    assert "AAPL" in response.json()["detail"]


def test_forecast_route_returns_learned_contract(monkeypatch) -> None:
    from routes import simple_forecast

    monkeypatch.setattr(simple_forecast, "_download_ohlcv", lambda _symbol: _frame())
    monkeypatch.setattr(
        simple_forecast,
        "train_and_forecast",
        lambda symbol, _frame: {
            "ticker": symbol,
            "forecast_days": 7,
            "predicted_prices": [101.0] * 7,
            "model": {"kind": "learned_historical_model"},
        },
    )
    response = CLIENT.get("/api/v1/forecast?ticker=MSFT&days=7")
    assert response.status_code == 200
    assert response.json()["model"]["kind"] == "learned_historical_model"
    assert len(response.json()["predicted_prices"]) == 7


def test_news_route_rejects_unsupported_ticker() -> None:
    response = CLIENT.get("/api/v1/news?ticker=INVALID")
    assert response.status_code == 400
    assert "AAPL" in response.json()["detail"]


def test_news_route_returns_contract(monkeypatch) -> None:
    from routes import simple_forecast

    monkeypatch.setattr(
        simple_forecast,
        "fetch_recent_news",
        lambda *args, **kwargs: {
            "status": "available",
            "items": [
                {
                    "headline": "Mock Headline",
                    "source": "Mock Source",
                    "published_at": "2026-09-04T12:00:00Z",
                    "url": "https://example.com/news",
                    "sentiment": 0.25,
                    "sentiment_label": "positive",
                }
            ],
            "role": "context_only",
            "used_by_model": False,
        },
    )
    response = CLIENT.get("/api/v1/news?ticker=NVDA")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ticker"] == "NVDA"
    assert payload["status"] == "available"
    assert payload["role"] == "context_only"
    assert len(payload["items"]) == 1
    assert payload["items"][0]["headline"] == "Mock Headline"


def test_news_service_returns_rich_headline_objects() -> None:
    from services.market_news import clear_news_cache, fetch_recent_news

    clear_news_cache()
    res = fetch_recent_news("TSLA")
    assert res["status"] == "available"
    assert res["provider"] in ("alpaca", "yahoo", "sec_edgar", "cached_archive")
    assert len(res["items"]) >= 1

    item = res["items"][0]
    assert "title" in item and item["title"]
    assert "headline" in item and item["headline"]
    assert "source" in item and item["source"]
    assert "published_at" in item and item["published_at"]
    assert "summary" in item
    assert "url" in item
    assert "sentiment" in item and isinstance(item["sentiment"], float)
    assert "sentiment_label" in item and item["sentiment_label"] in (
        "positive",
        "negative",
        "neutral",
    )
    assert "sentiment_badge" in item and item["sentiment_badge"] in (
        "Bullish",
        "Bearish",
        "Neutral",
    )
    assert "sentiment_pos" in item and isinstance(item["sentiment_pos"], float)
    assert "sentiment_neg" in item and isinstance(item["sentiment_neg"], float)
    assert "sentiment_neu" in item and isinstance(item["sentiment_neu"], float)


def test_news_service_fallbacks(monkeypatch) -> None:
    from services.market_news import clear_news_cache, fetch_recent_news

    clear_news_cache()

    # When Yahoo fails, falls back to SEC EDGAR
    monkeypatch.setattr(
        "services.market_news._fetch_from_yahoo",
        lambda _s: (_ for _ in ()).throw(Exception("Yahoo Down")),
    )
    res_edgar = fetch_recent_news("AAPL")
    assert res_edgar["status"] == "available"
    assert res_edgar["provider"] == "sec_edgar"
    assert len(res_edgar["items"]) >= 1

    clear_news_cache()
    # When Yahoo and SEC EDGAR fail, falls back to cached archive
    monkeypatch.setattr(
        "services.market_news._fetch_from_sec_edgar",
        lambda _s: (_ for _ in ()).throw(Exception("EDGAR Down")),
    )
    res_cache = fetch_recent_news("MSFT")
    assert res_cache["status"] == "available"
    assert res_cache["provider"] == "cached_archive"
    assert len(res_cache["items"]) >= 1


def test_train_and_forecast_calibrated_cones_and_learned_models() -> None:
    from services.simple_forecast import clear_forecast_cache, train_and_forecast

    clear_forecast_cache()
    frame = _frame(rows=700)
    res = train_and_forecast("AAPL", frame, model_name="ridge")

    assert res["ticker"] == "AAPL"
    assert res["forecast_days"] == 7
    assert len(res["predicted_prices"]) == 7
    assert len(res["lower_prices"]) == 7
    assert len(res["upper_prices"]) == 7
    # Cones must expand or stay strictly bounded
    for day in range(7):
        assert res["lower_prices"][day] <= res["predicted_prices"][day] <= res["upper_prices"][day]
        assert res["lower_prices"][day] < res["upper_prices"][day]

    assert res["model"]["kind"] == "learned_historical_model"
    assert res["model"]["name"] == "ridge"
    assert "mae_percent" in res["backtest"]
    assert "rmse_percent" in res["backtest"]
    assert "direction_accuracy" in res["backtest"]


def test_cold_process_artifact_inference_is_exact(monkeypatch, tmp_path):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    from services import simple_forecast as sf

    directory = tmp_path / "models"
    monkeypatch.setattr(sf.settings, "forecast_model_cache_dir", str(directory))
    sf.clear_forecast_cache()
    frame = _frame(700)
    frame.to_parquet(tmp_path / "frame.parquet")
    first = sf.train_and_forecast("AAPL", frame, "ridge")
    assert list(directory.glob("*.artifact.json"))
    code = """
import json, sys, pandas as pd
from config import settings
from services import simple_forecast as sf
settings.forecast_model_cache_dir = sys.argv[1]
def forbidden(*args, **kwargs):
    raise AssertionError('Cold artifact load must not rebuild/train')
sf.build_dataset = forbidden
result = sf.train_and_forecast('AAPL', pd.read_parquet(sys.argv[2]), 'ridge')
print(json.dumps(result, sort_keys=True))
"""
    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(root / "backend"), str(root)])}
    run = subprocess.run(
        [sys.executable, "-c", code, str(directory), str(tmp_path / "frame.parquet")],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=True,
    )
    assert json.dumps(first, sort_keys=True) == run.stdout.strip()


def test_financial_sentiment_lexicon_scoring() -> None:
    from services.market_news import _format_news_item

    # Bullish market headlines
    bullish_item = _format_news_item(
        title="Tesla surges 12% on record deliveries and profit beat",
        summary="Automaker beats quarterly Wall Street expectations.",
        source="Reuters",
    )
    assert bullish_item is not None
    assert bullish_item["sentiment"] > 0.15
    assert bullish_item["sentiment_label"] == "positive"
    assert bullish_item["sentiment_badge"] == "Bullish"

    # Bearish market headlines (specifically testing the probe/falls case from live feed)
    bearish_item = _format_news_item(
        title="Tesla Stock Falls 6% as NHTSA Probes Its Cybercabs",
        summary="Federal regulators launch inquiry following autonomous pilot launch.",
        source="Bloomberg",
    )
    assert bearish_item is not None
    assert bearish_item["sentiment"] < -0.15
    assert bearish_item["sentiment_label"] == "negative"
    assert bearish_item["sentiment_badge"] == "Bearish"


def test_alpaca_credentials_whitespace_sanitization(monkeypatch) -> None:
    from services.market_news import clear_news_cache, fetch_recent_news

    clear_news_cache()
    # Mock alpaca to ensure it is NOT called when keys are whitespace only
    called_alpaca = []

    def fake_alpaca(*args, **kwargs):
        called_alpaca.append(True)
        return []

    monkeypatch.setattr("services.market_news._fetch_from_alpaca", fake_alpaca)
    # When keys are whitespace only, it should skip Alpaca without calling it
    res = fetch_recent_news("AAPL", key_id="   ", secret_key="  \t ")
    assert len(called_alpaca) == 0
    assert res["provider"] != "alpaca"


def test_train_and_forecast_gpu_lstm_selected_when_available() -> None:
    from services.simple_forecast import (
        _load_gpu_lstm_model,
        clear_forecast_cache,
        train_and_forecast,
    )

    if _load_gpu_lstm_model() is None:
        return

    clear_forecast_cache()
    frame = _frame(rows=900)
    res = train_and_forecast("TSLA", frame, model_name="auto")
    assert res["ticker"] == "TSLA"
    assert len(res["predicted_prices"]) == 7
    # Candidate scores must contain gpu_lstm
    assert "gpu_lstm" in res["model"]["candidate_validation_mae"]
    # Backtest metrics must be computed and present
    assert res["backtest"]["mae_percent"] > 0
    assert res["backtest"]["direction_accuracy"] > 0
    assert res["backtest"]["test_samples"] > 0


def test_multi_exchange_lse_forecast_contract() -> None:
    from services.simple_forecast import clear_forecast_cache, get_ticker_meta, train_and_forecast

    meta = get_ticker_meta("SHEL.L")
    assert meta["exchange_mic"] == "XLON"
    assert meta["currency"] == "GBp"
    assert meta["currency_symbol"] == "p"

    clear_forecast_cache()
    frame = _frame(rows=700)
    res = train_and_forecast("SHEL.L", frame)
    assert res["ticker"] == "SHEL.L"
    assert res["exchange_mic"] == "XLON"
    assert res["currency"] == "GBp"
    assert res["currency_symbol"] == "p"
    assert res["provenance"]["calendar"] == "LSE"
    assert len(res["predicted_prices"]) == 7


def test_multi_exchange_news_cutoff_and_timing(monkeypatch) -> None:
    from datetime import timedelta

    from services import market_news
    from services.market_news import clear_news_cache, fetch_recent_news, get_exchange_market_close

    close_lse = get_exchange_market_close("SHEL.L")
    assert close_lse is not None

    close_nyse = get_exchange_market_close("AAPL")
    assert close_nyse is not None

    def headlines(symbol):
        assert symbol == "SHEL.L"
        return [
            {
                "id": "before",
                "headline": "Quarterly results",
                "published_at": (close_lse - timedelta(minutes=1)).isoformat(),
            },
            {
                "id": "after",
                "headline": "Company update",
                "published_at": (close_lse + timedelta(minutes=1)).isoformat(),
            },
        ]

    def unexpected_provider(*args, **kwargs):
        raise AssertionError("Offline test must not contact another provider")

    monkeypatch.setattr(market_news, "_fetch_from_yahoo", headlines)
    for name in ("_fetch_from_alpaca", "_fetch_from_sec_edgar", "_fetch_from_cache"):
        monkeypatch.setattr(market_news, name, unexpected_provider)

    clear_news_cache()
    news = fetch_recent_news("SHEL.L")
    assert news["status"] == "available"
    assert "exchange_close_utc" in news
    assert {item["id"]: item["after_market_close"] for item in news["items"]} == {
        "before": False,
        "after": True,
    }
    for item in news["items"]:
        assert "after_market_close" in item
        assert item["session_timing"] in ("regular_hours", "after_hours")
    clear_news_cache()
