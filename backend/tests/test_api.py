# backend/tests/test_api.py
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def _make_prices(n):
    return [[float(i)] for i in range(n)]


def _make_dates(n):
    import pandas as pd

    return pd.date_range("2023-01-01", periods=n, freq="B")


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_validate_ticker_rejects_path_traversal():
    res = client.get("/api/v1/predict?ticker=../etc/passwd")
    assert res.status_code == 400
    assert "Invalid ticker" in res.json()["detail"]


def test_validate_ticker_rejects_empty():
    res = client.get("/api/v1/predict?ticker=")
    assert res.status_code == 400


def test_validate_ticker_rejects_too_long():
    res = client.get("/api/v1/predict?ticker=ABCDEFGHIJKLM")  # 13 chars
    assert res.status_code == 400


def test_validate_ticker_accepts_valid():
    with (
        patch("api.get_pipeline") as mock_pipe,
        patch("api.load_or_train") as mock_model,
        patch("api.evaluate_model") as mock_eval,
        patch("api.predict_future") as mock_pred,
        patch("api.run_in_threadpool") as mock_thread,
    ):

        async def mock_run(*args, **kwargs):
            return mock_model()

        mock_thread.side_effect = mock_run

        mock_pipe.return_value = (
            (MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()),
            _make_prices(100),
            _make_dates(100),
        )
        mock_eval.return_value = {
            "rmse": 1.0,
            "mae": 0.5,
            "mape": 1.2,
            "r2": 0.95,
            "directional_accuracy": 0.6,
        }
        mock_pred.return_value = [150.0] * 7
        res = client.get("/api/v1/predict?ticker=AAPL&days=7")
        assert res.status_code == 200


def test_predict_response_schema():
    with (
        patch("api.get_pipeline") as mock_pipe,
        patch("api.load_or_train") as mock_model,
        patch("api.evaluate_model") as mock_eval,
        patch("api.predict_future") as mock_pred,
        patch("api.run_in_threadpool") as mock_thread,
    ):

        async def mock_run(*args, **kwargs):
            return mock_model()

        mock_thread.side_effect = mock_run

        mock_pipe.return_value = (
            (MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()),
            _make_prices(100),
            _make_dates(100),
        )
        mock_eval.return_value = {
            "rmse": 1.0,
            "mae": 0.5,
            "mape": 1.2,
            "r2": 0.95,
            "directional_accuracy": 0.6,
        }
        mock_pred.return_value = [150.0] * 7
        res = client.get("/api/v1/predict?ticker=AAPL&days=7")
        body = res.json()
        assert set(body.keys()) >= {
            "ticker",
            "historical_dates",
            "historical_prices",
            "future_dates",
            "predicted_prices",
            "forecast_days",
            "metrics",
        }
        assert body["forecast_days"] == 7
        assert len(body["predicted_prices"]) == 7
        assert len(body["future_dates"]) == 7


def test_predict_days_clamped():
    res = client.get("/api/v1/predict?ticker=AAPL&days=99")
    assert res.status_code == 422  # FastAPI Query validation


def test_search_returns_list():
    with patch("api.yf.Search") as mock_search:
        mock_search.return_value.quotes = [
            {"symbol": "AAPL", "longname": "Apple Inc.", "quoteType": "EQUITY"}
        ]
        res = client.get("/api/v1/search?query=Apple")
        assert res.status_code == 200
        assert "results" in res.json()
        assert res.json()["results"][0]["ticker"] == "AAPL"


def test_search_error_returns_500_not_stacktrace():
    with patch("api.yf.Search", side_effect=RuntimeError("boom")):
        res = client.get("/api/v1/search?query=Apple")
        assert res.status_code == 500
        assert "boom" not in res.json()["detail"]  # sanitised


def test_info_caches_response():
    with patch("api.yf.Ticker") as mock_ticker:
        mock_ticker.return_value.info = {"longName": "Apple Inc."}
        client.get("/api/v1/info?ticker=AAPL")
        client.get("/api/v1/info?ticker=AAPL")
        assert mock_ticker.call_count == 1  # second call hits cache
