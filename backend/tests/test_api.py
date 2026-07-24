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


def _make_mock_feature_df(n):
    import numpy as np
    import pandas as pd

    from config import FEATURES

    dates = _make_dates(n)
    df = pd.DataFrame(index=dates)
    for f in FEATURES:
        df[f] = np.random.rand(n)
    return df


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"
    assert "version" in res.json()


def test_ready():
    res = client.get("/ready")
    assert res.status_code == 200
    assert res.json()["status"] == "ready"
    assert "dependencies" in res.json()


def test_models():
    res = client.get("/models")
    assert res.status_code == 200
    assert "manifest" in res.json()


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
        patch("api.fetch_data") as mock_fetch,
        patch("api.load_or_train") as mock_model,
        patch("api.evaluate_model") as mock_eval,
        patch("api.predict_future") as mock_pred,
        patch("api.run_in_threadpool") as mock_thread,
    ):
        mock_scaler = MagicMock()

        async def mock_run(*args, **kwargs):
            return mock_model(), mock_scaler

        mock_thread.side_effect = mock_run

        mock_df = _make_mock_feature_df(100)
        mock_pipe.return_value = (
            (MagicMock(), MagicMock(), MagicMock(), MagicMock(), mock_scaler, [], []),
            _make_prices(100),
            _make_dates(100),
            {"feature_count": 21},
        )
        mock_fetch.return_value = (
            mock_df,
            _make_prices(100),
            _make_dates(100),
            {"feature_count": 21},
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
        patch("api.fetch_data") as mock_fetch,
        patch("api.load_or_train") as mock_model,
        patch("api.evaluate_model") as mock_eval,
        patch("api.predict_future") as mock_pred,
        patch("api.run_in_threadpool") as mock_thread,
    ):
        mock_scaler = MagicMock()

        async def mock_run(*args, **kwargs):
            return mock_model(), mock_scaler

        mock_thread.side_effect = mock_run

        mock_df = _make_mock_feature_df(100)
        mock_pipe.return_value = (
            (MagicMock(), MagicMock(), MagicMock(), MagicMock(), mock_scaler, [], []),
            _make_prices(100),
            _make_dates(100),
            {"feature_count": 21},
        )
        mock_fetch.return_value = (
            mock_df,
            _make_prices(100),
            _make_dates(100),
            {"feature_count": 21},
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
        assert mock_ticker.call_count == 1  # second call hits cache


def test_predict_direction_schema():
    with (
        patch("api.fetch_data") as mock_fetch,
        patch("api.prepare_return_data") as mock_prep,
        patch("api.load_or_train") as mock_model,
        patch("api.predict_direction") as mock_pred,
        patch("api.load_metrics") as mock_metrics,
        patch("api.run_in_threadpool") as mock_thread,
        patch("api.get_financial_sentiment") as mock_sentiment,
    ):
        mock_scaler = MagicMock()

        async def mock_run(*args, **kwargs):
            return mock_model(), mock_scaler

        mock_thread.side_effect = mock_run

        mock_df = _make_mock_feature_df(100)
        mock_fetch.return_value = (
            mock_df,
            _make_prices(100),
            _make_dates(100),
            {"feature_count": 21},
        )
        mock_prep.return_value = (
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            mock_scaler,
            [],
            [],
        )
        mock_pred.return_value = (["Up"] * 7, [0.6] * 7, [0.1] * 60)
        mock_metrics.return_value = {
            "precision": 0.8,
            "recall": 0.7,
            "f1": 0.75,
            "naive_baseline": 0.5,
        }
        mock_sentiment.return_value = {
            "sentiment": {
                "score": 0.5,
                "status": "live",
                "provider": "yfinance",
                "method": "vader_financial",
            }
        }

        res = client.get("/api/v1/predict/direction?ticker=AAPL&days=7")
        body = res.json()
        assert res.status_code == 200
        assert set(body.keys()) >= {
            "ticker",
            "forecast_days",
            "future_dates",
            "directions",
            "probabilities",
            "attention_weights",
            "metrics",
            "sentiment",
        }
        assert body["forecast_days"] == 7
        assert len(body["directions"]) == 7
        assert len(body["probabilities"]) == 7
        assert body["sentiment"]["score"] == 0.5
        assert body["sentiment"]["status"] == "live"


def test_predict_direction_attention_alignment():
    with (
        patch("api.fetch_data") as mock_fetch,
        patch("api.prepare_return_data") as mock_prep,
        patch("api.load_or_train") as mock_model,
        patch("api.predict_direction") as mock_pred,
        patch("api.load_metrics") as mock_metrics,
        patch("api.run_in_threadpool") as mock_thread,
        patch("api.get_financial_sentiment") as mock_sentiment,
    ):
        mock_scaler = MagicMock()

        async def mock_run(*args, **kwargs):
            return mock_model(), mock_scaler

        mock_thread.side_effect = mock_run

        mock_df = _make_mock_feature_df(100)
        mock_fetch.return_value = (
            mock_df,
            _make_prices(100),
            _make_dates(100),
            {"feature_count": 21},
        )
        mock_prep.return_value = (
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            mock_scaler,
            [],
            [],
        )
        mock_pred.return_value = (["Up"] * 7, [0.6] * 7, [0.05] * 60)
        mock_metrics.return_value = {
            "precision": 0.8,
            "recall": 0.7,
            "f1": 0.75,
            "naive_baseline": 0.5,
        }
        mock_sentiment.return_value = {
            "sentiment": {
                "score": 0.0,
                "status": "fallback",
                "provider": "yfinance",
                "method": "vader_financial",
            }
        }

        res = client.get("/api/v1/predict/direction?ticker=AAPL&days=7")
        body = res.json()
        attn = body["attention_weights"]

        # 1. Correct length matching WINDOW_SIZE (60)
        assert len(attn) == 60

        # 2. Sequential indexing & numeric weight types
        for idx, item in enumerate(attn):
            assert item["index"] == idx
            assert isinstance(item["weight"], int | float)
            assert isinstance(item["date"], str)

        # 3. Dates ordered oldest -> newest with no duplicate dates
        dates = [item["date"] for item in attn]
        assert dates == sorted(dates)
        assert len(set(dates)) == len(dates)


def test_diagnostics_404_when_not_trained():
    """Phase 3: diagnostics endpoint returns 404 if no walk-forward data exists."""
    with (
        patch("api.load_cross_validation") as mock_cv,
        patch("api.load_validation_results") as mock_vr,
        patch("api.load_metadata") as mock_meta,
    ):
        mock_cv.return_value = {}
        mock_vr.return_value = []
        mock_meta.return_value = {}

        res = client.get("/api/v1/diagnostics/AAPL")
        assert res.status_code == 404
        assert "Train the model first" in res.json()["detail"]


def test_diagnostics_returns_cv_and_folds():
    """Phase 3: diagnostics endpoint returns cross_validation and fold_results."""
    with (
        patch("api.load_cross_validation") as mock_cv,
        patch("api.load_validation_results") as mock_vr,
        patch("api.load_metadata") as mock_meta,
    ):
        mock_cv.return_value = {
            "folds": 5,
            "folds_completed": 5,
            "average_rmse": 2.34,
            "std_rmse": 0.12,
        }
        mock_vr.return_value = [
            {
                "fold": 1,
                "train_start": "2020-01-01",
                "validation_start": "2021-01-01",
                "actuals": [100.0, 101.0],
                "predictions": [99.5, 101.5],
                "residuals": [
                    {"date": "2021-01-05", "actual": 100.0, "residual": 0.5, "absolute_error": 0.5}
                ],
            }
        ]
        mock_meta.return_value = {
            "schema_version": 2,
            "validation_method": "expanding",
            "validation_folds": 5,
        }

        res = client.get("/api/v1/diagnostics/AAPL?model_type=bilstm_attention_direction")
        body = res.json()
        assert res.status_code == 200
        assert "cross_validation" in body
        assert "fold_results" in body
        assert "model_metadata" in body
        assert body["cross_validation"]["folds_completed"] == 5
        assert len(body["fold_results"]) == 1
        assert body["model_metadata"]["validation_method"] == "expanding"
