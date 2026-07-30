from unittest.mock import MagicMock, patch

import api
from config import MAX_FORECAST_DAYS, WINDOW_SIZE
from tests.test_api import _feature_snapshot


def test_direction_pipeline_preserves_rich_flat_sentiment_context():
    snapshot = _feature_snapshot()
    sentiment = {
        "score": 0.2,
        "status": "live",
        "provider": "yfinance",
        "method": "vader_financial",
        "article_count": 3,
        "timestamped_article_count": 2,
    }
    with (
        patch("api.fetch_data", return_value=(snapshot, snapshot.Close.values, snapshot.index, {})),
        patch("api.load_fresh_artifact", return_value=(MagicMock(), MagicMock())),
        patch(
            "api.predict_direction",
            return_value=(
                ["Up"],
                [0.6],
                [1 / WINDOW_SIZE] * WINDOW_SIZE,
            ),
        ),
        patch("api.future_trading_dates", return_value=(["2026-08-01"], "NYSE")),
        patch("api.get_financial_sentiment", return_value=sentiment),
        patch("api.load_metrics", return_value={"metric_source": "unavailable"}),
    ):
        result = api._direction_prediction_pipeline("AAPL", 1)

    assert result["sentiment"] == sentiment
    assert result["metadata"]["calendar"] == "NYSE"
    assert MAX_FORECAST_DAYS == 30
