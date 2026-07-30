from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pandas as pd

from news_features import (
    build_daily_sentiment_features,
    get_live_financial_sentiment,
    normalise_news_articles,
)


def test_normalises_legacy_and_nested_yfinance_news_records():
    articles = normalise_news_articles(
        [
            {"title": "Legacy headline", "providerPublishTime": 1_700_000_000},
            {
                "content": {
                    "title": "Nested headline",
                    "summary": "Details",
                    "pubDate": "2024-01-02T12:00:00Z",
                }
            },
        ]
    )

    assert [article["title"] for article in articles] == ["Legacy headline", "Nested headline"]
    assert all(article["published_at"] is not None for article in articles)


def test_historical_sentiment_excludes_future_articles_and_has_confidence():
    sessions = pd.DatetimeIndex(["2024-01-02", "2024-01-03"], tz="UTC")
    articles = [
        {
            "title": "Company beat expectations",
            "summary": "",
            "published_at": datetime(2024, 1, 2, 12, tzinfo=UTC),
            "publisher": None,
            "link": None,
        },
        {
            "title": "Future news must not leak",
            "summary": "",
            "published_at": datetime(2024, 1, 4, 12, tzinfo=UTC),
            "publisher": None,
            "link": None,
        },
    ]

    features = build_daily_sentiment_features(articles, sessions, half_life_days=7)

    assert features.loc[sessions[0], "News_Article_Count"] == 0
    assert 0 < features.loc[sessions[1], "News_Article_Count"] < 1
    assert features.loc[sessions[1], "News_Sentiment"] > 0
    assert 0 < features.loc[sessions[1], "News_Sentiment_Confidence"] <= 1


@patch("news_features.yf.Ticker")
def test_live_sentiment_includes_coverage_metadata(mock_ticker):
    instance = MagicMock()
    instance.news = [{"content": {"title": "Earnings beat expectations"}}]
    mock_ticker.return_value = instance

    sentiment = get_live_financial_sentiment("AAPL")

    assert sentiment["status"] == "live"
    assert sentiment["article_count"] == 1
    assert sentiment["timestamped_article_count"] == 0
