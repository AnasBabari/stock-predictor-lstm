"""Unit tests for NewsCoverageAuditorV11."""

from research.volatility_forecasting.news_aggregator_v2 import (
    EnrichedNewsArticle,
)
from research.volatility_forecasting.news_coverage_auditor_v11 import (
    NewsCoverageAuditorV11,
)


def test_news_coverage_audit_metrics():
    origins = [
        ("AMGN", "2024-06-10"),
        ("AMGN", "2024-06-11"),
        ("AAPL", "2024-06-10"),
        ("AAPL", "2024-06-11"),
    ]

    articles = [
        EnrichedNewsArticle(
            article_id="ART_1",
            ticker="AMGN",
            headline="Amgen clinical trial report",
            source="Reuters",
            published_at="2024-06-10T14:00:00Z",
            first_seen_at="2024-06-10T14:00:05Z",
            delivery_time="2024-06-10T14:00:10Z",
            ticker_relevance=1.0,
            event_type="clinical_trial",
            sentiment_score=0.5,
            sentiment_magnitude=0.7,
            severity_score=0.4,
            uncertainty_score=0.2,
            embedding_vector=[1.0, 0.0, 0.0, 0.0],
        ),
        EnrichedNewsArticle(
            article_id="ART_2",
            ticker="AMGN",
            headline="Amgen oncology pipeline updates",
            source="Bloomberg",
            published_at="2024-06-10T15:00:00Z",
            first_seen_at="2024-06-10T15:00:05Z",
            delivery_time="2024-06-10T15:00:10Z",
            ticker_relevance=1.0,
            event_type="clinical_trial",
            sentiment_score=0.6,
            sentiment_magnitude=0.8,
            severity_score=0.3,
            uncertainty_score=0.1,
            embedding_vector=[0.8, 0.2, 0.0, 0.0],
        ),
    ]

    report = NewsCoverageAuditorV11.audit_coverage(
        stock_dates=origins,
        news_articles=articles,
        min_coverage_threshold=0.20,
    )

    assert report.total_stock_origins == 4
    assert report.origins_with_ge_1_article == 2  # AMGN rows have news, AAPL rows have 0
    assert report.coverage_by_ticker["AMGN"] == 100.0
    assert report.coverage_by_ticker["AAPL"] == 0.0
