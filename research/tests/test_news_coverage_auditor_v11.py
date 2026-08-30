"""Unit tests for separated company vs macro NewsCoverageAuditorV11."""

from research.volatility_forecasting.news_aggregator_v2 import (
    EnrichedNewsArticle,
)
from research.volatility_forecasting.news_coverage_auditor_v11 import (
    NewsCoverageAuditorV11,
    NewsCoverageGateV11,
)


def test_separated_company_vs_macro_news_audit():
    origins = [
        ("AMGN", "2024-06-10"),
        ("AMGN", "2024-06-11"),
        ("AAPL", "2024-06-10"),
        ("AAPL", "2024-06-11"),
    ]

    articles = [
        # Company-specific article for AMGN
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
        # Company-specific article from second source
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
        # Macro article — must NOT be counted as company-specific for AAPL
        EnrichedNewsArticle(
            article_id="ART_3",
            ticker="MARKET",
            headline="Fed Chair remarks on interest rate policy",
            source="WSJ",
            published_at="2024-06-10T15:30:00Z",
            first_seen_at="2024-06-10T15:30:05Z",
            delivery_time="2024-06-10T15:30:10Z",
            ticker_relevance=0.5,
            event_type="macro",
            sentiment_score=0.0,
            sentiment_magnitude=0.2,
            severity_score=0.2,
            uncertainty_score=0.3,
            embedding_vector=[0.0, 1.0, 0.0, 0.0],
        ),
    ]

    report = NewsCoverageAuditorV11.audit_coverage(
        stock_dates=origins,
        news_articles=articles,
        gate_config=NewsCoverageGateV11(min_total_origins=4, minimum_security_count=2),
    )

    assert report.total_stock_origins == 4
    # AMGN has 2 origins with company news, AAPL has 0 company news (macro does NOT count as company news!)
    assert report.company_origins_ge_1_article == 2
    assert report.company_coverage_percentage == 50.0
    assert report.macro_origins_ge_1_article == 4  # Macro informs all 4
    assert report.coverage_by_security["AMGN"] == 100.0
    assert report.coverage_by_security["AAPL"] == 0.0
