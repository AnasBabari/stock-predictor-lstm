"""Unit tests for hardened MultiDimensionalNewsAggregator."""

from research.volatility_forecasting.news_aggregator_v2 import (
    EnrichedNewsArticle,
    MultiDimensionalNewsAggregator,
)


def test_news_ticker_filtering_and_real_windows():
    articles = [
        EnrichedNewsArticle(
            article_id="ART_001",
            ticker="AMGN",
            headline="Amgen Tezspire Phase 3 success",
            source="Reuters",
            published_at="2026-08-28T15:30:00Z",
            first_seen_at="2026-08-28T15:30:05Z",
            delivery_time="2026-08-28T15:30:10Z",
            ticker_relevance=1.0,
            event_type="clinical_trial",
            sentiment_score=0.8,
            sentiment_magnitude=0.9,
            severity_score=0.4,
            uncertainty_score=0.2,
            embedding_vector=[1.0, 0.0, 0.0, 0.0],
        ),
        EnrichedNewsArticle(
            article_id="ART_002",
            ticker="NVDA",  # Different ticker — must NOT leak into AMGN row
            headline="Nvidia launches new GPU architecture",
            source="Bloomberg",
            published_at="2026-08-28T15:00:00Z",
            first_seen_at="2026-08-28T15:00:05Z",
            delivery_time="2026-08-28T15:00:10Z",
            ticker_relevance=1.0,
            event_type="earnings",
            sentiment_score=0.9,
            sentiment_magnitude=0.9,
            severity_score=0.5,
            uncertainty_score=0.1,
            embedding_vector=[0.0, 1.0, 0.0, 0.0],
        ),
        EnrichedNewsArticle(
            article_id="ART_003",
            ticker="MARKET",  # Macro news — permitted to inform AMGN
            headline="Federal Reserve signals interest rate decision",
            source="WSJ",
            published_at="2026-08-28T14:30:00Z",
            first_seen_at="2026-08-28T14:30:05Z",
            delivery_time="2026-08-28T14:30:10Z",
            ticker_relevance=0.8,
            event_type="macro",
            sentiment_score=0.1,
            sentiment_magnitude=0.3,
            severity_score=0.3,
            uncertainty_score=0.4,
            embedding_vector=[0.0, 0.0, 1.0, 0.0],
        ),
    ]

    # AMGN aggregation strictly receives AMGN + Macro, ignoring NVDA
    amgn_feats = MultiDimensionalNewsAggregator.aggregate_causal_window(
        articles=articles,
        target_ticker="AMGN",
        cutoff_iso="2026-08-28T16:00:00Z",
    )

    assert amgn_feats.articles_1h == 1.0  # ART_001 in 1h window
    assert amgn_feats.articles_4h == 2.0  # ART_001 + ART_003 in 4h window
    assert amgn_feats.clinical_trial_events_5d == 1.0
    assert amgn_feats.earnings_guidance_events_5d == 0.0  # NVDA earnings excluded!
    assert amgn_feats.unique_sources_5d == 2.0  # Reuters + WSJ
