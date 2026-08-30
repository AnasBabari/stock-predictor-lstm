"""Unit tests for MultiDimensionalNewsAggregator."""

import numpy as np

from research.volatility_forecasting.news_aggregator_v2 import (
    AggregatedNewsFeatures,
    EnrichedNewsArticle,
    MultiDimensionalNewsAggregator,
)


def test_news_deduplication_and_aggregation():
    articles = [
        EnrichedNewsArticle(
            article_id="ART_001",
            headline="Amgen drug demonstrates significant efficacy in Phase 3 trial",
            source="Reuters",
            published_at="2026-08-27T10:00:00Z",
            first_seen_at="2026-08-27T10:00:05Z",
            delivery_time="2026-08-27T10:00:10Z",
            ticker_relevance=1.0,
            event_type="clinical_trial",
            sentiment_score=0.75,
            sentiment_magnitude=0.85,
            severity_score=0.60,
            uncertainty_score=0.30,
        ),
        EnrichedNewsArticle(
            article_id="ART_002",
            headline="Amgen drug demonstrates significant efficacy in Phase 3 trial",  # Syndicated duplicate
            source="MarketWatch",
            published_at="2026-08-27T10:05:00Z",
            first_seen_at="2026-08-27T10:05:05Z",
            delivery_time="2026-08-27T10:05:10Z",
            ticker_relevance=1.0,
            event_type="clinical_trial",
            sentiment_score=0.75,
            sentiment_magnitude=0.85,
            severity_score=0.60,
            uncertainty_score=0.30,
        ),
    ]

    deduped = MultiDimensionalNewsAggregator.deduplicate_articles(articles)
    assert len(deduped) == 1  # Successfully deduplicated duplicate story

    feats = MultiDimensionalNewsAggregator.aggregate_causal_window(
        articles=articles, cutoff_iso="2026-08-28T16:00:00Z"
    )
    arr = feats.to_array()
    assert len(arr) == len(AggregatedNewsFeatures.feature_names())
    assert np.isfinite(arr).all()
    assert feats.clinical_trial_events_5d == 1.0
    assert feats.mean_sentiment_5d == 0.75
