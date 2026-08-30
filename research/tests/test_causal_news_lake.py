"""Unit tests for point-in-time causal news lake and negative controls."""

from research.volatility_forecasting.causal_news_lake import (
    CausalNewsFeatureExtractor,
    NewsEventRecord,
)


def test_news_event_record_available_at_is_causal_max():
    rec = NewsEventRecord(
        headline_id="NWS_001",
        security_id="SEC_BP_001",
        published_at="2024-01-15T08:00:00Z",
        first_seen_at="2024-01-15T08:00:05Z",
        delivery_time="2024-01-15T08:00:10Z",
        sentiment_score=0.45,
        category="energy",
    )
    assert rec.available_at == "2024-01-15T08:00:10Z"


def test_future_news_is_strictly_excluded_by_cutoff():
    records = [
        NewsEventRecord(
            headline_id="NWS_001",
            security_id="SEC_BP_001",
            published_at="2024-01-15T08:00:00Z",
            first_seen_at="2024-01-15T08:00:00Z",
            delivery_time="2024-01-15T08:00:00Z",
            sentiment_score=-0.60,
            category="earnings",
        ),
        NewsEventRecord(
            headline_id="NWS_002",
            security_id="SEC_BP_001",
            published_at="2024-01-16T14:00:00Z",
            first_seen_at="2024-01-16T14:00:00Z",
            delivery_time="2024-01-16T14:00:00Z",
            sentiment_score=0.80,
            category="macro",
        ),
    ]

    # Query at cutoff of Jan 15 23:59:59 (Jan 16 news must NOT be visible)
    feats = CausalNewsFeatureExtractor.extract_point_in_time_features(
        records, "SEC_BP_001", cutoff_timestamp="2024-01-15T23:59:59Z"
    )
    assert feats["news_count_5d"] == 1.0
    assert feats["sentiment_mean_5d"] == -0.60


def test_negative_control_random_embeddings():
    records = []
    feats = CausalNewsFeatureExtractor.extract_point_in_time_features(
        records,
        "SEC_BP_001",
        cutoff_timestamp="2024-01-15T23:59:59Z",
        ablation_control="random_embeddings",
    )
    assert "news_count_5d" in feats
    assert "sentiment_mean_5d" in feats
