"""Unit tests for the live news-aggregation provider (no network)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from services.news_aggregator import (
    LIVE_SUPPORTED_FEATURE_NAMES,
    NewsAggregationProvider,
    NewsProviderUnavailable,
    aggregate_live_features,
    build_live_events,
)

CUTOFF = datetime(2026, 8, 21, 20, 0, tzinfo=UTC)


def _article(
    *,
    title: str = "MSFT beats expectations",
    published: datetime | None = None,
    received: datetime | None = None,
    sentiment: float = 0.5,
    publisher: str = "Reuters",
) -> dict[str, object]:
    return {
        "title": title,
        "summary": "",
        "publisher": publisher,
        "ticker": "MSFT",
        "published_at": published,
        "received_at": received,
        "sentiment_score": sentiment,
    }


def test_events_respect_the_strict_cutoff_and_availability_delay() -> None:
    articles = [
        _article(published=CUTOFF - timedelta(hours=3), received=CUTOFF - timedelta(hours=3)),
        _article(
            title="too recent",
            published=CUTOFF - timedelta(seconds=60),
            received=CUTOFF - timedelta(seconds=60),
        ),
    ]
    events = build_live_events(
        articles,
        cutoff_at=CUTOFF,
        availability_delay=timedelta(minutes=15),
    )
    # The 3-hour-old article becomes eligible at published + 15 minutes: inside
    # the window. The article 60 seconds old is still inside its delay: excluded.
    assert len(events) == 1
    assert events[0].source == "Reuters"


def test_date_only_publication_becomes_eligible_next_utc_day() -> None:
    articles = [_article(published=datetime(2026, 8, 21, 0, 0, tzinfo=UTC))]
    events = build_live_events(
        articles,
        cutoff_at=CUTOFF,
        availability_delay=timedelta(0),
    )
    # Midnight timestamps carry no clock time and cannot enter a same-day cut.
    assert len(events) == 0

    next_day = build_live_events(
        articles,
        cutoff_at=datetime(2026, 8, 22, 0, 0, tzinfo=UTC) + timedelta(minutes=1),
        availability_delay=timedelta(0),
    )
    assert len(next_day) == 1


def test_missing_timestamps_and_sentiment_clamping() -> None:
    events = build_live_events(
        [_article(published=None, received=None)],
        cutoff_at=CUTOFF,
        availability_delay=timedelta(0),
    )
    assert events == []
    extreme = build_live_events(
        [_article(published=CUTOFF - timedelta(hours=1), sentiment=7.5)],
        cutoff_at=CUTOFF,
        availability_delay=timedelta(0),
    )
    assert extreme[0].sentiment == 1.0


def test_decay_intensity_matches_frozen_half_life_semantics() -> None:
    articles = [_article(published=CUTOFF - timedelta(hours=6))]
    events = build_live_events(articles, cutoff_at=CUTOFF, availability_delay=timedelta(0))
    vector = aggregate_live_features(
        events,
        cutoff_at=CUTOFF,
        feature_names=("News_Ticker_Intensity_1D",),
    )
    expected = float(np.exp(-np.log(2.0) * 6.0 / 12.0))
    assert vector[0] == pytest.approx(expected, rel=1e-6)
    # Beyond the 24h maximum age the event does not contribute.
    stale = aggregate_live_features(
        build_live_events(
            [_article(published=CUTOFF - timedelta(hours=30))],
            cutoff_at=CUTOFF,
            availability_delay=timedelta(0),
        ),
        cutoff_at=CUTOFF,
        feature_names=("News_Ticker_Intensity_1D",),
    )
    assert stale[0] == 0.0


def test_missing_indicator_source_diversity_and_novelty() -> None:
    no_events = aggregate_live_features(
        build_live_events([], cutoff_at=CUTOFF, availability_delay=timedelta(0)),
        cutoff_at=CUTOFF,
        feature_names=("News_Ticker_Missing_1D", "News_Ticker_Source_Diversity_3D"),
    )
    assert no_events[0] == 1.0
    assert no_events[1] == 0.0

    articles = [
        _article(published=CUTOFF - timedelta(hours=1), publisher="Reuters"),
        _article(
            title="MSFT guidance raise",
            published=CUTOFF - timedelta(hours=2),
            publisher="Bloomberg",
        ),
    ]
    events = build_live_events(articles, cutoff_at=CUTOFF, availability_delay=timedelta(0))
    vector = aggregate_live_features(
        events,
        cutoff_at=CUTOFF,
        feature_names=("News_Ticker_Missing_1D", "News_Ticker_Source_Diversity_3D"),
    )
    assert vector[0] == 0.0
    assert vector[1] == 2.0


def test_repeated_story_has_lower_novelty_than_new_information() -> None:
    articles = [
        _article(
            title="Microsoft announces AI chip for datacenters",
            published=CUTOFF - timedelta(hours=2),
        ),
        _article(
            title="Microsoft announces AI chip for datacenters",  # syndication
            publisher="Bloomberg",
            published=CUTOFF - timedelta(hours=1),
        ),
    ]
    events = build_live_events(articles, cutoff_at=CUTOFF, availability_delay=timedelta(0))
    assert events[0].novelty == 1.0
    assert events[1].novelty < 0.5
    fresh = build_live_events(
        [
            _article(
                title="Nvidia CEO unexpectedly resigns",
                published=CUTOFF - timedelta(hours=1),
            )
        ],
        cutoff_at=CUTOFF,
        availability_delay=timedelta(0),
    )
    assert fresh[0].novelty == 1.0


def test_negative_and_absolute_sentiment_split_extremes() -> None:
    articles = [
        _article(
            title="MSFT profit warning",
            published=CUTOFF - timedelta(hours=1),
            sentiment=-0.8,
        ),
        _article(
            title="MSFT beat estimates",
            published=CUTOFF - timedelta(hours=2),
            sentiment=0.9,
        ),
    ]
    events = build_live_events(articles, cutoff_at=CUTOFF, availability_delay=timedelta(0))
    vector = aggregate_live_features(
        events,
        cutoff_at=CUTOFF,
        feature_names=("News_Ticker_Negative_1D", "News_Ticker_Absolute_Sentiment_1D"),
    )
    age_one = float(np.exp(-np.log(2.0) * 1.0 / 12.0))
    age_two = float(np.exp(-np.log(2.0) * 2.0 / 12.0))
    assert vector[0] == pytest.approx(0.8 * age_one, rel=1e-6)
    assert vector[1] == pytest.approx(0.8 * age_one + 0.9 * age_two, rel=1e-6)


def test_feature_order_matches_the_request_and_rejects_unknown_names() -> None:
    events = build_live_events(
        [_article(published=CUTOFF - timedelta(hours=1))],
        cutoff_at=CUTOFF,
        availability_delay=timedelta(0),
    )
    names = ("News_Ticker_Severity_3D", "News_Ticker_Intensity_1H")
    vector = aggregate_live_features(events, cutoff_at=CUTOFF, feature_names=names)
    assert vector.shape == (2,)
    assert vector.dtype == np.float32
    with pytest.raises(NewsProviderUnavailable, match="unsupported live news feature"):
        aggregate_live_features(
            events,
            cutoff_at=CUTOFF,
            feature_names=("News_Market_Intensity_1D",),
        )
    assert "News_Market_Intensity_1D" not in LIVE_SUPPORTED_FEATURE_NAMES
    assert "News_Ticker_Missing_1D" in LIVE_SUPPORTED_FEATURE_NAMES


def test_provider_end_to_end_with_fake_source_is_deterministic() -> None:
    def source(ticker: str) -> list[dict[str, object]]:
        assert ticker == "MSFT"
        return [
            {
                "ticker": "MSFT",
                "content": {
                    "title": "MSFT beats estimates",
                    "pubDate": (CUTOFF - timedelta(hours=2)).isoformat(),
                },
            }
        ]

    names = ("News_Ticker_Intensity_1D", "News_Ticker_Negative_1D")
    provider = NewsAggregationProvider(event_source=source, availability_delay=timedelta(0))
    first = provider.features_for("MSFT", cutoff_at=CUTOFF, feature_names=names)
    second = provider.features_for("MSFT", cutoff_at=CUTOFF, feature_names=names)
    assert np.array_equal(first.values, second.values)
    assert first.eligible_article_count == 1
    assert first.feature_names == names
    assert first.cutoff_at.endswith("+00:00")


def test_provider_rejects_market_wide_features_and_transport_failures() -> None:
    provider = NewsAggregationProvider(
        event_source=lambda _ticker: [],
        availability_delay=timedelta(0),
    )
    with pytest.raises(NewsProviderUnavailable, match="market-wide point-in-time"):
        provider.features_for(
            "MSFT",
            cutoff_at=CUTOFF,
            feature_names=("News_Exposure_Intensity_1D",),
        )

    def broken(_ticker: str) -> list[dict[str, object]]:
        raise OSError("upstream down")

    failing = NewsAggregationProvider(event_source=broken)
    with pytest.raises(NewsProviderUnavailable, match="ingestion failed"):
        failing.features_for(
            "MSFT",
            cutoff_at=CUTOFF,
            feature_names=("News_Ticker_Intensity_1D",),
        )
