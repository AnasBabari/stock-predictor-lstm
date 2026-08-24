from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from volatility_forecasting.news import (
    NEWS_FEATURE_NAMES_V1,
    NewsEvent,
    NewsLicenseNotAcknowledged,
    NewsOrigin,
    NewsValidationError,
    aggregate_news_features,
    build_news_snapshot,
)


def _event(
    event_id: str,
    at: str,
    *,
    cluster_id: str | None = None,
    tickers: tuple[str, ...] = ("MSFT",),
    topics: tuple[str, ...] = ("earnings",),
    quality: str = "precise",
    published_at: str | None = None,
    severity: float = 0.7,
) -> NewsEvent:
    return NewsEvent(
        event_id=event_id,
        cluster_id=cluster_id or event_id,
        source="wire.example",
        first_seen_at=pd.Timestamp(at),
        published_at=pd.Timestamp(published_at or at) if quality == "precise" else None,
        timestamp_quality=quality,
        tickers=tickers,
        topics=topics,
        positive_probability=0.1,
        neutral_probability=0.2,
        negative_probability=0.7,
        novelty=0.8,
        severity=severity,
        confidence=0.9,
        source_reliability=0.95,
    )


def _feature(matrix, name: str) -> float:
    return float(matrix.values[0, NEWS_FEATURE_NAMES_V1.index(name)])


def test_post_cutoff_article_cannot_change_origin_features() -> None:
    origin = NewsOrigin("MSFT", pd.Timestamp("2026-01-05T21:00:00Z"))
    visible = _event("before", "2026-01-05T20:00:00Z")
    future = _event("after", "2026-01-05T21:00:01Z", severity=1.0)

    baseline = aggregate_news_features([visible], [origin])
    changed = aggregate_news_features([visible, future], [origin])

    np.testing.assert_array_equal(baseline.values, changed.values)


def test_syndicated_cluster_is_counted_once_as_of_cutoff() -> None:
    origin = NewsOrigin("MSFT", pd.Timestamp("2026-01-05T21:00:00Z"))
    first = _event("mention-a", "2026-01-05T19:00:00Z", cluster_id="story-1")
    duplicate = _event("mention-b", "2026-01-05T19:10:00Z", cluster_id="story-1")

    one = aggregate_news_features([first], [origin])
    two = aggregate_news_features([first, duplicate], [origin])

    assert _feature(one, "News_Ticker_Intensity_1D") == _feature(two, "News_Ticker_Intensity_1D")


def test_frozen_exposure_map_propagates_oil_disruption_without_fake_ticker_link() -> None:
    origin = NewsOrigin("XOM", pd.Timestamp("2026-01-05T21:00:00Z"))
    event = _event(
        "oil-shock",
        "2026-01-05T18:00:00Z",
        tickers=(),
        topics=("military_conflict", "oil_supply"),
        severity=1.0,
    )
    matrix = aggregate_news_features(
        [event],
        [origin],
        exposure_map={"XOM": {"oil_supply": 0.9, "military_conflict": 0.4}},
    )

    assert _feature(matrix, "News_Ticker_Intensity_1D") == 0.0
    assert _feature(matrix, "News_Exposure_Intensity_1D") > 0.0
    assert _feature(matrix, "News_Exposure_Conflict_Severity_3D") > 0.0
    assert _feature(matrix, "News_Exposure_Commodity_Severity_3D") > 0.0


def test_unknown_timestamp_is_excluded_and_first_seen_only_is_flagged() -> None:
    origin = NewsOrigin("MSFT", pd.Timestamp("2026-01-05T21:00:00Z"))
    unknown = _event("unknown", "2026-01-05T18:00:00Z", quality="unknown")
    first_seen = _event("first-seen", "2026-01-05T19:00:00Z", quality="first_seen_only")
    matrix = aggregate_news_features([unknown, first_seen], [origin])

    assert _feature(matrix, "News_Ticker_Intensity_1D") > 0.0
    assert _feature(matrix, "News_Low_Timestamp_Quality_Fraction_3D") == 1.0


def test_date_only_event_cannot_enter_same_day_close() -> None:
    event = _event("date-only", "2026-01-05T00:00:00Z", quality="date_only")
    same_day = aggregate_news_features(
        [event],
        [NewsOrigin("MSFT", pd.Timestamp("2026-01-05T21:00:00Z"))],
    )
    next_day = aggregate_news_features(
        [event],
        [NewsOrigin("MSFT", pd.Timestamp("2026-01-06T00:01:00Z"))],
    )
    assert _feature(same_day, "News_Ticker_Intensity_1D") == 0.0
    assert _feature(next_day, "News_Ticker_Intensity_1D") > 0.0


def test_shared_cutoff_results_match_independent_aggregation() -> None:
    events = [
        _event("a", "2026-01-05T18:00:00Z"),
        _event("b", "2026-01-05T19:00:00Z", tickers=("AAPL",)),
    ]
    origins = [
        NewsOrigin("MSFT", pd.Timestamp("2026-01-05T21:00:00Z")),
        NewsOrigin("AAPL", pd.Timestamp("2026-01-05T21:00:00Z")),
    ]
    together = aggregate_news_features(events, origins)
    separately = np.vstack([aggregate_news_features(events, [origin]).values for origin in origins])
    np.testing.assert_array_equal(together.values, separately)


def test_snapshot_digest_is_order_invariant_and_license_gated() -> None:
    first = _event("a", "2026-01-05T18:00:00Z")
    second = _event("b", "2026-01-05T19:00:00Z")
    with pytest.raises(NewsLicenseNotAcknowledged):
        build_news_snapshot([first], license_acknowledged=False, provider="fixture")

    left = build_news_snapshot([first, second], license_acknowledged=True, provider="fixture")
    right = build_news_snapshot([second, first], license_acknowledged=True, provider="fixture")
    assert left["snapshot_id"] == right["snapshot_id"]


def test_naive_timestamps_fail_closed() -> None:
    with pytest.raises(NewsValidationError, match="timezone-aware"):
        _event("bad", "2026-01-05T18:00:00")
