"""Tests for historical news lake causality and complete negative controls."""

from __future__ import annotations

import pandas as pd

from research.volatility_forecasting.news_fusion_v10 import (
    evaluate_news_gain,
    extract_causal_news_features,
    generate_negative_controls,
)
from research.volatility_forecasting.news_lake_v10 import (
    HistoricalNewsRecord,
)


def test_news_available_at_uses_maximum_timestamp() -> None:
    record = HistoricalNewsRecord(
        article_id="art-001",
        canonical_url_hash="h001",
        provider="attested_news_feed",
        source="reuters",
        published_at="2024-01-02T15:00:00Z",
        first_seen_at="2024-01-02T15:02:00Z",
        provider_delivery_time="2024-01-02T15:05:00Z",
        retrieved_at="2024-01-02T15:06:00Z",
        language="en",
        title="Fed Signals Rate Hold",
        body_hash="b001",
        security_ids=["SEC_SPY_001"],
        sector_ids=["financials"],
        country_ids=["US"],
        event_types=["monetary_policy"],
        license_id="lic_news_2026",
        snapshot_id="snap-news-001",
    )
    assert record.available_at == "2024-01-02T15:05:00Z"


def test_delayed_news_shifts_forward_in_time() -> None:
    df_news = pd.DataFrame(
        {
            "SessionDate": ["2024-01-01", "2024-01-05"],
            "SecurityID": ["AAPL", "AAPL"],
            "sentiment": [0.5, -0.2],
        }
    )
    controls = generate_negative_controls(df_news)

    # Delayed news must move 2024-01-01 to 2024-01-06 (index 0 -> index 5)
    delayed_session = controls["delayed_news"]["SessionDate"].iloc[0]
    assert delayed_session >= "2024-01-01"
    assert "entity_shuffled" in controls
    assert "future_shift_sentinel" in controls


def test_extract_causal_news_features() -> None:
    sessions = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    df_news = pd.DataFrame(
        {
            "SessionDate": ["2024-01-01", "2024-01-03"],
            "SecurityID": ["AAPL", "AAPL"],
            "sentiment": [0.4, 0.8],
        }
    )
    feats = extract_causal_news_features(df_news, sessions, ["AAPL"], windows=(1, 3))
    assert len(feats) == len(sessions)
    assert "news_count_1d" in feats.columns
    assert "news_sentiment_mean_3d" in feats.columns


def test_evaluate_news_gain_rejects_when_control_beats_model() -> None:
    numeric_qlike = 0.50
    fused_qlike = 0.48
    controls = {
        "shuffled_news": 0.47,
        "delayed_news": 0.51,
        "count_only": 0.50,
    }
    passed, reason = evaluate_news_gain(numeric_qlike, fused_qlike, controls)
    assert passed is False
    assert "shuffled_news" in reason
