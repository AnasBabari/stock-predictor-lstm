"""Tests for historical news lake causality and negative controls."""

from __future__ import annotations

import pandas as pd
import pytest

from research.volatility_forecasting.news_fusion_v10 import (
    evaluate_news_gain,
    generate_negative_controls,
)
from research.volatility_forecasting.news_lake_v10 import (
    HistoricalNewsRecord,
    align_article_to_trading_session,
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
    # Available at must be max(15:00, 15:02, 15:05) -> 15:05:00Z
    assert record.available_at == "2024-01-02T15:05:00Z"


def test_align_article_after_close_rolls_to_next_session() -> None:
    # Exchange close is 21:00:00Z (4:00 PM EST / 21:00 UTC)
    # Article at 21:15:00Z must roll to 2024-01-03
    session = align_article_to_trading_session(
        available_at_utc="2024-01-02T21:15:00Z",
        exchange_session_date="2024-01-02",
        exchange_close_time_utc="21:00:00Z",
        next_session_date="2024-01-03",
    )
    assert session == "2024-01-03"

    # Article at 18:00:00Z stays in 2024-01-02
    session_intraday = align_article_to_trading_session(
        available_at_utc="2024-01-02T18:00:00Z",
        exchange_session_date="2024-01-02",
        exchange_close_time_utc="21:00:00Z",
        next_session_date="2024-01-03",
    )
    assert session_intraday == "2024-01-02"


def test_evaluate_news_gain_rejects_when_control_beats_model() -> None:
    numeric_qlike = 0.50
    fused_qlike = 0.48
    controls = {
        "shuffled_news": 0.47,  # Shuffled news spurious beat
        "delayed_news": 0.51,
        "count_only": 0.50,
    }
    passed, reason = evaluate_news_gain(numeric_qlike, fused_qlike, controls)
    assert passed is False
    assert "shuffled_news" in reason


def test_evaluate_news_gain_passes_when_superior_to_all_controls() -> None:
    numeric_qlike = 0.50
    fused_qlike = 0.46
    controls = {
        "shuffled_news": 0.49,
        "delayed_news": 0.51,
        "count_only": 0.49,
    }
    passed, _ = evaluate_news_gain(numeric_qlike, fused_qlike, controls)
    assert passed is True
