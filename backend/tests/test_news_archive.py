import json
from datetime import UTC, datetime, time

import pandas as pd
import pytest

from news_features import (
    TransformerSentimentScorer,
    align_article_to_session,
    deduplicate_articles,
    filter_ticker_relevance,
    load_news_archive,
    normalise_news_articles,
)
from news_import import main as import_news


def test_archive_requires_timestamped_provider_neutral_contract(tmp_path):
    archive = tmp_path / "news.jsonl"
    archive.write_text(
        json.dumps(
            {
                "provider": "licensed",
                "ticker": "AAPL",
                "title": "Apple earnings",
                "published_at_utc": "2024-01-02T12:00:00Z",
                "received_at_utc": "2024-01-02T12:01:00Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    articles = load_news_archive(archive)
    assert articles[0]["ticker"] == "AAPL"
    assert articles[0]["received_at"] > articles[0]["published_at"]


def test_archive_rejects_missing_or_impossible_timestamps(tmp_path):
    archive = tmp_path / "bad.jsonl"
    archive.write_text(
        json.dumps(
            {
                "provider": "licensed",
                "ticker": "AAPL",
                "title": "Apple",
                "published_at_utc": "2024-01-02T12:00:00Z",
                "received_at_utc": "2024-01-01T12:00:00Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid timestamps"):
        load_news_archive(archive)


def test_deduplication_prefers_stable_provider_id_then_url_then_title():
    articles = normalise_news_articles(
        [
            {
                "provider": "one",
                "article_id": "abc",
                "title": "Earnings beat",
                "published_at_utc": "2024-01-02T10:00:00Z",
            },
            {
                "provider": "two",
                "article_id": "abc",
                "title": "Different syndicated title",
                "published_at_utc": "2024-01-02T11:00:00Z",
            },
            {
                "provider": "two",
                "title": "Earnings beat",
                "published_at_utc": "2024-01-02T12:00:00Z",
            },
        ]
    )
    assert len(deduplicate_articles(articles)) == 2


def test_relevance_rejects_unrelated_article_and_handles_aliases():
    articles = normalise_news_articles(
        [
            {"title": "Apple launches product", "published_at_utc": "2024-01-02T12:00:00Z"},
            {"title": "AAPL rises", "published_at_utc": "2024-01-02T12:00:00Z"},
            {"title": "Banana prices rise", "published_at_utc": "2024-01-02T12:00:00Z"},
        ]
    )
    relevant = filter_ticker_relevance(articles, "AAPL", ("Apple",))
    assert [article["title"] for article in relevant] == ["Apple launches product", "AAPL rises"]


def test_session_alignment_prevents_after_close_and_weekend_leakage():
    sessions = pd.DatetimeIndex(["2024-01-05", "2024-01-08", "2024-01-09"], tz="UTC")
    friday_after_close = {
        "published_at": datetime(2024, 1, 5, 17, tzinfo=UTC),
        "received_at": datetime(2024, 1, 5, 17, tzinfo=UTC),
    }
    friday_before_close = {
        "published_at": datetime(2024, 1, 5, 15, tzinfo=UTC),
        "received_at": datetime(2024, 1, 5, 15, tzinfo=UTC),
    }
    assert (
        align_article_to_session(friday_after_close, sessions, close_time=time(16)) == sessions[1]
    )
    assert (
        align_article_to_session(friday_before_close, sessions, close_time=time(16)) == sessions[0]
    )


def test_transformer_scorer_requires_a_pinned_revision():
    with pytest.raises(ValueError, match="pinned"):
        TransformerSentimentScorer("ProsusAI/finbert", "")


def test_news_import_writes_hashed_immutable_snapshot(tmp_path):
    source = tmp_path / "news.jsonl"
    source.write_text(
        json.dumps(
            {
                "provider": "licensed",
                "ticker": "AAPL",
                "title": "Apple earnings",
                "published_at_utc": "2024-01-02T12:00:00Z",
                "received_at_utc": "2024-01-02T12:01:00Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "snapshot"

    assert import_news(["--input", str(source), "--output", str(output)]) == 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["article_count"] == 1
    assert len(manifest["archive_sha256"]) == 64
    assert len(manifest["manifest_sha256"]) == 64

    with pytest.raises(FileExistsError, match="absent or empty"):
        import_news(["--input", str(source), "--output", str(output)])
