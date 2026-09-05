import hashlib

import pytest

from research.price_forecasting.news_archive import _normalise_article, merge_news_archive


def record(provider="alpaca", headline="Original"):
    return {
        "id": "1",
        "provider": provider,
        "headline": headline,
        "published_at": "2023-01-03T14:00:00Z",
    }


def test_manifest_hashes_exact_bytes_and_separates_providers(tmp_path):
    path = tmp_path / "news.jsonl"
    manifest = merge_news_archive(path, [record(), record("yahoo")])
    assert manifest["article_count"] == 2
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["sha256"]
    assert b"\r\n" not in path.read_bytes()
    assert merge_news_archive(path, [record()])["article_count"] == 2


def test_changed_revision_cannot_silently_replace_archive(tmp_path):
    path = tmp_path / "news.jsonl"
    merge_news_archive(path, [record()])
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Conflicting"):
        merge_news_archive(path, [record(headline="Revised")])
    assert path.read_bytes() == before


def test_alpaca_string_content_does_not_imply_yahoo():
    item = _normalise_article(
        {
            "id": "1",
            "headline": "News",
            "content": "",
            "created_at": "2023-03-01T10:00:00Z",
            "updated_at": "2023-03-02T10:00:00Z",
        },
        "AAPL",
        "2026-09-05T10:00:00Z",
    )
    assert item["provider"] == "alpaca"
    assert item["provider_updated_at"] == "2023-03-02T10:00:00Z"
