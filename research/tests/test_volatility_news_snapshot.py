from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from volatility_forecasting.news import NewsEvent, NewsLicenseNotAcknowledged
from volatility_forecasting.news_snapshot import (
    NewsSnapshotError,
    load_news_snapshot,
    save_news_snapshot,
)


def _event(event_id: str = "event-1") -> NewsEvent:
    return NewsEvent(
        event_id=event_id,
        cluster_id=event_id,
        source="sec.gov",
        first_seen_at=pd.Timestamp("2025-01-02T20:15:00Z"),
        published_at=pd.Timestamp("2025-01-02T20:00:00Z"),
        timestamp_quality="first_seen_only",
        tickers=("MSFT",),
        topics=("regulatory_filing",),
        positive_probability=0.0,
        neutral_probability=1.0,
        negative_probability=0.0,
        novelty=0.5,
        severity=0.4,
        confidence=1.0,
        source_reliability=1.0,
        canonical_url_hash="abc123",
        license_class="sec_public_filing_metadata",
    )


def _save(directory: Path, events: list[NewsEvent]) -> dict[str, object]:
    return save_news_snapshot(
        directory,
        events,
        provider="fixture",
        license_acknowledged=True,
        coverage_start="2025-01-01T00:00:00Z",
        coverage_end_exclusive="2025-02-01T00:00:00Z",
        provenance={"source_files": 1},
    )


def test_news_snapshot_round_trip_is_canonical_and_text_free(tmp_path: Path) -> None:
    directory = tmp_path / "snapshot"
    manifest = _save(directory, [_event("event-2"), _event("event-1")])
    loaded, verified = load_news_snapshot(directory)
    assert [event.event_id for event in loaded] == ["event-1", "event-2"]
    assert verified == manifest
    assert verified["contains_article_text"] is False
    assert verified["provenance"] == {"source_files": 1}
    rows = [
        json.loads(line)
        for line in (directory / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all("headline" not in row and "article_text" not in row for row in rows)


def test_news_snapshot_rejects_tampered_event_bytes(tmp_path: Path) -> None:
    directory = tmp_path / "snapshot"
    _save(directory, [_event()])
    path = directory / "events.jsonl"
    path.write_text(path.read_text(encoding="utf-8").replace("0.4", "0.9"), encoding="utf-8")
    with pytest.raises(NewsSnapshotError, match="checksum"):
        load_news_snapshot(directory)


def test_news_snapshot_rejects_manifest_path_substitution(tmp_path: Path) -> None:
    directory = tmp_path / "snapshot"
    _save(directory, [_event()])
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["events_file"] = "../outside.jsonl"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(NewsSnapshotError, match="path"):
        load_news_snapshot(directory)


def test_news_snapshot_requires_license_acknowledgement(tmp_path: Path) -> None:
    with pytest.raises(NewsLicenseNotAcknowledged):
        save_news_snapshot(
            tmp_path / "snapshot",
            [_event()],
            provider="fixture",
            license_acknowledged=False,
            coverage_start="2025-01-01T00:00:00Z",
            coverage_end_exclusive="2025-02-01T00:00:00Z",
        )


def test_news_snapshot_rejects_duplicate_event_ids(tmp_path: Path) -> None:
    with pytest.raises(NewsSnapshotError, match="duplicate"):
        save_news_snapshot(
            tmp_path / "snapshot",
            [_event(), _event()],
            provider="fixture",
            license_acknowledged=True,
            coverage_start="2025-01-01T00:00:00Z",
            coverage_end_exclusive="2025-02-01T00:00:00Z",
        )


def test_news_snapshot_rejects_events_outside_declared_coverage(tmp_path: Path) -> None:
    with pytest.raises(NewsSnapshotError, match="outside declared"):
        save_news_snapshot(
            tmp_path / "snapshot",
            [_event()],
            provider="fixture",
            license_acknowledged=True,
            coverage_start="2025-02-01T00:00:00Z",
            coverage_end_exclusive="2025-03-01T00:00:00Z",
        )
