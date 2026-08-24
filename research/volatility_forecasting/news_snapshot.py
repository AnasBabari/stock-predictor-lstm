"""Immutable, checksummed storage for point-in-time news-event snapshots."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import fields
from pathlib import Path

import pandas as pd

from .news import (
    NEWS_SCHEMA_VERSION,
    NewsEvent,
    NewsLicenseNotAcknowledged,
    NewsValidationError,
    build_news_snapshot,
)

NEWS_EVENTS_FILENAME = "events.jsonl"
NEWS_MANIFEST_FILENAME = "manifest.json"
NEWS_SNAPSHOT_FORMAT = "stocklstm-news-snapshot-v1"
_EVENT_FIELDS = frozenset(field.name for field in fields(NewsEvent))


class NewsSnapshotError(RuntimeError):
    """An immutable news snapshot is missing, corrupt, or incompatible."""


def _event_payload(event: NewsEvent) -> dict[str, object]:
    return {
        "event_id": event.event_id,
        "cluster_id": event.cluster_id,
        "source": event.source,
        "first_seen_at": event.first_seen_at.isoformat(),
        "published_at": event.published_at.isoformat() if event.published_at is not None else None,
        "timestamp_quality": event.timestamp_quality,
        "tickers": list(event.tickers),
        "topics": list(event.topics),
        "positive_probability": event.positive_probability,
        "neutral_probability": event.neutral_probability,
        "negative_probability": event.negative_probability,
        "novelty": event.novelty,
        "severity": event.severity,
        "confidence": event.confidence,
        "source_reliability": event.source_reliability,
        "headline_hash": event.headline_hash,
        "canonical_url_hash": event.canonical_url_hash,
        "language": event.language,
        "license_class": event.license_class,
        "volume": event.volume,
    }


def _event_from_payload(payload: object) -> NewsEvent:
    if not isinstance(payload, dict):
        raise NewsSnapshotError("news event row must be a JSON object")
    if set(payload) != _EVENT_FIELDS:
        raise NewsSnapshotError("news event row fields do not match the frozen schema")
    tickers = payload["tickers"]
    topics = payload["topics"]
    if not isinstance(tickers, list) or not all(isinstance(value, str) for value in tickers):
        raise NewsSnapshotError("news event tickers must be a JSON string array")
    if not isinstance(topics, list) or not all(isinstance(value, str) for value in topics):
        raise NewsSnapshotError("news event topics must be a JSON string array")
    try:
        return NewsEvent(
            event_id=str(payload["event_id"]),
            cluster_id=str(payload["cluster_id"]),
            source=str(payload["source"]),
            first_seen_at=pd.Timestamp(str(payload["first_seen_at"])),
            published_at=(
                pd.Timestamp(str(payload["published_at"]))
                if payload["published_at"] is not None
                else None
            ),
            timestamp_quality=str(payload["timestamp_quality"]),  # type: ignore[arg-type]
            tickers=tuple(tickers),
            topics=tuple(topics),
            positive_probability=float(payload["positive_probability"]),
            neutral_probability=float(payload["neutral_probability"]),
            negative_probability=float(payload["negative_probability"]),
            novelty=float(payload["novelty"]),
            severity=float(payload["severity"]),
            confidence=float(payload["confidence"]),
            source_reliability=float(payload["source_reliability"]),
            headline_hash=str(payload["headline_hash"]),
            canonical_url_hash=str(payload["canonical_url_hash"]),
            language=str(payload["language"]),
            license_class=str(payload["license_class"]),
            volume=float(payload["volume"]),
        )
    except (KeyError, TypeError, ValueError, NewsValidationError) as error:
        raise NewsSnapshotError("news event row is invalid") from error


def _canonical_events(events: Sequence[NewsEvent]) -> tuple[NewsEvent, ...]:
    ordered = tuple(sorted(events, key=lambda event: event.event_id))
    identities = [event.event_id for event in ordered]
    if len(set(identities)) != len(identities):
        raise NewsSnapshotError("news snapshot contains duplicate event IDs")
    return ordered


def _coverage_timestamp(value: pd.Timestamp | str, *, field: str) -> pd.Timestamp:
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise NewsSnapshotError(f"{field} is not a valid timestamp") from error
    if parsed.tzinfo is None:
        raise NewsSnapshotError(f"{field} must be timezone-aware")
    return parsed.tz_convert("UTC")


def _events_bytes(events: Sequence[NewsEvent]) -> bytes:
    rows = [
        json.dumps(_event_payload(event), sort_keys=True, separators=(",", ":"))
        for event in _canonical_events(events)
    ]
    return (("\n".join(rows) + "\n") if rows else "").encode("utf-8")


def save_news_snapshot(
    directory: Path,
    events: Sequence[NewsEvent],
    *,
    provider: str,
    license_acknowledged: bool,
    coverage_start: pd.Timestamp | str,
    coverage_end_exclusive: pd.Timestamp | str,
    provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Write an immutable metadata-only snapshot and return its manifest."""
    if not license_acknowledged:
        raise NewsLicenseNotAcknowledged(
            "News provider terms must be reviewed before snapshot construction."
        )
    target = directory.resolve()
    if target.exists() and any(target.iterdir()):
        raise NewsSnapshotError("news snapshot directory must be empty")
    target.mkdir(parents=True, exist_ok=True)

    ordered = _canonical_events(events)
    coverage_start_utc = _coverage_timestamp(coverage_start, field="coverage_start")
    coverage_end_utc = _coverage_timestamp(
        coverage_end_exclusive,
        field="coverage_end_exclusive",
    )
    if coverage_end_utc <= coverage_start_utc:
        raise NewsSnapshotError("news snapshot coverage end must follow its start")
    outside_coverage = [
        event.event_id
        for event in ordered
        if event.eligible_at is not None
        and not coverage_start_utc <= event.eligible_at < coverage_end_utc
    ]
    if outside_coverage:
        raise NewsSnapshotError("news event eligibility falls outside declared provider coverage")
    encoded = _events_bytes(ordered)
    event_digest = hashlib.sha256(encoded).hexdigest()
    base = build_news_snapshot(
        ordered,
        license_acknowledged=True,
        provider=provider,
    )
    provenance_payload = dict(provenance or {})
    try:
        provenance_bytes = json.dumps(
            provenance_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise NewsSnapshotError("news snapshot provenance must be finite JSON data") from error
    eligible = [event.eligible_at for event in ordered if event.eligible_at is not None]
    manifest = {
        **base,
        "snapshot_format": NEWS_SNAPSHOT_FORMAT,
        "events_file": NEWS_EVENTS_FILENAME,
        "events_sha256": event_digest,
        "eligible_start": min(eligible).isoformat() if eligible else None,
        "eligible_end": max(eligible).isoformat() if eligible else None,
        "coverage_start": coverage_start_utc.isoformat(),
        "coverage_end_exclusive": coverage_end_utc.isoformat(),
        "contains_article_text": False,
        "provenance": provenance_payload,
        "provenance_sha256": hashlib.sha256(provenance_bytes).hexdigest(),
    }
    events_tmp = target / f".{NEWS_EVENTS_FILENAME}.tmp"
    manifest_tmp = target / f".{NEWS_MANIFEST_FILENAME}.tmp"
    events_tmp.write_bytes(encoded)
    manifest_tmp.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(events_tmp, target / NEWS_EVENTS_FILENAME)
    os.replace(manifest_tmp, target / NEWS_MANIFEST_FILENAME)
    return manifest


def load_news_snapshot(
    directory: Path,
    *,
    maximum_events: int = 10_000_000,
    maximum_bytes: int = 2_000_000_000,
) -> tuple[tuple[NewsEvent, ...], dict[str, object]]:
    """Verify and load an immutable event snapshot without article text."""
    if maximum_events < 1 or maximum_bytes < 1:
        raise ValueError("news snapshot resource limits must be positive")
    target = directory.resolve()
    manifest_path = target / NEWS_MANIFEST_FILENAME
    events_path = target / NEWS_EVENTS_FILENAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise NewsSnapshotError("news snapshot manifest is missing or invalid") from error
    if not isinstance(manifest, dict):
        raise NewsSnapshotError("news snapshot manifest must be an object")
    required = {
        "schema_version",
        "snapshot_format",
        "snapshot_id",
        "digest",
        "provider",
        "event_count",
        "license_acknowledged",
        "events_file",
        "events_sha256",
        "eligible_start",
        "eligible_end",
        "coverage_start",
        "coverage_end_exclusive",
        "contains_article_text",
        "provenance",
        "provenance_sha256",
    }
    if not required.issubset(manifest):
        raise NewsSnapshotError("news snapshot manifest is incomplete")
    if manifest["schema_version"] != NEWS_SCHEMA_VERSION:
        raise NewsSnapshotError("news snapshot schema is incompatible")
    if manifest["snapshot_format"] != NEWS_SNAPSHOT_FORMAT:
        raise NewsSnapshotError("news snapshot format is incompatible")
    if manifest["events_file"] != NEWS_EVENTS_FILENAME:
        raise NewsSnapshotError("news snapshot events path is not allowed")
    if manifest["license_acknowledged"] is not True:
        raise NewsSnapshotError("news snapshot provider terms were not acknowledged")
    if manifest["contains_article_text"] is not False:
        raise NewsSnapshotError("news snapshot must not contain article text")
    if not isinstance(manifest["provider"], str) or not manifest["provider"].strip():
        raise NewsSnapshotError("news snapshot provider is invalid")
    if not isinstance(manifest["provenance"], dict):
        raise NewsSnapshotError("news snapshot provenance must be an object")
    try:
        provenance_bytes = json.dumps(
            manifest["provenance"],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise NewsSnapshotError("news snapshot provenance is invalid") from error
    if hashlib.sha256(provenance_bytes).hexdigest() != manifest["provenance_sha256"]:
        raise NewsSnapshotError("news snapshot provenance checksum does not match")
    coverage_start = _coverage_timestamp(manifest["coverage_start"], field="coverage_start")
    coverage_end = _coverage_timestamp(
        manifest["coverage_end_exclusive"],
        field="coverage_end_exclusive",
    )
    if coverage_end <= coverage_start:
        raise NewsSnapshotError("news snapshot coverage end must follow its start")

    try:
        size = events_path.stat().st_size
        encoded = events_path.read_bytes()
    except OSError as error:
        raise NewsSnapshotError("news snapshot event file is missing") from error
    if size > maximum_bytes:
        raise NewsSnapshotError("news snapshot exceeds the configured byte limit")
    digest = hashlib.sha256(encoded).hexdigest()
    if digest != manifest["events_sha256"]:
        raise NewsSnapshotError("news snapshot event checksum does not match")

    try:
        text = encoded.decode("utf-8")
    except UnicodeDecodeError as error:
        raise NewsSnapshotError("news snapshot events are not valid UTF-8") from error
    events: list[NewsEvent] = []
    for line_number, raw in enumerate(text.splitlines(), start=1):
        if len(events) >= maximum_events:
            raise NewsSnapshotError("news snapshot exceeds the configured event limit")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise NewsSnapshotError(f"news event line {line_number} is invalid JSON") from error
        events.append(_event_from_payload(payload))
    ordered = _canonical_events(events)
    if tuple(events) != ordered:
        raise NewsSnapshotError("news snapshot events are not canonically ordered")
    if not isinstance(manifest["event_count"], int) or manifest["event_count"] != len(ordered):
        raise NewsSnapshotError("news snapshot event count does not match")
    rebuilt = build_news_snapshot(
        ordered,
        license_acknowledged=True,
        provider=str(manifest["provider"]),
    )
    if rebuilt["snapshot_id"] != manifest["snapshot_id"] or rebuilt["digest"] != manifest["digest"]:
        raise NewsSnapshotError("news snapshot identity does not match its events")
    eligible = [event.eligible_at for event in ordered if event.eligible_at is not None]
    expected_start = min(eligible).isoformat() if eligible else None
    expected_end = max(eligible).isoformat() if eligible else None
    if manifest["eligible_start"] != expected_start or manifest["eligible_end"] != expected_end:
        raise NewsSnapshotError("news snapshot eligibility bounds do not match its events")
    if any(
        event.eligible_at is not None and not coverage_start <= event.eligible_at < coverage_end
        for event in ordered
    ):
        raise NewsSnapshotError("news event eligibility falls outside declared provider coverage")
    return ordered, manifest
