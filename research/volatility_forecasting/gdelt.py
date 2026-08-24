"""Point-in-time GDELT event ingestion for geopolitical and macro news shocks.

GDELT's Event stream is not treated as a price-direction oracle. It supplies
timestamped event intensity, tone, and severity inputs that must pass the same
locked market-only versus market-plus-news ablation as every other feature.
"""

from __future__ import annotations

import csv
import hashlib
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from .news import NewsEvent, NewsValidationError

GDELT_V2_EXPORT_COLUMNS = 61
GDELT_V2_BASE_URL = "https://data.gdeltproject.org/gdeltv2"
GDELT_V1_EXPORT_COLUMNS = 58
GDELT_V1_BASE_URL = "https://data.gdeltproject.org/events"
_CONFLICT_ROOT_CODES = frozenset({"15", "16", "17", "18", "19", "20"})


@dataclass(frozen=True)
class GdeltArchiveFile:
    available_at: datetime
    url: str


@dataclass(frozen=True)
class GdeltV1DailyArchive:
    archive_date: date
    available_at: datetime
    url: str


@dataclass(frozen=True)
class GdeltV1EventRow:
    global_event_id: str
    sql_date: str
    actor1_name: str
    actor2_name: str
    event_code: str
    event_base_code: str
    event_root_code: str
    quad_class: int
    goldstein_scale: float
    num_mentions: int
    num_sources: int
    num_articles: int
    average_tone: float
    date_added: date
    source_url: str

    def __post_init__(self) -> None:
        if not self.global_event_id or not self.event_code or not self.event_root_code:
            raise NewsValidationError("GDELT event identity and codes are required")
        if self.quad_class not in (1, 2, 3, 4):
            raise NewsValidationError("GDELT quad class must be in [1, 4]")
        if any(value < 0 for value in (self.num_mentions, self.num_sources, self.num_articles)):
            raise NewsValidationError("GDELT mention counts cannot be negative")
        if not np.isfinite(self.goldstein_scale) or not np.isfinite(self.average_tone):
            raise NewsValidationError("GDELT tone and Goldstein scale must be finite")


@dataclass(frozen=True)
class GdeltDailyAggregationStats:
    archive_date: str
    total_rows: int
    retained_rows: int
    invalid_rows: int
    output_events: int


@dataclass(frozen=True)
class TickerAliasMatcher:
    pattern: re.Pattern[str]
    alias_to_tickers: Mapping[str, tuple[str, ...]]

    def match(self, text: str) -> tuple[str, ...]:
        matched: set[str] = set()
        for result in self.pattern.finditer(text):
            matched.update(self.alias_to_tickers[result.group(0).casefold()])
        return tuple(sorted(matched))


def compile_ticker_aliases(aliases: Mapping[str, tuple[str, ...]]) -> TickerAliasMatcher:
    alias_to_tickers: dict[str, set[str]] = defaultdict(set)
    for ticker, names in aliases.items():
        for name in names:
            normalized = name.strip().casefold()
            if len(normalized) < 3:
                raise NewsValidationError(
                    "GDELT company aliases must contain at least three characters"
                )
            alias_to_tickers[normalized].add(ticker.upper())
    if not alias_to_tickers:
        # A never-matching expression preserves the same matcher contract for
        # market-only event snapshots.
        return TickerAliasMatcher(re.compile(r"(?!x)x"), {})
    ordered = sorted(alias_to_tickers, key=lambda value: (-len(value), value))
    pattern = re.compile(
        rf"(?<!\w)(?:{'|'.join(re.escape(value) for value in ordered)})(?!\w)",
        flags=re.IGNORECASE,
    )
    return TickerAliasMatcher(
        pattern=pattern,
        alias_to_tickers={
            alias: tuple(sorted(tickers)) for alias, tickers in alias_to_tickers.items()
        },
    )


@dataclass(frozen=True)
class GdeltEventRow:
    global_event_id: str
    sql_date: str
    actor1_name: str
    actor2_name: str
    event_code: str
    event_base_code: str
    event_root_code: str
    quad_class: int
    goldstein_scale: float
    num_mentions: int
    num_sources: int
    num_articles: int
    average_tone: float
    date_added: pd.Timestamp
    source_url: str

    def __post_init__(self) -> None:
        if not self.global_event_id or not self.event_code or not self.event_root_code:
            raise NewsValidationError("GDELT event identity and codes are required")
        if self.quad_class not in (1, 2, 3, 4):
            raise NewsValidationError("GDELT quad class must be in [1, 4]")
        if any(value < 0 for value in (self.num_mentions, self.num_sources, self.num_articles)):
            raise NewsValidationError("GDELT mention counts cannot be negative")
        if not np.isfinite(self.goldstein_scale) or not np.isfinite(self.average_tone):
            raise NewsValidationError("GDELT tone and Goldstein scale must be finite")
        added = pd.Timestamp(self.date_added)
        if added.tzinfo is None:
            added = added.tz_localize("UTC")
        object.__setattr__(self, "date_added", added.tz_convert("UTC"))


def iter_gdelt_v2_archives(start: datetime, end: datetime) -> tuple[GdeltArchiveFile, ...]:
    """Return deterministic 15-minute Event archive URLs in [start, end)."""
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("GDELT archive bounds must be timezone-aware")
    start_utc = start.astimezone(UTC)
    end_utc = end.astimezone(UTC)
    if start_utc.minute % 15 or start_utc.second or start_utc.microsecond:
        raise ValueError("GDELT archive start must be aligned to 15 minutes")
    if end_utc <= start_utc:
        raise ValueError("GDELT archive end must follow start")
    files: list[GdeltArchiveFile] = []
    current = start_utc
    while current < end_utc:
        stamp = current.strftime("%Y%m%d%H%M%S")
        files.append(
            GdeltArchiveFile(
                available_at=current,
                url=f"{GDELT_V2_BASE_URL}/{stamp}.export.CSV.zip",
            )
        )
        current += timedelta(minutes=15)
    return tuple(files)


def iter_gdelt_v1_daily_archives(
    start: date,
    end: date,
) -> tuple[GdeltV1DailyArchive, ...]:
    """Return daily files in [start, end) with conservative known-at times.

    GDELT documents each daily file as being posted the following morning. We
    use noon UTC on that following day, deliberately later than the stated
    schedule, instead of pretending its daily event date was known intraday.
    """
    if end <= start:
        raise ValueError("GDELT daily archive end must follow start")
    files: list[GdeltV1DailyArchive] = []
    current = start
    while current < end:
        files.append(
            GdeltV1DailyArchive(
                archive_date=current,
                available_at=datetime.combine(
                    current + timedelta(days=1),
                    time(12),
                    tzinfo=UTC,
                ),
                url=f"{GDELT_V1_BASE_URL}/{current:%Y%m%d}.export.CSV.zip",
            )
        )
        current += timedelta(days=1)
    return tuple(files)


def _integer(value: str, *, field: str) -> int:
    try:
        return int(value or "0")
    except ValueError as error:
        raise NewsValidationError(f"invalid GDELT integer: {field}") from error


def _number(value: str, *, field: str) -> float:
    try:
        parsed = float(value or "0")
    except ValueError as error:
        raise NewsValidationError(f"invalid GDELT number: {field}") from error
    if not np.isfinite(parsed):
        raise NewsValidationError(f"non-finite GDELT number: {field}")
    return parsed


def parse_gdelt_v2_export_line(line: str) -> GdeltEventRow:
    fields = next(csv.reader([line], delimiter="\t"))
    if len(fields) != GDELT_V2_EXPORT_COLUMNS:
        raise NewsValidationError(
            f"GDELT v2 export row must have {GDELT_V2_EXPORT_COLUMNS} columns"
        )
    try:
        added = pd.to_datetime(fields[59], format="%Y%m%d%H%M%S", utc=True)
    except ValueError as error:
        raise NewsValidationError("invalid GDELT DATEADDED timestamp") from error
    return GdeltEventRow(
        global_event_id=fields[0].strip(),
        sql_date=fields[1].strip(),
        actor1_name=fields[6].strip(),
        actor2_name=fields[16].strip(),
        event_code=fields[26].strip(),
        event_base_code=fields[27].strip(),
        event_root_code=fields[28].strip(),
        quad_class=_integer(fields[29], field="QuadClass"),
        goldstein_scale=_number(fields[30], field="GoldsteinScale"),
        num_mentions=_integer(fields[31], field="NumMentions"),
        num_sources=_integer(fields[32], field="NumSources"),
        num_articles=_integer(fields[33], field="NumArticles"),
        average_tone=_number(fields[34], field="AvgTone"),
        date_added=added,
        source_url=fields[60].strip(),
    )


def parse_gdelt_v2_export(lines: Iterable[str]) -> tuple[GdeltEventRow, ...]:
    return tuple(parse_gdelt_v2_export_line(line.rstrip("\r\n")) for line in lines if line.strip())


def parse_gdelt_v1_export_line(line: str) -> GdeltV1EventRow:
    fields = next(csv.reader([line], delimiter="\t"))
    if len(fields) != GDELT_V1_EXPORT_COLUMNS:
        raise NewsValidationError(
            f"GDELT v1 export row must have {GDELT_V1_EXPORT_COLUMNS} columns"
        )
    try:
        date_added = datetime.strptime(fields[56], "%Y%m%d").date()
    except ValueError as error:
        raise NewsValidationError("invalid GDELT v1 DATEADDED date") from error
    return GdeltV1EventRow(
        global_event_id=fields[0].strip(),
        sql_date=fields[1].strip(),
        actor1_name=fields[6].strip(),
        actor2_name=fields[16].strip(),
        event_code=fields[26].strip(),
        event_base_code=fields[27].strip(),
        event_root_code=fields[28].strip(),
        quad_class=_integer(fields[29], field="QuadClass"),
        goldstein_scale=_number(fields[30], field="GoldsteinScale"),
        num_mentions=_integer(fields[31], field="NumMentions"),
        num_sources=_integer(fields[32], field="NumSources"),
        num_articles=_integer(fields[33], field="NumArticles"),
        average_tone=_number(fields[34], field="AvgTone"),
        date_added=date_added,
        source_url=fields[57].strip(),
    )


def _topics(row: GdeltEventRow | GdeltV1EventRow) -> tuple[str, ...]:
    topics: set[str] = set()
    if row.event_root_code in _CONFLICT_ROOT_CODES:
        topics.add("military_conflict")
    if row.event_root_code == "20":
        topics.add("terrorism")
    if row.event_base_code == "163":
        topics.add("sanctions")
    actor_text = f"{row.actor1_name} {row.actor2_name}".casefold()
    if any(term in actor_text for term in ("oil", "petroleum", "opec", "energy")):
        topics.add("oil_supply")
    if any(term in actor_text for term in ("ship", "port", "maritime", "freight")):
        topics.add("shipping_disruption")
    return tuple(sorted(topics))


def _mapped_tickers(
    row: GdeltEventRow | GdeltV1EventRow,
    aliases: Mapping[str, tuple[str, ...]] | TickerAliasMatcher,
) -> tuple[str, ...]:
    haystack = " ".join((row.actor1_name, row.actor2_name, row.source_url)).casefold()
    matcher = (
        aliases if isinstance(aliases, TickerAliasMatcher) else compile_ticker_aliases(aliases)
    )
    return matcher.match(haystack)


def _tone_probabilities(average_tone: float) -> tuple[float, float, float]:
    signed = float(np.tanh(np.clip(average_tone, -100.0, 100.0) / 10.0))
    strength = min(abs(signed), 0.95)
    positive = strength if signed > 0 else 0.0
    negative = strength if signed < 0 else 0.0
    neutral = 1.0 - strength
    return positive, neutral, negative


def gdelt_row_to_news_event(
    row: GdeltEventRow,
    *,
    ticker_aliases: Mapping[str, tuple[str, ...]] | TickerAliasMatcher,
    source_reliability: Mapping[str, float] | None = None,
) -> NewsEvent:
    """Convert a GDELT row without inventing a historical publication time."""
    source = urlparse(row.source_url).netloc.casefold() or "gdelt-unknown-source"
    reliability = float((source_reliability or {}).get(source, 0.5))
    positive, neutral, negative = _tone_probabilities(row.average_tone)
    conflict_weight = 1.0 if row.event_root_code in _CONFLICT_ROOT_CODES else 0.25
    severity = min(1.0, conflict_weight * abs(row.goldstein_scale) / 10.0)
    confidence = min(1.0, math.log1p(max(row.num_sources, 1)) / math.log(6.0))
    return NewsEvent(
        event_id=f"gdelt2:{row.global_event_id}:{row.date_added.strftime('%Y%m%d%H%M%S')}",
        cluster_id=f"gdelt2:{row.global_event_id}",
        source=source,
        first_seen_at=row.date_added,
        published_at=None,
        timestamp_quality="first_seen_only",
        tickers=_mapped_tickers(row, ticker_aliases),
        topics=_topics(row),
        positive_probability=positive,
        neutral_probability=neutral,
        negative_probability=negative,
        # Event novelty needs a trailing point-in-time reference distribution;
        # the raw adapter stays neutral rather than using future corpus counts.
        novelty=0.5,
        severity=severity,
        confidence=confidence,
        source_reliability=reliability,
        canonical_url_hash=(
            hashlib.sha256(row.source_url.encode("utf-8")).hexdigest() if row.source_url else ""
        ),
        license_class="gdelt_event_metadata",
    )


def gdelt_v1_row_to_news_event(
    row: GdeltV1EventRow,
    *,
    archive: GdeltV1DailyArchive,
    ticker_aliases: Mapping[str, tuple[str, ...]] | TickerAliasMatcher,
    source_reliability: Mapping[str, float] | None = None,
) -> NewsEvent:
    """Convert a daily row using archive availability, never its event date."""
    if row.date_added != archive.archive_date:
        raise NewsValidationError("GDELT v1 row DATEADDED does not match its daily archive")
    source = urlparse(row.source_url).netloc.casefold() or "gdelt-unknown-source"
    reliability = float((source_reliability or {}).get(source, 0.5))
    positive, neutral, negative = _tone_probabilities(row.average_tone)
    conflict_weight = 1.0 if row.event_root_code in _CONFLICT_ROOT_CODES else 0.25
    severity = min(1.0, conflict_weight * abs(row.goldstein_scale) / 10.0)
    confidence = min(1.0, math.log1p(max(row.num_sources, 1)) / math.log(6.0))
    return NewsEvent(
        event_id=f"gdelt1:{row.global_event_id}:{row.date_added:%Y%m%d}",
        cluster_id=f"gdelt1:{row.global_event_id}",
        source=source,
        first_seen_at=pd.Timestamp(archive.available_at),
        published_at=None,
        timestamp_quality="first_seen_only",
        tickers=_mapped_tickers(row, ticker_aliases),
        topics=_topics(row),
        positive_probability=positive,
        neutral_probability=neutral,
        negative_probability=negative,
        novelty=0.5,
        severity=severity,
        confidence=confidence,
        source_reliability=reliability,
        canonical_url_hash=(
            hashlib.sha256(row.source_url.encode("utf-8")).hexdigest() if row.source_url else ""
        ),
        license_class="gdelt_event_metadata",
    )


def aggregate_gdelt_v1_daily_lines(
    lines: Iterable[str],
    *,
    archive: GdeltV1DailyArchive,
    ticker_aliases: Mapping[str, tuple[str, ...]] | TickerAliasMatcher,
    source_reliability: Mapping[str, float] | None = None,
    maximum_invalid_fraction: float = 0.001,
) -> tuple[tuple[NewsEvent, ...], GdeltDailyAggregationStats]:
    """Compress one complete daily archive into bounded point-in-time events."""
    if not 0 <= maximum_invalid_fraction < 1:
        raise ValueError("maximum invalid fraction must be in [0, 1)")
    alias_matcher = (
        ticker_aliases
        if isinstance(ticker_aliases, TickerAliasMatcher)
        else compile_ticker_aliases(ticker_aliases)
    )
    groups: dict[str, list[NewsEvent]] = defaultdict(list)
    total_rows = 0
    retained_rows = 0
    invalid_rows = 0
    for line in lines:
        if not line.strip():
            continue
        total_rows += 1
        try:
            row = parse_gdelt_v1_export_line(line.rstrip("\r\n"))
            event = gdelt_v1_row_to_news_event(
                row,
                archive=archive,
                ticker_aliases=alias_matcher,
                source_reliability=source_reliability,
            )
        except NewsValidationError:
            invalid_rows += 1
            continue
        if not event.topics and not event.tickers:
            continue
        retained_rows += 1
        if event.tickers:
            for ticker in event.tickers:
                groups[f"ticker:{ticker}"].append(event)
        else:
            groups["market"].append(event)

    if total_rows == 0:
        raise NewsValidationError("GDELT daily archive contains no event rows")
    if invalid_rows / total_rows > maximum_invalid_fraction:
        raise NewsValidationError("GDELT daily archive exceeded the invalid-row guardrail")

    aggregated: list[NewsEvent] = []
    for scope, events in sorted(groups.items()):
        volume = float(len(events))
        weights = np.asarray([max(event.confidence, 1e-6) for event in events], dtype=np.float64)
        weights /= weights.sum()
        positive = float(np.dot(weights, [event.positive_probability for event in events]))
        neutral = float(np.dot(weights, [event.neutral_probability for event in events]))
        negative = max(0.0, 1.0 - positive - neutral)
        topics = tuple(sorted({topic for event in events for topic in event.topics}))
        ticker = scope.partition(":")[2] if scope.startswith("ticker:") else ""
        identity = f"gdelt1-daily:{archive.archive_date:%Y%m%d}:{scope}"
        aggregated.append(
            NewsEvent(
                event_id=identity,
                cluster_id=identity,
                source="gdelt-daily",
                first_seen_at=pd.Timestamp(archive.available_at),
                published_at=None,
                timestamp_quality="first_seen_only",
                tickers=(ticker,) if ticker else (),
                topics=topics,
                positive_probability=positive,
                neutral_probability=neutral,
                negative_probability=negative,
                novelty=0.5,
                severity=max(event.severity for event in events),
                confidence=float(np.mean([event.confidence for event in events])),
                source_reliability=float(np.mean([event.source_reliability for event in events])),
                license_class="gdelt_event_metadata_aggregated",
                volume=volume,
            )
        )
    return tuple(aggregated), GdeltDailyAggregationStats(
        archive_date=archive.archive_date.isoformat(),
        total_rows=total_rows,
        retained_rows=retained_rows,
        invalid_rows=invalid_rows,
        output_events=len(aggregated),
    )
