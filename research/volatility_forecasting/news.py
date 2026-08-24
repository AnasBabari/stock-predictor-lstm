"""Point-in-time financial-news contracts and causal feature aggregation.

The module has no provider client and never scrapes during evaluation. It
consumes immutable, license-reviewed event records with first-seen timestamps.
Every aggregation is reconstructed as-of an explicit UTC forecast cutoff.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd

NEWS_SCHEMA_VERSION = "point_in_time_news_v2"
NEWS_FEATURE_SCHEMA_VERSION = "point_in_time_news_features_v2"
TimestampQuality = Literal["precise", "first_seen_only", "date_only", "unknown"]

CONFLICT_TOPICS = frozenset({"military_conflict", "war", "sanctions", "terrorism"})
COMMODITY_TOPICS = frozenset(
    {"oil_supply", "commodity_disruption", "shipping_disruption", "energy_policy"}
)
MACRO_POLICY_TOPICS = frozenset({"monetary_policy", "inflation", "regulation", "fiscal_policy"})

NEWS_FEATURE_NAMES_V2 = (
    "News_Ticker_Intensity_1H",
    "News_Ticker_Intensity_1D",
    "News_Ticker_Intensity_3D",
    "News_Ticker_Negative_1D",
    "News_Ticker_Absolute_Sentiment_1D",
    "News_Ticker_Novelty_3D",
    "News_Ticker_Severity_3D",
    "News_Ticker_Source_Diversity_3D",
    "News_Exposure_Intensity_1D",
    "News_Exposure_Conflict_Severity_3D",
    "News_Exposure_Commodity_Severity_3D",
    "News_Market_Intensity_1D",
    "News_Market_Negative_1D",
    "News_Market_Conflict_Severity_3D",
    "News_Market_Commodity_Severity_3D",
    "News_Macro_Policy_Severity_3D",
    "News_Low_Timestamp_Quality_Fraction_3D",
    "News_Ticker_Missing_1D",
)


class NewsValidationError(ValueError):
    """A news record cannot participate in point-in-time evaluation."""


class NewsLicenseNotAcknowledged(RuntimeError):
    """The caller has not acknowledged the news provider's data terms."""


def _utc(value: pd.Timestamp | str | None, *, field: str) -> pd.Timestamp | None:
    if value is None:
        return None
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        raise NewsValidationError(f"{field} must be timezone-aware")
    return parsed.tz_convert("UTC")


@dataclass(frozen=True)
class NewsEvent:
    """One immutable event mention as it was available to the pipeline."""

    event_id: str
    cluster_id: str
    source: str
    first_seen_at: pd.Timestamp
    published_at: pd.Timestamp | None
    timestamp_quality: TimestampQuality
    tickers: tuple[str, ...]
    topics: tuple[str, ...]
    positive_probability: float
    neutral_probability: float
    negative_probability: float
    novelty: float
    severity: float
    confidence: float
    source_reliability: float
    headline_hash: str = ""
    canonical_url_hash: str = ""
    language: str = "en"
    license_class: str = "research_only"
    volume: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "first_seen_at", _utc(self.first_seen_at, field="first_seen_at"))
        object.__setattr__(
            self,
            "published_at",
            _utc(self.published_at, field="published_at"),
        )
        object.__setattr__(self, "tickers", tuple(sorted({t.upper() for t in self.tickers})))
        object.__setattr__(self, "topics", tuple(sorted({t.lower() for t in self.topics})))
        if not self.event_id or not self.cluster_id or not self.source:
            raise NewsValidationError("event_id, cluster_id, and source are required")
        if self.timestamp_quality not in {"precise", "first_seen_only", "date_only", "unknown"}:
            raise NewsValidationError("invalid timestamp_quality")
        probabilities = (
            self.positive_probability,
            self.neutral_probability,
            self.negative_probability,
        )
        if any(not np.isfinite(value) or not 0 <= value <= 1 for value in probabilities):
            raise NewsValidationError("sentiment probabilities must be finite and in [0, 1]")
        if not np.isclose(sum(probabilities), 1.0, atol=1e-6):
            raise NewsValidationError("sentiment probabilities must sum to one")
        for field_name in ("novelty", "severity", "confidence", "source_reliability"):
            value = float(getattr(self, field_name))
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise NewsValidationError(f"{field_name} must be finite and in [0, 1]")
        if not np.isfinite(self.volume) or self.volume <= 0:
            raise NewsValidationError("volume must be finite and positive")
        if self.timestamp_quality == "precise" and self.published_at is None:
            raise NewsValidationError("precise events require published_at")

    @property
    def eligible_at(self) -> pd.Timestamp | None:
        """Conservative first instant at which this record may be used."""
        if self.timestamp_quality == "unknown":
            return None
        if self.published_at is None:
            eligible = self.first_seen_at
        else:
            # A historical publication timestamp does not prove our data feed
            # had the article then. max(...) prevents revision/ingestion hindsight.
            eligible = max(self.published_at, self.first_seen_at)
        if self.timestamp_quality == "date_only":
            # A date without a trustworthy clock time cannot enter a same-day
            # close. It becomes eligible at the next UTC day boundary.
            return eligible.normalize() + pd.Timedelta(days=1)
        return eligible

    @property
    def sentiment(self) -> float:
        return float(self.positive_probability - self.negative_probability)


@dataclass(frozen=True)
class NewsOrigin:
    ticker: str
    cutoff_at: pd.Timestamp

    def __post_init__(self) -> None:
        object.__setattr__(self, "ticker", self.ticker.upper())
        object.__setattr__(self, "cutoff_at", _utc(self.cutoff_at, field="cutoff_at"))
        if not self.ticker:
            raise NewsValidationError("origin ticker is required")


@dataclass(frozen=True)
class NewsFeatureMatrix:
    values: np.ndarray
    tickers: np.ndarray
    cutoffs: np.ndarray
    feature_names: tuple[str, ...] = NEWS_FEATURE_NAMES_V2

    def __post_init__(self) -> None:
        if self.values.shape != (len(self.tickers), len(self.feature_names)):
            raise NewsValidationError("news feature matrix shape does not match origins")
        if len(self.cutoffs) != len(self.tickers):
            raise NewsValidationError("news cutoff identities do not match rows")
        if not np.isfinite(self.values).all():
            raise NewsValidationError("news features must be finite")


def build_news_snapshot(
    events: Sequence[NewsEvent],
    *,
    license_acknowledged: bool,
    provider: str,
) -> dict[str, object]:
    """Create a deterministic manifest without storing raw article text."""
    if not license_acknowledged:
        raise NewsLicenseNotAcknowledged(
            "News provider terms must be reviewed before snapshot construction."
        )
    if not provider.strip():
        raise NewsValidationError("provider is required")
    canonical_rows: list[dict[str, object]] = []
    for event in sorted(events, key=lambda item: item.event_id):
        row = asdict(event)
        row["first_seen_at"] = event.first_seen_at.isoformat()
        row["published_at"] = event.published_at.isoformat() if event.published_at else None
        canonical_rows.append(row)
    payload = json.dumps(canonical_rows, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return {
        "schema_version": NEWS_SCHEMA_VERSION,
        "snapshot_id": f"news-{digest[:16]}",
        "digest": f"sha256:{digest}",
        "provider": provider,
        "event_count": len(events),
        "license_acknowledged": True,
    }


def _deduplicated_timeline(events: Sequence[NewsEvent]) -> list[NewsEvent]:
    by_cluster: dict[str, list[NewsEvent]] = defaultdict(list)
    for event in events:
        eligible = event.eligible_at
        if eligible is not None:
            by_cluster[event.cluster_id].append(event)

    selected: list[NewsEvent] = []
    for mentions in by_cluster.values():
        # Earliest eligible mention wins. Metadata tie-breakers use only
        # records already visible at the cutoff.
        selected.append(
            min(
                mentions,
                key=lambda item: (
                    item.eligible_at,
                    -item.confidence,
                    -item.source_reliability,
                    item.event_id,
                ),
            )
        )
    return sorted(selected, key=lambda item: (item.eligible_at, item.cluster_id))


def _deduplicated_as_of(events: Sequence[NewsEvent], cutoff: pd.Timestamp) -> list[NewsEvent]:
    return [event for event in _deduplicated_timeline(events) if event.eligible_at <= cutoff]


def _decay(age_hours: float, half_life_hours: float) -> float:
    return float(np.exp(-np.log(2.0) * max(age_hours, 0.0) / half_life_hours))


def _count(_event: NewsEvent) -> float:
    # Preserve one raw event as unit intensity while compressing large daily
    # aggregates onto a stable logarithmic scale.
    return float(np.log1p(_event.volume) / np.log(2.0))


def _negative(event: NewsEvent) -> float:
    return event.negative_probability


def _absolute_sentiment(event: NewsEvent) -> float:
    return abs(event.sentiment)


def _novelty(event: NewsEvent) -> float:
    return event.novelty


def _severity(event: NewsEvent) -> float:
    return event.severity


def _conflict_severity(event: NewsEvent) -> float:
    return event.severity if CONFLICT_TOPICS.intersection(event.topics) else 0.0


def _commodity_severity(event: NewsEvent) -> float:
    return event.severity if COMMODITY_TOPICS.intersection(event.topics) else 0.0


def _macro_severity(event: NewsEvent) -> float:
    return event.severity if MACRO_POLICY_TOPICS.intersection(event.topics) else 0.0


def _weighted_sum(
    events: Sequence[tuple[NewsEvent, float]],
    cutoff: pd.Timestamp,
    *,
    maximum_age_hours: float,
    half_life_hours: float,
    value,
) -> float:
    total = 0.0
    for event, exposure_weight in events:
        eligible = event.eligible_at
        if eligible is None:
            continue
        age_hours = float((cutoff - eligible) / pd.Timedelta(hours=1))
        if 0 <= age_hours <= maximum_age_hours:
            total += (
                _decay(age_hours, half_life_hours)
                * event.confidence
                * event.source_reliability
                * exposure_weight
                * float(value(event))
            )
    return float(total)


def aggregate_news_features(
    events: Sequence[NewsEvent],
    origins: Sequence[NewsOrigin],
    *,
    exposure_map: Mapping[str, Mapping[str, float]] | None = None,
) -> NewsFeatureMatrix:
    """Aggregate immutable event records at explicit point-in-time origins."""
    exposures = {
        ticker.upper(): {topic.lower(): float(weight) for topic, weight in weights.items()}
        for ticker, weights in (exposure_map or {}).items()
    }
    timeline = _deduplicated_timeline(events)
    eligible_ns = np.asarray([event.eligible_at.value for event in timeline], dtype=np.int64)
    recent_by_cutoff: dict[int, list[NewsEvent]] = {}
    rows: list[list[float]] = []

    for origin in origins:
        cutoff_ns = origin.cutoff_at.value
        recent = recent_by_cutoff.get(cutoff_ns)
        if recent is None:
            lower_ns = (origin.cutoff_at - pd.Timedelta(days=7)).value
            left = int(np.searchsorted(eligible_ns, lower_ns, side="left"))
            right = int(np.searchsorted(eligible_ns, cutoff_ns, side="right"))
            recent = timeline[left:right]
            recent_by_cutoff[cutoff_ns] = recent
        ticker_events = [(event, 1.0) for event in recent if origin.ticker in event.tickers]
        ticker_exposures = exposures.get(origin.ticker, {})
        exposure_events: list[tuple[NewsEvent, float]] = []
        for event in recent:
            weight = max((ticker_exposures.get(topic, 0.0) for topic in event.topics), default=0.0)
            if weight > 0:
                exposure_events.append((event, min(weight, 1.0)))
        market_events = [(event, 1.0) for event in recent]

        ticker_1d = _weighted_sum(
            ticker_events,
            origin.cutoff_at,
            maximum_age_hours=24,
            half_life_hours=12,
            value=_count,
        )
        sources_3d = {
            event.source
            for event, _ in ticker_events
            if event.eligible_at is not None
            and origin.cutoff_at - event.eligible_at <= pd.Timedelta(days=3)
        }
        eligible_3d = [
            event
            for event, _ in market_events
            if event.eligible_at is not None
            and origin.cutoff_at - event.eligible_at <= pd.Timedelta(days=3)
        ]
        low_quality_fraction = (
            sum(event.timestamp_quality != "precise" for event in eligible_3d) / len(eligible_3d)
            if eligible_3d
            else 0.0
        )

        rows.append(
            [
                _weighted_sum(
                    ticker_events,
                    origin.cutoff_at,
                    maximum_age_hours=1,
                    half_life_hours=1,
                    value=_count,
                ),
                ticker_1d,
                _weighted_sum(
                    ticker_events,
                    origin.cutoff_at,
                    maximum_age_hours=72,
                    half_life_hours=36,
                    value=_count,
                ),
                _weighted_sum(
                    ticker_events,
                    origin.cutoff_at,
                    maximum_age_hours=24,
                    half_life_hours=12,
                    value=_negative,
                ),
                _weighted_sum(
                    ticker_events,
                    origin.cutoff_at,
                    maximum_age_hours=24,
                    half_life_hours=12,
                    value=_absolute_sentiment,
                ),
                _weighted_sum(
                    ticker_events,
                    origin.cutoff_at,
                    maximum_age_hours=72,
                    half_life_hours=36,
                    value=_novelty,
                ),
                _weighted_sum(
                    ticker_events,
                    origin.cutoff_at,
                    maximum_age_hours=72,
                    half_life_hours=36,
                    value=_severity,
                ),
                float(len(sources_3d)),
                _weighted_sum(
                    exposure_events,
                    origin.cutoff_at,
                    maximum_age_hours=24,
                    half_life_hours=12,
                    value=_count,
                ),
                _weighted_sum(
                    exposure_events,
                    origin.cutoff_at,
                    maximum_age_hours=72,
                    half_life_hours=36,
                    value=_conflict_severity,
                ),
                _weighted_sum(
                    exposure_events,
                    origin.cutoff_at,
                    maximum_age_hours=72,
                    half_life_hours=36,
                    value=_commodity_severity,
                ),
                _weighted_sum(
                    market_events,
                    origin.cutoff_at,
                    maximum_age_hours=24,
                    half_life_hours=12,
                    value=_count,
                ),
                _weighted_sum(
                    market_events,
                    origin.cutoff_at,
                    maximum_age_hours=24,
                    half_life_hours=12,
                    value=_negative,
                ),
                _weighted_sum(
                    market_events,
                    origin.cutoff_at,
                    maximum_age_hours=72,
                    half_life_hours=36,
                    value=_conflict_severity,
                ),
                _weighted_sum(
                    market_events,
                    origin.cutoff_at,
                    maximum_age_hours=72,
                    half_life_hours=36,
                    value=_commodity_severity,
                ),
                _weighted_sum(
                    market_events,
                    origin.cutoff_at,
                    maximum_age_hours=72,
                    half_life_hours=36,
                    value=_macro_severity,
                ),
                float(low_quality_fraction),
                float(ticker_1d == 0.0),
            ]
        )

    return NewsFeatureMatrix(
        values=np.asarray(rows, dtype=np.float32),
        tickers=np.asarray([origin.ticker for origin in origins], dtype=str),
        cutoffs=np.asarray(
            [origin.cutoff_at.to_datetime64() for origin in origins],
            dtype="datetime64[ns]",
        ),
    )
