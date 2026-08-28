"""Live point-in-time news aggregation for certified news-fusion serving.

The provider turns live ticker headlines into the exact ordered news feature
vector demanded by a signed release contract. It is deterministic for a fixed
event stream and cutoff, strictly causal (an article participates only after
its conservative eligibility instant, always before the cutoff), and
fail-closed: features it cannot honestly reproduce from per-ticker headlines
raise :class:`NewsProviderUnavailable` instead of being fabricated.

Educational research only. The provider never trains on this data and never
persists raw article text.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import numpy as np

from news_features import (
    FINANCIAL_LEXICON,
    deduplicate_articles,
    filter_ticker_relevance,
    normalise_news_articles,
    score_news_articles,
)

logger = logging.getLogger(__name__)

# Frozen research schema (research/volatility_forecasting/news.py). A signed
# news-certified release names its required features in this vocabulary.
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

# Feature names this live-headline provider can honestly reproduce from a
# per-ticker article stream. Market-wide, topic-exposure, and macro-policy
# aggregates require a market-wide point-in-time event provider and are
# deliberately unsupported here: the provider fails closed on them.
LIVE_SUPPORTED_FEATURE_NAMES = frozenset(
    {
        "News_Ticker_Intensity_1H",
        "News_Ticker_Intensity_1D",
        "News_Ticker_Intensity_3D",
        "News_Ticker_Negative_1D",
        "News_Ticker_Absolute_Sentiment_1D",
        "News_Ticker_Novelty_3D",
        "News_Ticker_Severity_3D",
        "News_Ticker_Source_Diversity_3D",
        "News_Ticker_Missing_1D",
    }
)

# Decay windows mirrored from the frozen research aggregation
# (research/volatility_forecasting/news.py): (maximum age, half-life) in hours.
_WINDOW_1H = (1.0, 1.0)
_WINDOW_1D = (24.0, 12.0)
_WINDOW_3D = (72.0, 36.0)

# Deterministic weighting constants for the live provider. Unlike the research
# pipeline, live headlines carry no provider confidence or source-reliability
# score, so both are fixed at 1.0 and documented as part of this provider's
# feature definition.
_LIVE_CONFIDENCE = 1.0
_LIVE_RELIABILITY = 1.0

_LOG_TWO = float(np.log(2.0))


class NewsProviderUnavailable(RuntimeError):
    """The live news provider cannot produce the certified feature vector."""


@dataclass(frozen=True)
class _LiveEvent:
    """One deduplicated, scored, causally eligible headline event."""

    eligible_at: Any  # timezone-aware pandas.Timestamp
    source: str
    sentiment: float
    negative: float
    absolute_sentiment: float
    novelty: float
    severity: float

    @property
    def low_quality(self) -> bool:
        return False


@dataclass(frozen=True)
class NewsFeatureVector:
    """Schema-exact news vector plus the telemetry needed for evidence."""

    values: np.ndarray
    feature_names: tuple[str, ...]
    cutoff_at: str
    eligible_article_count: int


def _to_utc(value: Any) -> Any:
    """Return a timezone-aware UTC timestamp; naive values are read as UTC."""
    import pandas as pd

    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _lexicon_severity(text: str) -> float:
    """Map financial-lexicon phrase magnitude to [0, 1] event severity."""
    adjustment = sum(
        value for phrase, value in FINANCIAL_LEXICON.items() if phrase in text.casefold()
    )
    return float(min(1.0, abs(adjustment) / 3.5))


def _token_set(text: str) -> frozenset[str]:
    return frozenset(token for token in re.findall(r"\w+", text.casefold()) if len(token) > 2)


def _novelty(title: str, summary: str, earlier_tokens: list[frozenset[str]]) -> float:
    """1 - max Jaccard similarity against earlier eligible events."""
    tokens = _token_set(f"{title} {summary}")
    if not tokens or not earlier_tokens:
        return 1.0
    maximum = max(
        len(tokens & other) / len(tokens | other) if tokens | other else 0.0
        for other in earlier_tokens
    )
    return float(np.clip(1.0 - maximum, 0.0, 1.0))


def build_live_events(
    scored_articles: Sequence[dict[str, Any]],
    *,
    cutoff_at: Any,
    availability_delay: timedelta,
) -> list[_LiveEvent]:
    """Convert scored articles into causally eligible events, ordered by time.

    An event becomes eligible at ``max(published_at, received_at) +
    availability_delay``; a date-only publication timestamp (no clock time:
    exactly midnight UTC with no distinct receipt time) cannot enter a
    same-day close and becomes eligible at the next UTC day boundary. Events
    are excluded unless ``eligible_at < cutoff_at``.
    """
    import pandas as pd

    cutoff = _to_utc(cutoff_at)
    events: list[_LiveEvent] = []
    earlier_tokens: list[frozenset[str]] = []
    for article in scored_articles:
        published = _to_utc(article.get("published_at"))
        received = _to_utc(article.get("received_at"))
        if published is None and received is None:
            continue
        if published is None:
            eligible_at = received + availability_delay
        elif published == published.normalize() and (received is None or received == published):
            # A date without a trustworthy clock time cannot enter a same-day
            # close; it becomes eligible at the next UTC day boundary.
            eligible_at = published.normalize() + pd.Timedelta(days=1)
        else:
            eligible_at = max(published, received or published) + availability_delay
        if not eligible_at < cutoff:
            continue
        sentiment = float(np.clip(float(article.get("sentiment_score") or 0.0), -1.0, 1.0))
        title = str(article.get("title") or "")
        summary = str(article.get("summary") or "")
        events.append(
            _LiveEvent(
                eligible_at=eligible_at,
                source=str(article.get("publisher") or article.get("provider") or "unknown"),
                sentiment=sentiment,
                negative=float(max(0.0, -sentiment)),
                absolute_sentiment=float(abs(sentiment)),
                novelty=_novelty(title, summary, earlier_tokens),
                severity=_lexicon_severity(f"{title} {summary}"),
            )
        )
        earlier_tokens.append(_token_set(f"{title} {summary}"))
    events.sort(key=lambda event: event.eligible_at)
    return events


def _decay_weighted_sum(
    events: Sequence[_LiveEvent],
    cutoff_at: Any,
    values: np.typing.ArrayLike,
    *,
    maximum_age_hours: float,
    half_life_hours: float,
) -> float:
    """Confidence × reliability × value, exponentially decayed by event age."""
    import pandas as pd

    if len(events) == 0:
        return 0.0
    cutoff_ns = pd.Timestamp(_to_utc(cutoff_at)).value
    hour_ns = float(pd.Timedelta(hours=1).value)
    total = 0.0
    for event, value in zip(events, values, strict=True):
        age_hours = (cutoff_ns - pd.Timestamp(event.eligible_at).value) / hour_ns
        if age_hours < 0.0 or age_hours > maximum_age_hours:
            continue
        decay = float(np.exp(-_LOG_TWO * age_hours / half_life_hours))
        total += _LIVE_CONFIDENCE * _LIVE_RELIABILITY * decay * float(value)
    return float(total)


def aggregate_live_features(
    events: Sequence[_LiveEvent],
    *,
    cutoff_at: Any,
    feature_names: Sequence[str],
) -> np.ndarray:
    """Compute the requested schema-exact feature vector from eligible events.

    Semantics mirror the frozen research aggregation
    (research/volatility_forecasting/news.py): confidence- and
    reliability-weighted values with exponential age decay inside each
    feature's (maximum age, half-life) window.
    """

    cutoff = _to_utc(cutoff_at)
    negatives = np.asarray([event.negative for event in events], dtype=np.float64)
    absolutes = np.asarray([event.absolute_sentiment for event in events], dtype=np.float64)
    novelties = np.asarray([event.novelty for event in events], dtype=np.float64)
    severities = np.asarray([event.severity for event in events], dtype=np.float64)
    sources_3d = {
        event.source
        for event in events
        if 0.0 <= (cutoff - event.eligible_at).total_seconds() / 3600.0 <= _WINDOW_3D[0]
    }
    within_1d = [
        event
        for event in events
        if 0.0 <= (cutoff - event.eligible_at).total_seconds() / 3600.0 <= _WINDOW_1D[0]
    ]
    computed = {
        "News_Ticker_Intensity_1H": _decay_weighted_sum(
            events,
            cutoff,
            np.ones(len(events)),
            maximum_age_hours=_WINDOW_1H[0],
            half_life_hours=_WINDOW_1H[1],
        ),
        "News_Ticker_Intensity_1D": _decay_weighted_sum(
            events,
            cutoff,
            np.ones(len(events)),
            maximum_age_hours=_WINDOW_1D[0],
            half_life_hours=_WINDOW_1D[1],
        ),
        "News_Ticker_Intensity_3D": _decay_weighted_sum(
            events,
            cutoff,
            np.ones(len(events)),
            maximum_age_hours=_WINDOW_3D[0],
            half_life_hours=_WINDOW_3D[1],
        ),
        "News_Ticker_Negative_1D": _decay_weighted_sum(
            events,
            cutoff,
            negatives,
            maximum_age_hours=_WINDOW_1D[0],
            half_life_hours=_WINDOW_1D[1],
        ),
        "News_Ticker_Absolute_Sentiment_1D": _decay_weighted_sum(
            events,
            cutoff,
            absolutes,
            maximum_age_hours=_WINDOW_1D[0],
            half_life_hours=_WINDOW_1D[1],
        ),
        "News_Ticker_Novelty_3D": _decay_weighted_sum(
            events,
            cutoff,
            novelties,
            maximum_age_hours=_WINDOW_3D[0],
            half_life_hours=_WINDOW_3D[1],
        ),
        "News_Ticker_Severity_3D": _decay_weighted_sum(
            events,
            cutoff,
            severities,
            maximum_age_hours=_WINDOW_3D[0],
            half_life_hours=_WINDOW_3D[1],
        ),
        "News_Ticker_Source_Diversity_3D": float(len(sources_3d)),
        "News_Ticker_Missing_1D": 0.0 if within_1d else 1.0,
    }
    values = []
    for name in feature_names:
        if name not in computed:
            raise NewsProviderUnavailable(f"unsupported live news feature: {name}")
        values.append(computed[name])
    vector = np.asarray(values, dtype=np.float32)
    if not np.isfinite(vector).all():
        raise NewsProviderUnavailable("aggregated news features are not finite")
    return vector


def _fetch_live_articles(ticker: str) -> list[dict[str, Any]]:
    """Fetch current ticker headlines; transport failures fail closed."""
    try:
        import yfinance as yf

        return list(yf.Ticker(ticker).news or [])
    except Exception as error:
        raise NewsProviderUnavailable(f"live news ingestion failed: {error}") from error


class NewsAggregationProvider:
    """Deterministic causal news-feature provider for news-certified serving."""

    def __init__(
        self,
        *,
        event_source: Callable[[str], list[dict[str, Any]]] | None = None,
        availability_delay: timedelta = timedelta(minutes=15),
        scorer: Any | None = None,
    ) -> None:
        self._event_source = event_source or _fetch_live_articles
        self._availability_delay = availability_delay
        self._scorer = scorer

    def features_for(
        self,
        ticker: str,
        *,
        cutoff_at: Any,
        feature_names: Sequence[str],
    ) -> NewsFeatureVector:
        """Return the schema-exact news vector for one ticker at one cutoff."""
        names = tuple(feature_names)
        if not names:
            raise NewsProviderUnavailable("the release demands no news features")
        unsupported = sorted(set(names) - LIVE_SUPPORTED_FEATURE_NAMES)
        if unsupported:
            raise NewsProviderUnavailable(
                "these certified news features require a market-wide point-in-time "
                f"provider and cannot be reproduced from live headlines: {unsupported}"
            )
        try:
            raw = self._event_source(ticker.upper())
        except NewsProviderUnavailable:
            raise
        except Exception as error:
            raise NewsProviderUnavailable(f"live news ingestion failed: {error}") from error
        articles = filter_ticker_relevance(
            deduplicate_articles(normalise_news_articles(raw)),
            ticker.upper(),
        )
        scored = score_news_articles(articles, scorer=self._scorer)
        events = build_live_events(
            scored,
            cutoff_at=cutoff_at,
            availability_delay=self._availability_delay,
        )
        values = aggregate_live_features(events, cutoff_at=cutoff_at, feature_names=names)
        return NewsFeatureVector(
            values=values,
            feature_names=names,
            cutoff_at=_to_utc(cutoff_at).isoformat(),
            eligible_article_count=len(events),
        )


_PROVIDER: NewsAggregationProvider | None = None


def get_news_provider() -> NewsAggregationProvider:
    """Process-wide provider instance (deterministic, network-fail-closed)."""
    global _PROVIDER
    if _PROVIDER is None:
        _PROVIDER = NewsAggregationProvider()
    return _PROVIDER


def reset_news_provider() -> None:
    """Test and operations hook to force provider re-creation."""
    global _PROVIDER
    _PROVIDER = None
