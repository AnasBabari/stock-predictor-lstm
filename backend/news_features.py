"""Point-in-time-safe news archive, scoring, and feature utilities.

Raw licensed news is intentionally an input, never an application artifact.
Historical callers must supply publication *and* receipt times; live yfinance
news is descriptive context only and cannot become a training feature.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime, time
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import pandas as pd
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

ARCHIVE_REQUIRED_FIELDS = {
    "provider",
    "ticker",
    "title",
    "published_at_utc",
    "received_at_utc",
}
NEWS_FEATURE_COLUMNS = [
    "News_Sentiment",
    "News_Article_Count",
    "News_Sentiment_Confidence",
    "News_Positive_Share",
    "News_Negative_Share",
    "News_Source_Diversity",
    "News_Novelty",
    "News_Missing_Indicator",
    "News_Coverage_Quality",
]
FINANCIAL_LEXICON = {
    "guidance cut": -2.0,
    "missed estimates": -1.5,
    "beat expectations": 2.0,
    "upgrade": 1.5,
    "downgrade": -1.5,
    "bullish": 2.0,
    "bearish": -2.0,
    "profit warning": -2.0,
    "dividend hike": 1.5,
    "dividend cut": -1.5,
    "record high": 1.5,
    "record low": -1.5,
    "layoffs": -1.5,
    "bankruptcy": -3.0,
    "lawsuit": -1.5,
    "merger": 1.0,
    "acquisition": 1.0,
}
EVENT_KEYWORDS = {
    "earnings": ("earnings", "beat estimates", "missed estimates"),
    "guidance": ("guidance", "outlook"),
    "analyst_rating": ("upgrade", "downgrade"),
    "dividend": ("dividend",),
    "merger_acquisition": ("merger", "acquisition", "buyout"),
    "legal_regulatory": ("lawsuit", "investigation", "regulatory"),
    "product": ("launch", "product announcement"),
    "management_change": ("ceo", "resigns"),
    "financing": ("offering", "financing"),
    "restructuring": ("layoffs", "restructuring"),
}
_analyzer: SentimentIntensityAnalyzer | None = None


def _sentiment_analyzer() -> SentimentIntensityAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentIntensityAnalyzer()
        _analyzer.lexicon.update(FINANCIAL_LEXICON)
    return _analyzer


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    try:
        stamp = (
            pd.Timestamp(float(value), unit="s", tz="UTC")
            if isinstance(value, (int, float))
            else pd.Timestamp(value)
        )
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize(UTC)
        return stamp.to_pydatetime().astimezone(UTC)
    except (TypeError, ValueError, OverflowError):
        return None


def _read_path(article: dict[str, Any], *paths: tuple[str, ...]) -> Any:
    for path in paths:
        value: Any = article
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if value not in (None, ""):
            return value
    return None


def _normalised_url(value: Any) -> str | None:
    if not value:
        return None
    parsed = urlsplit(str(value).strip())
    return urlunsplit(
        (parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip("/"), "", "")
    )


def _title_key(title: str) -> str:
    return re.sub(r"\W+", " ", title.casefold()).strip()


def normalise_news_articles(raw_articles: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Normalise provider-neutral archive records and yfinance's two schemas."""
    result: list[dict[str, Any]] = []
    for raw in raw_articles or []:
        if not isinstance(raw, dict):
            continue
        title = str(_read_path(raw, ("title",), ("content", "title")) or "").strip()
        summary = str(
            _read_path(
                raw,
                ("summary",),
                ("description",),
                ("content", "summary"),
                ("content", "description"),
            )
            or ""
        ).strip()
        if not (title or summary):
            continue
        published = _parse_timestamp(
            _read_path(
                raw,
                ("published_at_utc",),
                ("providerPublishTime",),
                ("pubDate",),
                ("content", "pubDate"),
                ("content", "displayTime"),
            )
        )
        received = _parse_timestamp(_read_path(raw, ("received_at_utc",))) or published
        link = _normalised_url(
            _read_path(
                raw, ("canonical_url",), ("link",), ("content", "canonicalUrl"), ("content", "url")
            )
        )
        result.append(
            {
                "article_id": str(raw.get("article_id") or raw.get("provider_article_id") or "")
                or None,
                "provider": str(raw.get("provider") or "yfinance"),
                "ticker": str(raw.get("ticker") or "").upper() or None,
                "company_name": raw.get("company_name"),
                "title": title,
                "summary": summary,
                "published_at": published,
                "received_at": received,
                "publisher": _read_path(
                    raw, ("publisher",), ("content", "provider", "displayName")
                ),
                "link": link,
                "language": raw.get("language"),
            }
        )
    return result


def deduplicate_articles(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministically retain one copy of syndicated or repeated records."""
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for article in sorted(
        articles,
        key=lambda a: (
            str(a.get("published_at") or ""),
            str(a.get("article_id") or ""),
            a["title"],
        ),
    ):
        key = (
            ("id", str(article["article_id"]))
            if article.get("article_id")
            else ("url", str(article["link"]))
            if article.get("link")
            else ("title", _title_key(article["title"]))
        )
        if key not in seen:
            seen.add(key)
            unique.append(article)
    return unique


def filter_ticker_relevance(
    articles: list[dict[str, Any]], ticker: str, company_aliases: tuple[str, ...] = ()
) -> list[dict[str, Any]]:
    """Keep explicitly tagged records or unambiguous ticker/company references."""
    symbol = ticker.upper()
    aliases = [re.escape(alias.casefold()) for alias in company_aliases if len(alias.strip()) > 2]

    symbol_pattern = re.compile(rf"(?<![A-Z0-9]){re.escape(symbol)}(?![A-Z0-9])", re.IGNORECASE)
    alias_pattern = (
        re.compile(r"(?:" + "|".join(aliases) + r")", re.IGNORECASE) if aliases else None
    )

    kept = []
    for item in articles:
        text = f"{item['title']} {item['summary']}"
        tagged = item.get("ticker") == symbol

        symbol_match = bool(symbol_pattern.search(text)) and len(symbol) >= 2
        alias_match = bool(alias_pattern.search(text)) if alias_pattern else False

        if tagged or symbol_match or alias_match:
            kept.append({**item, "relevance": 1.0 if tagged or alias_match else 0.7})
    return kept


@dataclass(frozen=True)
class SentimentResult:
    compound: float
    positive: float
    neutral: float
    negative: float
    scorer: str
    revision: str | None = None


class SentimentScorer(Protocol):
    def score(self, article: dict[str, Any]) -> SentimentResult: ...


class VaderFinancialScorer:
    name = "vader_financial"

    def score(self, article: dict[str, Any]) -> SentimentResult:
        text = " ".join(part for part in (article["title"], article["summary"]) if part)
        values = _sentiment_analyzer().polarity_scores(text)
        adjustment = sum(
            value for phrase, value in FINANCIAL_LEXICON.items() if phrase in text.casefold()
        )
        compound = float(np.clip(values["compound"] + adjustment * 0.1, -1, 1))
        return SentimentResult(
            compound, float(values["pos"]), float(values["neu"]), float(values["neg"]), self.name
        )


class TransformerSentimentScorer:
    """Optional pinned adapter. It fails safely when transformers is not installed."""

    def __init__(self, model_id: str, revision: str) -> None:
        if not model_id or not revision:
            raise ValueError("Transformer scorer requires a pinned model identifier and revision.")
        self.model_id, self.revision = model_id, revision
        self._pipeline: Any | None = None

    def score(self, article: dict[str, Any]) -> SentimentResult:
        try:
            if self._pipeline is None:
                from transformers import pipeline  # type: ignore[import-not-found]

                self._pipeline = pipeline(
                    "text-classification", model=self.model_id, revision=self.revision, top_k=None
                )
            labels = self._pipeline(
                " ".join((article["title"], article["summary"])), truncation=True
            )[0]
            probs = {str(row["label"]).lower(): float(row["score"]) for row in labels}
            positive, negative = probs.get("positive", 0.0), probs.get("negative", 0.0)
            return SentimentResult(
                positive - negative,
                positive,
                probs.get("neutral", 0.0),
                negative,
                "transformer",
                self.revision,
            )
        except Exception as exc:
            raise RuntimeError("transformer sentiment scorer unavailable") from exc


def score_news_articles(
    articles: list[dict[str, Any]], scorer: SentimentScorer | None = None
) -> list[dict[str, Any]]:
    scorer = scorer or VaderFinancialScorer()
    output = []
    for article in articles:
        result = scorer.score(article)
        output.append(
            {
                **article,
                "sentiment_score": result.compound,
                "sentiment_positive": result.positive,
                "sentiment_neutral": result.neutral,
                "sentiment_negative": result.negative,
                "method": result.scorer,
                "scorer_revision": result.revision,
                "event_categories": [
                    name
                    for name, words in EVENT_KEYWORDS.items()
                    if any(
                        word in f"{article['title']} {article['summary']}".casefold()
                        for word in words
                    )
                ],
            }
        )
    return output


def align_article_to_session(
    article: dict[str, Any],
    sessions: pd.DatetimeIndex,
    *,
    close_time: time = time(16),
    crypto: bool = False,
) -> pd.Timestamp | None:
    """Map an article to the first session after it was available, never before."""
    timestamps = [
        value
        for value in (article.get("published_at"), article.get("received_at"))
        if value is not None
    ]
    available = max(timestamps) if timestamps else None
    if available is None:
        return None
    index = pd.DatetimeIndex(sessions)
    index = index.tz_localize(UTC) if index.tz is None else index.tz_convert(UTC)
    if crypto:
        return (
            index[index >= pd.Timestamp(available)].min()
            if any(index >= pd.Timestamp(available))
            else None
        )
    day = pd.Timestamp(available).normalize()
    close = day + pd.Timedelta(hours=close_time.hour, minutes=close_time.minute)
    eligible = index[index > day] if pd.Timestamp(available) >= close else index[index >= day]
    return eligible.min() if len(eligible) else None


def build_daily_sentiment_features(
    articles: list[dict[str, Any]], sessions: pd.DatetimeIndex, *, half_life_days: float = 7.0
) -> pd.DataFrame:
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive.")
    index = pd.DatetimeIndex(sessions)
    index = index.tz_localize(UTC) if index.tz is None else index.tz_convert(UTC)
    result = pd.DataFrame(0.0, index=index, columns=NEWS_FEATURE_COLUMNS)
    scored = score_news_articles(deduplicate_articles(articles))

    valid_scored = []
    for a in scored:
        if a.get("published_at"):
            ts = max(a["published_at"], a.get("received_at") or a["published_at"])
            valid_scored.append((ts, a))
    valid_scored.sort(key=lambda x: x[0])

    left_idx = 0
    right_idx = 0
    from datetime import timedelta

    for timestamp in index:
        current_ts = timestamp.to_pydatetime()
        cutoff_ts = current_ts - timedelta(days=half_life_days * 10)

        while right_idx < len(valid_scored) and valid_scored[right_idx][0] < current_ts:
            right_idx += 1

        while left_idx < right_idx and valid_scored[left_idx][0] < cutoff_ts:
            left_idx += 1

        eligible = [item[1] for item in valid_scored[left_idx:right_idx]]

        if not eligible:
            result.loc[timestamp, "News_Missing_Indicator"] = 1.0
            continue

        ages = np.array(
            [
                (current_ts - item[0]).total_seconds() / 86400
                for item in valid_scored[left_idx:right_idx]
            ]
        )
        weights = np.exp(-math.log(2) * ages / half_life_days) * np.array(
            [a.get("relevance", 1.0) for a in eligible]
        )
        scores = np.array([a["sentiment_score"] for a in eligible])
        result.loc[timestamp, "News_Sentiment"] = float(np.average(scores, weights=weights))
        result.loc[timestamp, "News_Article_Count"] = float(weights.sum())
        result.loc[timestamp, "News_Sentiment_Confidence"] = min(1.0, float(weights.sum()) / 5)
        result.loc[timestamp, "News_Positive_Share"] = float(
            np.average([a["sentiment_positive"] for a in eligible], weights=weights)
        )
        result.loc[timestamp, "News_Negative_Share"] = float(
            np.average([a["sentiment_negative"] for a in eligible], weights=weights)
        )
        result.loc[timestamp, "News_Source_Diversity"] = float(
            len({a.get("publisher") or a.get("provider") for a in eligible})
        )
        result.loc[timestamp, "News_Novelty"] = float(
            len({_title_key(a["title"]) for a in eligible}) / len(eligible)
        )
        result.loc[timestamp, "News_Coverage_Quality"] = float(
            sum(a.get("published_at") is not None for a in eligible) / len(eligible)
        )
    return result


def merge_historical_news_features(
    feature_frame: pd.DataFrame, articles: list[dict[str, Any]], *, half_life_days: float = 7.0
) -> pd.DataFrame:
    if not isinstance(feature_frame.index, pd.DatetimeIndex):
        raise ValueError("feature_frame must have a DatetimeIndex.")
    news = build_daily_sentiment_features(
        articles, feature_frame.index, half_life_days=half_life_days
    )
    if set(news).intersection(feature_frame):
        raise ValueError("Feature frame already contains news columns.")
    return feature_frame.join(news)


def load_news_archive(path: Path) -> list[dict[str, Any]]:
    """Read JSONL or Parquet, validate immutable archive records, and normalise."""
    if path.suffix.lower() == ".jsonl":
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif path.suffix.lower() in {".parquet", ".pq"}:
        records = pd.read_parquet(path).to_dict("records")
    else:
        raise ValueError("News archive must be JSONL or Parquet.")
    for index, record in enumerate(records):
        missing = ARCHIVE_REQUIRED_FIELDS.difference(record)
        if missing:
            raise ValueError(f"Archive row {index} missing required fields: {sorted(missing)}")
        published, received = (
            _parse_timestamp(record["published_at_utc"]),
            _parse_timestamp(record["received_at_utc"]),
        )
        if not published or not received or received < published:
            raise ValueError(f"Archive row {index} has invalid timestamps.")
    return deduplicate_articles(normalise_news_articles(records))


def archive_manifest(articles: list[dict[str, Any]], source: Path) -> dict[str, Any]:
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    timestamps = [a["published_at"] for a in articles if a.get("published_at")]
    return {
        "schema_version": 1,
        "source_sha256": digest,
        "article_count": len(articles),
        "timestamped_article_count": len(timestamps),
        "first_published_at": min(timestamps).isoformat() if timestamps else None,
        "last_published_at": max(timestamps).isoformat() if timestamps else None,
    }


def get_live_financial_sentiment(ticker: str) -> dict[str, Any]:
    fallback = {
        "score": 0.0,
        "status": "fallback",
        "provider": "yfinance",
        "method": "vader_financial",
        "article_count": 0,
        "timestamped_article_count": 0,
        "unique_source_count": 0,
        "model_feature_status": "context_only",
        "reason": "no_usable_news",
    }
    try:
        articles = score_news_articles(
            deduplicate_articles(normalise_news_articles(yf.Ticker(ticker).news))
        )
    except Exception:
        logger.exception("Error fetching news sentiment for %s", ticker)
        return fallback | {"reason": "upstream_error"}
    if not articles:
        return fallback
    timestamps = [a["published_at"] for a in articles if a.get("published_at")]
    weights = np.array(
        [
            1.0
            if not a.get("published_at")
            else math.exp(
                -math.log(2)
                * max(0, (datetime.now(UTC) - a["published_at"]).total_seconds())
                / 86400
                / 3
            )
            for a in articles
        ]
    )
    return {
        "score": round(
            float(np.average([a["sentiment_score"] for a in articles], weights=weights)), 4
        ),
        "weighted_score": round(
            float(np.average([a["sentiment_score"] for a in articles], weights=weights)), 4
        ),
        "status": "live",
        "provider": "yfinance",
        "method": "vader_financial",
        "article_count": len(articles),
        "timestamped_article_count": len(timestamps),
        "unique_source_count": len({a.get("publisher") or a["provider"] for a in articles}),
        "freshest_article_at": max(timestamps).isoformat() if timestamps else None,
        "event_categories": sorted({event for a in articles for event in a["event_categories"]}),
        "model_feature_status": "context_only",
        "reason": None,
    }
