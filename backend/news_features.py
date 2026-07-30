"""Point-in-time-safe news normalisation and sentiment features.

The live endpoint uses this module only as descriptive context.  Historical
features are built exclusively from timestamped article records supplied by a
licensed archive, so an article is never allowed to influence an earlier
trading session.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

_analyzer: SentimentIntensityAnalyzer | None = None

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
}


def _sentiment_analyzer() -> SentimentIntensityAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentIntensityAnalyzer()
        _analyzer.lexicon.update(FINANCIAL_LEXICON)
    return _analyzer


def _read_path(article: dict[str, Any], *paths: tuple[str, ...]) -> Any:
    for path in paths:
        value: Any = article
        for key in path:
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(key)
        if value not in (None, ""):
            return value
    return None


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=UTC)
        parsed = pd.Timestamp(value)
        if parsed.tzinfo is None:
            parsed = parsed.tz_localize(UTC)
        return parsed.to_pydatetime().astimezone(UTC)
    except (TypeError, ValueError, OverflowError):
        return None


def normalise_news_articles(raw_articles: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Normalise both legacy and nested yfinance news schemas."""

    normalised = []
    for article in raw_articles or []:
        if not isinstance(article, dict):
            continue
        title = _read_path(article, ("title",), ("content", "title"))
        summary = _read_path(
            article,
            ("summary",),
            ("description",),
            ("content", "summary"),
            ("content", "description"),
        )
        published_at = _parse_timestamp(
            _read_path(
                article,
                ("providerPublishTime",),
                ("pubDate",),
                ("content", "pubDate"),
                ("content", "displayTime"),
            )
        )
        text = " ".join(part.strip() for part in (str(title or ""), str(summary or "")) if part)
        if not text:
            continue
        normalised.append(
            {
                "title": str(title or "").strip(),
                "summary": str(summary or "").strip(),
                "published_at": published_at,
                "publisher": _read_path(
                    article, ("publisher",), ("content", "provider", "displayName")
                ),
                "link": _read_path(article, ("link",), ("content", "canonicalUrl", "url")),
            }
        )
    return normalised


def score_news_articles(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Score normalised articles without discarding provenance or timestamps."""

    analyzer = _sentiment_analyzer()
    scored = []
    for article in articles:
        text = " ".join(part for part in (article["title"], article["summary"]) if part)
        phrase_adjustment = sum(
            value for phrase, value in FINANCIAL_LEXICON.items() if phrase in text.lower()
        )
        score = float(
            np.clip(analyzer.polarity_scores(text)["compound"] + 0.1 * phrase_adjustment, -1, 1)
        )
        scored.append({**article, "sentiment_score": score, "method": "vader_financial"})
    return scored


def build_daily_sentiment_features(
    articles: list[dict[str, Any]],
    sessions: pd.DatetimeIndex,
    *,
    half_life_days: float = 7.0,
) -> pd.DataFrame:
    """Create lag-safe, session-indexed historical news features.

    Articles without a publication timestamp are excluded from training
    features: including them would make point-in-time evaluation impossible.
    """

    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive.")
    source_index = pd.DatetimeIndex(sessions)
    index = (
        source_index.tz_localize(UTC) if source_index.tz is None else source_index.tz_convert(UTC)
    )
    result = pd.DataFrame(
        0.0,
        index=index,
        columns=["News_Sentiment", "News_Article_Count", "News_Sentiment_Confidence"],
    )
    scored = score_news_articles(articles)
    for timestamp in index:
        eligible = [
            item
            for item in scored
            if item["published_at"] is not None and item["published_at"] < timestamp.to_pydatetime()
        ]
        if not eligible:
            continue
        ages = np.asarray(
            [
                (timestamp.to_pydatetime() - item["published_at"]).total_seconds() / 86400
                for item in eligible
            ]
        )
        weights = np.exp(-np.log(2) * ages / half_life_days)
        scores = np.asarray([item["sentiment_score"] for item in eligible])
        result.loc[timestamp, "News_Sentiment"] = float(np.average(scores, weights=weights))
        result.loc[timestamp, "News_Article_Count"] = float(weights.sum())
        result.loc[timestamp, "News_Sentiment_Confidence"] = float(min(1.0, weights.sum() / 5))
    return result


def merge_historical_news_features(
    feature_frame: pd.DataFrame,
    articles: list[dict[str, Any]],
    *,
    half_life_days: float = 7.0,
) -> pd.DataFrame:
    """Join lag-safe news features for an offline ablation without mutating inputs."""

    if not isinstance(feature_frame.index, pd.DatetimeIndex):
        raise ValueError("feature_frame must have a DatetimeIndex.")
    news = build_daily_sentiment_features(
        articles, feature_frame.index, half_life_days=half_life_days
    )
    merged = feature_frame.copy()
    for column in news:
        if column in merged:
            raise ValueError(f"Feature frame already contains {column}.")
        merged[column] = news[column].to_numpy()
    return merged


def get_live_financial_sentiment(ticker: str) -> dict[str, Any]:
    """Return an observable live context value; failure is explicit and non-fatal."""

    fallback = {
        "score": 0.0,
        "status": "fallback",
        "provider": "yfinance",
        "method": "vader_financial",
        "article_count": 0,
        "timestamped_article_count": 0,
        "reason": "no_usable_news",
    }
    try:
        articles = score_news_articles(normalise_news_articles(yf.Ticker(ticker).news))
    except Exception:
        logger.exception("Error fetching news sentiment for %s", ticker)
        return fallback | {"reason": "upstream_error"}
    if not articles:
        return fallback
    scores = np.asarray([item["sentiment_score"] for item in articles])
    timestamps = [item["published_at"] for item in articles if item["published_at"] is not None]
    return {
        "score": round(float(np.mean(scores)), 4),
        "status": "live",
        "provider": "yfinance",
        "method": "vader_financial",
        "article_count": len(articles),
        "timestamped_article_count": len(timestamps),
        "freshest_article_at": max(timestamps).isoformat() if timestamps else None,
        "reason": None,
    }
