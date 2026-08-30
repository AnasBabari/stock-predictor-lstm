"""Multi-dimensional causal news aggregator with ticker binding, real time windows, velocity, acceleration, and novelty."""

from __future__ import annotations

import datetime
import hashlib
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EnrichedNewsArticle:
    article_id: str
    ticker: str  # Bound entity ticker (e.g. "AMGN", "AAPL", or "MARKET")
    headline: str
    source: str
    published_at: str  # ISO-8601 UTC
    first_seen_at: str  # ISO-8601 UTC
    delivery_time: str  # ISO-8601 UTC
    ticker_relevance: float  # [0.0, 1.0]
    event_type: str  # clinical_trial, regulatory_fda, earnings, m_and_a, legal, analyst_action, general, macro
    sentiment_score: float  # [-1.0, 1.0]
    sentiment_magnitude: float  # [0.0, 1.0]
    severity_score: float  # [0.0, 1.0]
    uncertainty_score: float  # [0.0, 1.0]
    embedding_vector: list[float] | None = None  # Cosine representation

    @property
    def available_at(self) -> str:
        return max(self.published_at, self.first_seen_at, self.delivery_time)

    @property
    def is_macro(self) -> bool:
        return self.ticker in ("MARKET", "MACRO", "QQQ", "SPY", "XLV") or self.event_type == "macro"

    @property
    def content_fingerprint(self) -> str:
        norm_text = "".join(c.lower() for c in self.headline if c.isalnum() or c.isspace()).strip()
        raw = f"{self.ticker}:{norm_text}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["available_at"] = self.available_at
        return d


@dataclass(frozen=True)
class AggregatedNewsFeatures:
    # 1. Volume & Velocity (7)
    total_articles_20d: float
    articles_1h: float
    articles_4h: float
    articles_1d: float
    articles_5d: float
    velocity_ratio_1d: float  # 1d count vs daily average of 20d window
    acceleration_1h: float  # articles in [T-1h, T] minus articles in [T-2h, T-1h]

    # 2. Source Diversity & Entropy (2)
    unique_sources_5d: float
    source_entropy_5d: float

    # 3. Sentiment & Dispersion / Disagreement (3)
    mean_sentiment_5d: float
    sentiment_magnitude_5d: float
    sentiment_disagreement_5d: float

    # 4. Severity, Novelty & Uncertainty (3)
    mean_severity_5d: float
    mean_uncertainty_5d: float
    max_novelty_score_5d: float

    # 5. Key Event Type Counts in 5d window (4)
    clinical_trial_events_5d: float
    fda_regulatory_events_5d: float
    earnings_guidance_events_5d: float
    analyst_action_events_5d: float

    def to_array(self) -> np.ndarray:
        return np.array(list(self.__dict__.values()), dtype=float)

    @classmethod
    def feature_names(cls) -> list[str]:
        return list(cls.__annotations__.keys())


class MultiDimensionalNewsAggregator:
    """Aggregates causal news with strict ticker binding and true rolling time windows."""

    @staticmethod
    def _parse_iso(iso_str: str) -> datetime.datetime:
        cleaned = iso_str.replace("Z", "+00:00")
        return datetime.datetime.fromisoformat(cleaned)

    @staticmethod
    def deduplicate_articles(articles: list[EnrichedNewsArticle]) -> list[EnrichedNewsArticle]:
        seen: set[str] = set()
        deduped: list[EnrichedNewsArticle] = []
        for art in sorted(articles, key=lambda a: a.available_at):
            fp = art.content_fingerprint
            if fp not in seen:
                seen.add(fp)
                deduped.append(art)
        return deduped

    @staticmethod
    def compute_cosine_novelty(
        article_emb: np.ndarray,
        past_embeddings: list[np.ndarray],
    ) -> float:
        if not past_embeddings:
            return 1.0
        norm_art = np.linalg.norm(article_emb)
        if norm_art < 1e-8:
            return 0.5
        max_cos = 0.0
        for past_emb in past_embeddings:
            norm_past = np.linalg.norm(past_emb)
            if norm_past > 1e-8:
                cos = float(np.dot(article_emb, past_emb) / (norm_art * norm_past))
                max_cos = max(max_cos, cos)
        return float(np.clip(1.0 - max_cos, 0.0, 1.0))

    @classmethod
    def aggregate_causal_window(
        cls,
        articles: list[EnrichedNewsArticle],
        target_ticker: str,
        cutoff_iso: str,
    ) -> AggregatedNewsFeatures:
        """Aggregate news causally available <= cutoff_iso for target_ticker."""
        cutoff_dt = cls._parse_iso(cutoff_iso)

        # 1. Strict Ticker & Causal Filter
        causal_raw = [
            a
            for a in articles
            if cls._parse_iso(a.available_at) <= cutoff_dt
            and (a.ticker == target_ticker or a.is_macro)
        ]
        deduped = cls.deduplicate_articles(causal_raw)

        if not deduped:
            return AggregatedNewsFeatures(
                total_articles_20d=0.0,
                articles_1h=0.0,
                articles_4h=0.0,
                articles_1d=0.0,
                articles_5d=0.0,
                velocity_ratio_1d=0.0,
                acceleration_1h=0.0,
                unique_sources_5d=0.0,
                source_entropy_5d=0.0,
                mean_sentiment_5d=0.0,
                sentiment_magnitude_5d=0.0,
                sentiment_disagreement_5d=0.0,
                mean_severity_5d=0.0,
                mean_uncertainty_5d=0.0,
                max_novelty_score_5d=0.0,
                clinical_trial_events_5d=0.0,
                fda_regulatory_events_5d=0.0,
                earnings_guidance_events_5d=0.0,
                analyst_action_events_5d=0.0,
            )

        # 2. Window Timestamps
        t_1h = cutoff_dt - datetime.timedelta(hours=1)
        t_2h = cutoff_dt - datetime.timedelta(hours=2)
        t_4h = cutoff_dt - datetime.timedelta(hours=4)
        t_1d = cutoff_dt - datetime.timedelta(days=1)
        t_5d = cutoff_dt - datetime.timedelta(days=5)
        t_20d = cutoff_dt - datetime.timedelta(days=20)

        # 3. Partition articles by genuine time deltas
        art_20d = [a for a in deduped if cls._parse_iso(a.available_at) >= t_20d]
        art_5d = [a for a in deduped if cls._parse_iso(a.available_at) >= t_5d]
        art_1d = [a for a in deduped if cls._parse_iso(a.available_at) >= t_1d]
        art_4h = [a for a in deduped if cls._parse_iso(a.available_at) >= t_4h]
        art_1h = [a for a in deduped if cls._parse_iso(a.available_at) >= t_1h]
        art_prev_1h = [a for a in deduped if t_2h <= cls._parse_iso(a.available_at) < t_1h]

        c_20d = float(len(art_20d))
        c_5d = float(len(art_5d))
        c_1d = float(len(art_1d))
        c_4h = float(len(art_4h))
        c_1h = float(len(art_1h))
        c_prev_1h = float(len(art_prev_1h))

        # Velocity & Acceleration
        daily_expected = max(c_20d / 20.0, 0.1)
        velocity_ratio = float(c_1d / daily_expected)
        acceleration_1h = float(c_1h - c_prev_1h)

        # 4. 5-Day Window Statistics
        if art_5d:
            sources = [a.source for a in art_5d]
            _, src_counts = np.unique(sources, return_counts=True)
            probs = src_counts / np.sum(src_counts)
            entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
            unique_sources = float(len(probs))

            sentiments = [a.sentiment_score for a in art_5d]
            magnitudes = [a.sentiment_magnitude for a in art_5d]
            severities = [a.severity_score for a in art_5d]
            uncertainties = [a.uncertainty_score for a in art_5d]

            mean_sent = float(np.mean(sentiments))
            mean_mag = float(np.mean(magnitudes))
            sent_disagree = float(np.std(sentiments)) if len(sentiments) > 1 else 0.0
            mean_sev = float(np.mean(severities))
            mean_unc = float(np.mean(uncertainties))

            # Compute real novelty if embeddings are present
            past_embs = [
                np.array(a.embedding_vector, dtype=float)
                for a in art_20d
                if a.embedding_vector and cls._parse_iso(a.available_at) < t_5d
            ]
            recent_embs = [
                np.array(a.embedding_vector, dtype=float) for a in art_5d if a.embedding_vector
            ]
            novelty_scores = (
                [cls.compute_cosine_novelty(e, past_embs) for e in recent_embs]
                if recent_embs
                else [0.5]
            )
            max_novelty = float(np.max(novelty_scores))

            # Event counts in 5d window
            events = [a.event_type for a in art_5d]
            trial_cnt = float(sum(1 for e in events if e == "clinical_trial"))
            fda_cnt = float(sum(1 for e in events if e == "regulatory_fda"))
            earn_cnt = float(sum(1 for e in events if e == "earnings"))
            analyst_cnt = float(sum(1 for e in events if e == "analyst_action"))
        else:
            unique_sources = 0.0
            entropy = 0.0
            mean_sent = 0.0
            mean_mag = 0.0
            sent_disagree = 0.0
            mean_sev = 0.0
            mean_unc = 0.0
            max_novelty = 0.0
            trial_cnt = 0.0
            fda_cnt = 0.0
            earn_cnt = 0.0
            analyst_cnt = 0.0

        return AggregatedNewsFeatures(
            total_articles_20d=c_20d,
            articles_1h=c_1h,
            articles_4h=c_4h,
            articles_1d=c_1d,
            articles_5d=c_5d,
            velocity_ratio_1d=velocity_ratio,
            acceleration_1h=acceleration_1h,
            unique_sources_5d=unique_sources,
            source_entropy_5d=entropy,
            mean_sentiment_5d=mean_sent,
            sentiment_magnitude_5d=mean_mag,
            sentiment_disagreement_5d=sent_disagree,
            mean_severity_5d=mean_sev,
            mean_uncertainty_5d=mean_unc,
            max_novelty_score_5d=max_novelty,
            clinical_trial_events_5d=trial_cnt,
            fda_regulatory_events_5d=fda_cnt,
            earnings_guidance_events_5d=earn_cnt,
            analyst_action_events_5d=analyst_cnt,
        )
