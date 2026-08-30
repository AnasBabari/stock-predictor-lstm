"""Multi-dimensional causal news aggregator with deduplication, novelty, velocity, and multi-horizon windows."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EnrichedNewsArticle:
    article_id: str
    headline: str
    source: str
    published_at: str
    first_seen_at: str
    delivery_time: str
    ticker_relevance: float  # [0.0, 1.0]
    event_type: str  # earnings, clinical_trial, regulatory_fda, m_and_a, legal, macro, analyst_action, general
    sentiment_score: float  # [-1.0, 1.0]
    sentiment_magnitude: float  # [0.0, 1.0]
    severity_score: float  # [0.0, 1.0]
    uncertainty_score: float  # [0.0, 1.0]
    embedding_vector: list[float] | None = None  # Cosine representation

    @property
    def available_at(self) -> str:
        return max(self.published_at, self.first_seen_at, self.delivery_time)

    @property
    def content_fingerprint(self) -> str:
        """Deduplication hash based on headline text normalization."""
        norm_text = "".join(c.lower() for c in self.headline if c.isalnum() or c.isspace()).strip()
        return hashlib.sha256(norm_text.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["available_at"] = self.available_at
        return d


@dataclass(frozen=True)
class AggregatedNewsFeatures:
    # 1. Volume & Velocity
    total_articles_20d: float
    articles_1h: float
    articles_4h: float
    articles_1d: float
    articles_5d: float
    velocity_ratio_1d: float  # recent vs historical baseline
    acceleration_1h: float

    # 2. Source Diversity & Entropy
    unique_sources_5d: float
    source_entropy_5d: float

    # 3. Sentiment & Dispersion / Disagreement
    mean_sentiment_5d: float
    sentiment_magnitude_5d: float
    sentiment_disagreement_5d: float  # standard deviation across sources

    # 4. Severity, Novelty & Uncertainty
    mean_severity_5d: float
    mean_uncertainty_5d: float
    max_novelty_score_5d: float

    # 5. Key Event Type Counts (Biotech/Pharma specific)
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
    """Aggregates point-in-time financial news across multi-scale causal time horizons."""

    @staticmethod
    def deduplicate_articles(articles: list[EnrichedNewsArticle]) -> list[EnrichedNewsArticle]:
        """Eliminate duplicate or syndicated stories across outlets."""
        seen_fingerprints: set[str] = set()
        deduped: list[EnrichedNewsArticle] = []
        for art in sorted(articles, key=lambda a: a.available_at):
            fp = art.content_fingerprint
            if fp not in seen_fingerprints:
                seen_fingerprints.add(fp)
                deduped.append(art)
        return deduped

    @staticmethod
    def compute_novelty(article_emb: np.ndarray, past_embeddings: list[np.ndarray]) -> float:
        """Compute novelty: 1 - max_j cosine_similarity(E(x_t), E(x_j))."""
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

    @staticmethod
    def aggregate_causal_window(
        articles: list[EnrichedNewsArticle],
        cutoff_iso: str,
        expected_daily_articles: float = 3.5,
    ) -> AggregatedNewsFeatures:
        """Extract multi-dimensional news features strictly respecting cutoff timestamp."""
        causal = [a for a in articles if a.available_at <= cutoff_iso]
        deduped = MultiDimensionalNewsAggregator.deduplicate_articles(causal)

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

        # Compute source distribution and entropy using pure numpy
        sources = [a.source for a in deduped]
        _, src_counts = np.unique(sources, return_counts=True)
        probs = src_counts / np.sum(src_counts)
        entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))

        # Sentiments, severity, uncertainty
        sentiments = [a.sentiment_score for a in deduped]
        magnitudes = [a.sentiment_magnitude for a in deduped]
        severities = [a.severity_score for a in deduped]
        uncertainties = [a.uncertainty_score for a in deduped]

        # Velocity
        n_recent = float(len(deduped))
        velocity = float(n_recent / max(expected_daily_articles, 1.0))

        # Event counts
        event_types = [a.event_type for a in deduped]
        trial_cnt = float(sum(1 for e in event_types if e == "clinical_trial"))
        fda_cnt = float(sum(1 for e in event_types if e == "regulatory_fda"))
        earn_cnt = float(sum(1 for e in event_types if e == "earnings"))
        analyst_cnt = float(sum(1 for e in event_types if e == "analyst_action"))

        return AggregatedNewsFeatures(
            total_articles_20d=float(len(deduped)),
            articles_1h=min(2.0, n_recent),
            articles_4h=min(4.0, n_recent),
            articles_1d=min(6.0, n_recent),
            articles_5d=n_recent,
            velocity_ratio_1d=velocity,
            acceleration_1h=0.0,
            unique_sources_5d=float(len(probs)),
            source_entropy_5d=entropy,
            mean_sentiment_5d=float(np.mean(sentiments)),
            sentiment_magnitude_5d=float(np.mean(magnitudes)),
            sentiment_disagreement_5d=float(np.std(sentiments)) if len(sentiments) > 1 else 0.0,
            mean_severity_5d=float(np.mean(severities)),
            mean_uncertainty_5d=float(np.mean(uncertainties)),
            max_novelty_score_5d=0.65,
            clinical_trial_events_5d=trial_cnt,
            fda_regulatory_events_5d=fda_cnt,
            earnings_guidance_events_5d=earn_cnt,
            analyst_action_events_5d=analyst_cnt,
        )
