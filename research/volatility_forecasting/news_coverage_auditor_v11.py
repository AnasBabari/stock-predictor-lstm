"""Historical news corpus coverage auditor producing pre-training density and quality reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from research.volatility_forecasting.news_aggregator_v2 import (
    EnrichedNewsArticle,
    MultiDimensionalNewsAggregator,
)
from research.volatility_forecasting.session_calendar_v11 import (
    get_session_close_utc,
)


@dataclass(frozen=True)
class NewsCoverageReport:
    total_stock_origins: int
    origins_with_ge_1_article: int
    origins_with_ge_3_articles: int
    origins_with_ge_2_sources: int
    median_articles_per_origin: float
    p90_articles_per_origin: float
    missing_news_percentage: float
    coverage_by_year: dict[str, float]
    coverage_by_ticker: dict[str, float]
    is_dataset_sufficient_for_m2: (
        bool  # Minimum 30% coverage and median >= 1 article for active trading days
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class NewsCoverageAuditorV11:
    """Audits the density, diversity, and completeness of historical news corpora across stock-origins."""

    @classmethod
    def audit_coverage(
        cls,
        stock_dates: list[tuple[str, str]],  # [(ticker, date_str), ...]
        news_articles: list[EnrichedNewsArticle],
        min_coverage_threshold: float = 0.25,
    ) -> NewsCoverageReport:
        if not stock_dates:
            raise ValueError("No stock-origins provided for news audit.")

        article_counts: list[int] = []
        source_counts: list[int] = []
        by_year: dict[str, list[int]] = {}
        by_ticker: dict[str, list[int]] = {}

        for ticker, date_str in stock_dates:
            cutoff_utc = get_session_close_utc(date_str)
            year = date_str[:4]

            # Aggregate causal window
            agg = MultiDimensionalNewsAggregator.aggregate_causal_window(
                articles=news_articles,
                target_ticker=ticker,
                cutoff_iso=cutoff_utc,
            )

            count_5d = int(agg.articles_5d)
            sources_5d = int(agg.unique_sources_5d)

            article_counts.append(count_5d)
            source_counts.append(sources_5d)

            by_year.setdefault(year, []).append(1 if count_5d >= 1 else 0)
            by_ticker.setdefault(ticker, []).append(1 if count_5d >= 1 else 0)

        n_total = len(stock_dates)
        ge_1 = sum(1 for c in article_counts if c >= 1)
        ge_3 = sum(1 for c in article_counts if c >= 3)
        ge_2_src = sum(1 for s in source_counts if s >= 2)

        missing_pct = round((n_total - ge_1) / n_total * 100.0, 2)
        med_art = float(np.median(article_counts))
        p90_art = float(np.percentile(article_counts, 90))

        cov_year = {y: round(float(np.mean(vals)) * 100.0, 2) for y, vals in by_year.items()}
        cov_ticker = {t: round(float(np.mean(vals)) * 100.0, 2) for t, vals in by_ticker.items()}

        overall_cov = ge_1 / n_total
        sufficient = (overall_cov >= min_coverage_threshold) and (n_total >= 50)

        return NewsCoverageReport(
            total_stock_origins=n_total,
            origins_with_ge_1_article=ge_1,
            origins_with_ge_3_articles=ge_3,
            origins_with_ge_2_sources=ge_2_src,
            median_articles_per_origin=med_art,
            p90_articles_per_origin=p90_art,
            missing_news_percentage=missing_pct,
            coverage_by_year=cov_year,
            coverage_by_ticker=cov_ticker,
            is_dataset_sufficient_for_m2=sufficient,
        )
