"""Historical news corpus coverage auditor separating company-specific from macro-context density."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
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
class NewsCoverageGateV11:
    min_company_nonempty_fraction: float = 0.25
    min_multi_source_fraction: float = 0.10
    min_total_origins: int = 50
    minimum_year_coverage_floor: float = 0.10
    minimum_security_count: int = 3

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NewsCoverageReport:
    total_stock_origins: int
    unique_securities: int
    date_range: tuple[str, str]

    # Company-Specific Metrics
    company_origins_ge_1_article: int
    company_origins_ge_3_articles: int
    company_origins_ge_2_sources: int
    company_coverage_percentage: float
    median_company_articles_per_origin: float
    median_company_articles_when_nonempty: float
    p90_company_articles_per_origin: float
    company_source_diversity_median: float

    # Macro & Combined Metrics
    macro_origins_ge_1_article: int
    macro_coverage_percentage: float
    combined_origins_ge_1_article: int
    combined_coverage_percentage: float

    # Breakdowns
    coverage_by_year: dict[str, float]
    coverage_by_security: dict[str, float]
    event_taxonomy_distribution: dict[str, int]

    # Gate Decision
    is_dataset_sufficient_for_m2: bool
    gate_evaluation_details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_markdown_report(self, target_path: Path) -> None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Historical News Corpus Coverage Audit Report (V11)",
            "",
            f"- **Total Stock Origins**: {self.total_stock_origins:,}",
            f"- **Unique Securities**: {self.unique_securities}",
            f"- **Date Range**: {self.date_range[0]} to {self.date_range[1]}",
            f"- **M2 Multimodal Eligibility**: **{'ELIGIBLE' if self.is_dataset_sufficient_for_m2 else 'INELIGIBLE (M0+M1 Training Only)'}**",
            "",
            "---",
            "### 1. Company-Specific Coverage",
            f"- **Origins with >= 1 Company Article**: {self.company_origins_ge_1_article:,} ({self.company_coverage_percentage:.2f}%)",
            f"- **Origins with >= 3 Company Articles**: {self.company_origins_ge_3_articles:,}",
            f"- **Origins with >= 2 Unique Sources**: {self.company_origins_ge_2_sources:,}",
            f"- **Median Articles per Origin**: {self.median_company_articles_per_origin:.1f}",
            f"- **Median Articles when Non-Empty**: {self.median_company_articles_when_nonempty:.1f}",
            f"- **90th Percentile Articles**: {self.p90_company_articles_per_origin:.1f}",
            "",
            "---",
            "### 2. Macro Context & Combined Coverage",
            f"- **Macro Origins with >= 1 Article**: {self.macro_origins_ge_1_article:,} ({self.macro_coverage_percentage:.2f}%)",
            f"- **Combined Origins with >= 1 Article**: {self.combined_origins_ge_1_article:,} ({self.combined_coverage_percentage:.2f}%)",
            "",
            "---",
            "### 3. Coverage by Year",
            "| Year | Company Coverage % |",
            "|---|:---:|",
        ]
        for y, p in sorted(self.coverage_by_year.items()):
            lines.append(f"| {y} | {p:.2f}% |")

        lines.extend(["", "---", "### 4. Event Taxonomy Distribution"])
        for ev, cnt in sorted(self.event_taxonomy_distribution.items()):
            lines.append(f"- **{ev}**: {cnt:,}")

        target_path.write_text("\n".join(lines), encoding="utf-8")


class NewsCoverageAuditorV11:
    """Audits the density, diversity, and company-specific completeness of historical news corpora."""

    @classmethod
    def audit_coverage(
        cls,
        stock_dates: list[tuple[str, str]],  # [(security_id, date_str), ...]
        news_articles: list[EnrichedNewsArticle],
        gate_config: NewsCoverageGateV11 | None = None,
    ) -> NewsCoverageReport:
        if not stock_dates:
            raise ValueError("No stock-origins provided for news audit.")

        gate = gate_config or NewsCoverageGateV11()

        company_counts: list[int] = []
        company_sources: list[int] = []
        macro_counts: list[int] = []
        combined_counts: list[int] = []

        by_year: dict[str, list[int]] = {}
        by_sec: dict[str, list[int]] = {}
        event_counts: dict[str, int] = {}

        unique_secs = len(set(sec for sec, _ in stock_dates))
        sorted_dates = sorted(d for _, d in stock_dates)
        date_range = (sorted_dates[0], sorted_dates[-1])

        for sec_id, date_str in stock_dates:
            cutoff_utc = get_session_close_utc(date_str)
            year = date_str[:4]

            # 1. Company-Specific News Filtering
            comp_articles = [
                a
                for a in news_articles
                if (a.ticker == sec_id or a.ticker.split(".")[-1] == sec_id) and not a.is_macro
            ]
            comp_agg = MultiDimensionalNewsAggregator.aggregate_causal_window(
                articles=comp_articles, target_ticker=sec_id, cutoff_iso=cutoff_utc
            )
            c_cnt = int(comp_agg.articles_5d)
            c_src = int(comp_agg.unique_sources_5d)

            # 2. Macro-Only News Filtering
            macro_articles = [a for a in news_articles if a.is_macro]
            macro_agg = MultiDimensionalNewsAggregator.aggregate_causal_window(
                articles=macro_articles, target_ticker="MARKET", cutoff_iso=cutoff_utc
            )
            m_cnt = int(macro_agg.articles_5d)

            # 3. Combined News
            comb_cnt = c_cnt + m_cnt

            company_counts.append(c_cnt)
            company_sources.append(c_src)
            macro_counts.append(m_cnt)
            combined_counts.append(comb_cnt)

            by_year.setdefault(year, []).append(1 if c_cnt >= 1 else 0)
            by_sec.setdefault(sec_id, []).append(1 if c_cnt >= 1 else 0)

            for a in comp_articles:
                if a.available_at[:10] <= date_str:
                    event_counts[a.event_type] = event_counts.get(a.event_type, 0) + 1

        n_total = len(stock_dates)
        comp_ge_1 = sum(1 for c in company_counts if c >= 1)
        comp_ge_3 = sum(1 for c in company_counts if c >= 3)
        comp_ge_2_src = sum(1 for s in company_sources if s >= 2)
        macro_ge_1 = sum(1 for m in macro_counts if m >= 1)
        comb_ge_1 = sum(1 for cb in combined_counts if cb >= 1)

        comp_cov_pct = round(comp_ge_1 / n_total * 100.0, 2)
        macro_cov_pct = round(macro_ge_1 / n_total * 100.0, 2)
        comb_cov_pct = round(comb_ge_1 / n_total * 100.0, 2)

        med_comp = float(np.median(company_counts))
        nonempty_comp = [c for c in company_counts if c > 0]
        med_when_nonempty = float(np.median(nonempty_comp)) if nonempty_comp else 0.0
        p90_comp = float(np.percentile(company_counts, 90))
        med_src_diversity = float(np.median(company_sources))

        cov_year = {y: round(float(np.mean(vals)) * 100.0, 2) for y, vals in by_year.items()}
        cov_sec = {s: round(float(np.mean(vals)) * 100.0, 2) for s, vals in by_sec.items()}

        min_year_cov = min(cov_year.values()) if cov_year else 0.0
        multi_src_pct = comp_ge_2_src / n_total

        gate_pass = (
            (comp_ge_1 / n_total >= gate.min_company_nonempty_fraction)
            and (multi_src_pct >= gate.min_multi_source_fraction)
            and (n_total >= gate.min_total_origins)
            and (unique_secs >= gate.minimum_security_count)
            and (min_year_cov >= gate.minimum_year_coverage_floor * 100.0)
        )

        return NewsCoverageReport(
            total_stock_origins=n_total,
            unique_securities=unique_secs,
            date_range=date_range,
            company_origins_ge_1_article=comp_ge_1,
            company_origins_ge_3_articles=comp_ge_3,
            company_origins_ge_2_sources=comp_ge_2_src,
            company_coverage_percentage=comp_cov_pct,
            median_company_articles_per_origin=med_comp,
            median_company_articles_when_nonempty=med_when_nonempty,
            p90_company_articles_per_origin=p90_comp,
            company_source_diversity_median=med_src_diversity,
            macro_origins_ge_1_article=macro_ge_1,
            macro_coverage_percentage=macro_cov_pct,
            combined_origins_ge_1_article=comb_ge_1,
            combined_coverage_percentage=comb_cov_pct,
            coverage_by_year=cov_year,
            coverage_by_security=cov_sec,
            event_taxonomy_distribution=event_counts,
            is_dataset_sufficient_for_m2=gate_pass,
            gate_evaluation_details={
                "gate_config": gate.to_dict(),
                "actual_company_coverage_fraction": round(comp_ge_1 / n_total, 4),
                "actual_multi_source_fraction": round(multi_src_pct, 4),
                "actual_total_origins": n_total,
                "actual_unique_securities": unique_secs,
                "min_year_coverage_pct": min_year_cov,
                "gate_passed": gate_pass,
            },
        )
