"""Synthetic V11 smoke-panel generator (development/integration use ONLY).

This module is the single home for every NumPy-RNG generated OHLCV and news payload
in the V11 stack. Real data execution must never import or call these generators.
Synthetic outputs MUST be labelled ``provider=synthetic_rng`` with the
``SYNTHETIC_INTEGRATION_SMOKE`` classification and must NEVER be described with words
such as *licensed*, *historical-real*, *certified*, or *production*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from research.volatility_forecasting.news_aggregator_v2 import EnrichedNewsArticle
from research.volatility_forecasting.session_calendar_v11 import get_session_close_utc
from research.volatility_forecasting.stable_security_identity_v11 import (
    SecurityIdentityResolver,
    StableSecurityIdentity,
)

# Fixed RNG seeds preserve bitwise reproducibility with the original V11 synthetic
# integration benchmark (market=1001, sector=1002, equities=2000+i, news=3001).
RNG_MARKET_SEED: int = 1001
RNG_SECTOR_SEED: int = 1002
RNG_EQUITY_SEED_BASE: int = 2000
RNG_NEWS_SEED: int = 3001

PROVIDER_LABEL: str = "synthetic_rng"
CLASSIFICATION_REASON: str = (
    "Synthetic market OHLCV and synthetic news were generated via NumPy RNG "
    "for integration/smoke benchmarking only."
)

# Words that must never appear in synthetic-mode provenance labels.
PROHIBITED_WORDS_IN_SYNTHETIC_MODE: tuple[str, ...] = (
    "licensed",
    "historical-real",
    "certified",
    "production",
)

SYNTHETIC_SESSION_START: str = "2021-01-04"
SYNTHETIC_SESSION_END: str = "2026-08-28"


@dataclass(frozen=True)
class SyntheticPanelPayload:
    equities_ohlcv: dict[str, pd.DataFrame]
    sector_ohlcv: pd.DataFrame
    market_ohlcv: pd.DataFrame
    news_articles: list[EnrichedNewsArticle]
    sessions: list[str]


def build_synthetic_sessions(
    start: str = SYNTHETIC_SESSION_START,
    end: str = SYNTHETIC_SESSION_END,
) -> list[str]:
    """Valid exchange sessions between start and end, holidays excluded."""
    raw_dates = pd.date_range(start, end, freq="B").strftime("%Y-%m-%d").tolist()
    valid_sessions: list[str] = []
    for d in raw_dates:
        try:
            get_session_close_utc(d)
            valid_sessions.append(d)
        except ValueError:
            pass
    return valid_sessions


def generate_synthetic_market_payload(
    universe_identities: list[StableSecurityIdentity],
    sessions: list[str],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Generates RNG market/sector/equities OHLCV payloads (smoke-only)."""
    n_sessions = len(sessions)

    # Market context (QQQ equivalent)
    rng_mkt = np.random.default_rng(RNG_MARKET_SEED)
    mkt_rets = rng_mkt.normal(0.0004, 0.012, size=n_sessions)
    mkt_close = 300.0 * np.exp(np.cumsum(mkt_rets))
    mkt_df = pd.DataFrame(
        {
            "Open": mkt_close * (1.0 - 0.002),
            "High": mkt_close * (1.0 + 0.006),
            "Low": mkt_close * (1.0 - 0.006),
            "Close": mkt_close,
            "Volume": rng_mkt.uniform(3e7, 8e7, size=n_sessions),
        },
        index=sessions,
    )

    # Sector context (XLK / XLV equivalent)
    rng_sec = np.random.default_rng(RNG_SECTOR_SEED)
    sec_rets = 0.7 * mkt_rets + rng_sec.normal(0.0001, 0.008, size=n_sessions)
    sec_close = 150.0 * np.exp(np.cumsum(sec_rets))
    sec_df = pd.DataFrame(
        {
            "Open": sec_close * (1.0 - 0.002),
            "High": sec_close * (1.0 + 0.005),
            "Low": sec_close * (1.0 - 0.005),
            "Close": sec_close,
            "Volume": rng_sec.uniform(1e7, 3e7, size=n_sessions),
        },
        index=sessions,
    )

    # Equities OHLCV keyed by stable security id
    equities_ohlcv: dict[str, pd.DataFrame] = {}
    for idx, ident in enumerate(universe_identities):
        rng_eq = np.random.default_rng(RNG_EQUITY_SEED_BASE + idx)
        beta = 0.8 + 0.1 * idx
        idiosyncratic = rng_eq.normal(0.0001, 0.014, size=n_sessions)
        eq_rets = beta * mkt_rets + idiosyncratic
        eq_close = (100.0 + 30.0 * idx) * np.exp(np.cumsum(eq_rets))
        eq_df = pd.DataFrame(
            {
                "Open": eq_close * (1.0 - 0.003),
                "High": eq_close * (1.0 + 0.008),
                "Low": eq_close * (1.0 - 0.008),
                "Close": eq_close,
                "Volume": rng_eq.uniform(2e6, 1.2e7, size=n_sessions),
            },
            index=sessions,
        )
        equities_ohlcv[ident.security_id] = eq_df

    return equities_ohlcv, sec_df, mkt_df


def generate_synthetic_news_payload(
    universe_identities: list[StableSecurityIdentity],
    sessions: list[str],
) -> list[EnrichedNewsArticle]:
    """Probabilistically generates a synthetic news corpus (smoke-only)."""
    resolver = SecurityIdentityResolver(universe_identities)
    news_articles: list[EnrichedNewsArticle] = []
    event_types_pool = ["clinical_trial", "earnings", "analyst", "regulatory", "general"]
    sources_pool = ["Reuters", "Bloomberg", "WSJ", "PR_Newswire", "BusinessWire"]

    art_id_counter = 1
    rng_news = np.random.default_rng(RNG_NEWS_SEED)

    for d_idx, d_str in enumerate(sessions):
        if d_idx % 3 == 0:
            news_articles.append(
                EnrichedNewsArticle(
                    article_id=f"ART_MACRO_{art_id_counter}",
                    ticker="MARKET",
                    headline=f"Macro economic and interest rate update on {d_str}",
                    source=sources_pool[rng_news.integers(0, len(sources_pool))],
                    published_at=f"{d_str}T14:00:00Z",
                    first_seen_at=f"{d_str}T14:00:05Z",
                    delivery_time=f"{d_str}T14:00:10Z",
                    ticker_relevance=0.4,
                    event_type="macro",
                    sentiment_score=float(rng_news.uniform(-0.5, 0.5)),
                    sentiment_magnitude=float(rng_news.uniform(0.2, 0.8)),
                    severity_score=float(rng_news.uniform(0.1, 0.6)),
                    uncertainty_score=float(rng_news.uniform(0.1, 0.5)),
                    embedding_vector=list(rng_news.normal(0, 1, size=4)),
                )
            )
            art_id_counter += 1

        for ident in universe_identities:
            if (
                resolver.is_active_constituent(ident.security_id, d_str)
                and rng_news.uniform(0, 1) < 0.40
            ):
                news_articles.append(
                    EnrichedNewsArticle(
                        article_id=f"ART_COMP_{art_id_counter}",
                        ticker=ident.security_id,
                        headline=f"Corporate update for {ident.security_id} on {d_str}",
                        source=sources_pool[rng_news.integers(0, len(sources_pool))],
                        published_at=f"{d_str}T15:30:00Z",
                        first_seen_at=f"{d_str}T15:30:05Z",
                        delivery_time=f"{d_str}T15:30:10Z",
                        ticker_relevance=1.0,
                        event_type=event_types_pool[rng_news.integers(0, len(event_types_pool))],
                        sentiment_score=float(rng_news.uniform(-0.8, 0.8)),
                        sentiment_magnitude=float(rng_news.uniform(0.3, 0.9)),
                        severity_score=float(rng_news.uniform(0.2, 0.7)),
                        uncertainty_score=float(rng_news.uniform(0.1, 0.6)),
                        embedding_vector=list(rng_news.normal(0, 1, size=4)),
                    )
                )
                art_id_counter += 1

    return news_articles


def synthetic_classification(
    market_payload_sha256: str,
    news_payload_sha256: str,
    pit_membership_sha256: str,
) -> dict[str, Any]:
    """Classification block for the synthetic smoke benchmark."""
    return {
        "experiment_type": "SYNTHETIC_INTEGRATION_SMOKE",
        "provider": PROVIDER_LABEL,
        "license_id": "synthetic_dev_smoke",
        "certification_eligible": False,
        "sealed_test_eligible": False,
        "reason": CLASSIFICATION_REASON,
        "market_payload_sha256": market_payload_sha256,
        "news_payload_sha256": news_payload_sha256,
        "pit_membership_sha256": pit_membership_sha256,
        "market_snapshot_id": "MARKET_SNAPSHOT_NDX100_V11_SYNTH",
        "news_snapshot_id": "NEWS_LAKE_NDX100_V11_SYNTH",
        "pit_snapshot_id": "PIT_UNIVERSE_NDX100_V11_SYNTH",
        "real_market_data": False,
        "real_news_data": False,
        "pit_membership": True,
    }


def build_synthetic_panel_payload(
    universe_identities: list[StableSecurityIdentity],
) -> SyntheticPanelPayload:
    """Complete synthetic payload used by the smoke benchmark entrypoint."""
    sessions = build_synthetic_sessions()
    equities_ohlcv, sec_df, mkt_df = generate_synthetic_market_payload(
        universe_identities, sessions
    )
    news_articles = generate_synthetic_news_payload(universe_identities, sessions)
    return SyntheticPanelPayload(
        equities_ohlcv=equities_ohlcv,
        sector_ohlcv=sec_df,
        market_ohlcv=mkt_df,
        news_articles=news_articles,
        sessions=sessions,
    )
