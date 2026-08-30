"""Canonical-row content addressing for V11 market, news, and PIT-membership payloads.

The snapshot digests must be computed from the canonical rows themselves (every OHLCV
value, article timestamp/headline field, and security identity), never from a JSON
manifest that merely describes the data. Any modification to one OHLCV value, one
article timestamp, one headline, one sentiment value, or one security identity must
change the corresponding snapshot digest.
"""

from __future__ import annotations

import hashlib
import json
from typing import Iterable

import numpy as np
import pandas as pd

from research.volatility_forecasting.news_aggregator_v2 import EnrichedNewsArticle
from research.volatility_forecasting.stable_security_identity_v11 import (
    StableSecurityIdentity,
)

# Canonical float formatting so python/numpy/pandas round-trips hash identically.
_FLOAT_PRECISION: int = 8

OHLCV_COLUMNS: tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume")


def _fmt_float(value: float | np.floating) -> str:
    return np.format_float_positional(
        float(value), unique=False, precision=_FLOAT_PRECISION, trim="-"
    )


def canonical_market_ohlcv_rows(
    equities_ohlcv: dict[str, pd.DataFrame],
    sector_ohlcv: pd.DataFrame,
    market_ohlcv: pd.DataFrame,
) -> list[str]:
    """Canonical, sorted, row-level serialization of every OHLCV payload.

    Lines are sorted by (payload_role, security_id, session_date) so hashing is
    deterministic regardless of insertion order. Column order is fixed by
    OHLCV_COLUMNS. Index values are normalized to ``YYYY-MM-DD`` strings.
    """
    lines: list[str] = []

    for sec_id in sorted(equities_ohlcv):
        df = equities_ohlcv[sec_id]
        for date_str in sorted(str(d)[:10] for d in df.index):
            row = df.loc[df.index.astype(str).str[:10] == date_str].iloc[-1]
            vals = [_fmt_float(row[col]) for col in OHLCV_COLUMNS]
            lines.append("EQUITY|" + "|".join([sec_id, date_str, *vals]))

    for role, df in (("SECTOR", sector_ohlcv), ("MARKET", market_ohlcv)):
        if df is None:
            continue
        for date_str in sorted(str(d)[:10] for d in df.index):
            row = df.loc[df.index.astype(str).str[:10] == date_str].iloc[-1]
            vals = [_fmt_float(row[col]) for col in OHLCV_COLUMNS]
            lines.append(role + "|" + "|".join([date_str, *vals]))

    return lines


def hash_market_ohlcv_payload(
    equities_ohlcv: dict[str, pd.DataFrame],
    sector_ohlcv: pd.DataFrame,
    market_ohlcv: pd.DataFrame,
) -> str:
    """SHA-256 over the canonical OHLCV rows of the entire market payload."""
    lines = canonical_market_ohlcv_rows(equities_ohlcv, sector_ohlcv, market_ohlcv)
    canonical = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def canonical_news_payload_rows(articles: Iterable[EnrichedNewsArticle]) -> list[dict[str, object]]:
    """Canonical per-article rows capturing every scientifically relevant field."""
    rows: list[dict[str, object]] = []
    for art in articles:
        rows.append(
            {
                "article_id": art.article_id,
                "ticker": art.ticker,
                "headline": art.headline,
                "source": art.source,
                "published_at": art.published_at,
                "first_seen_at": art.first_seen_at,
                "delivery_time": art.delivery_time,
                "ticker_relevance": round(float(art.ticker_relevance), _FLOAT_PRECISION),
                "event_type": art.event_type,
                "sentiment_score": round(float(art.sentiment_score), _FLOAT_PRECISION),
                "sentiment_magnitude": round(float(art.sentiment_magnitude), _FLOAT_PRECISION),
                "severity_score": round(float(art.severity_score), _FLOAT_PRECISION),
                "uncertainty_score": round(float(art.uncertainty_score), _FLOAT_PRECISION),
                "embedding_vector": [
                    round(float(v), _FLOAT_PRECISION) for v in (art.embedding_vector or [])
                ],
            }
        )
    rows.sort(key=lambda r: (str(r["published_at"]), str(r["article_id"])))
    return rows


def hash_news_payload(articles: Iterable[EnrichedNewsArticle]) -> str:
    """SHA-256 over canonical news rows (headline, timestamps, sentiment, identities)."""
    rows = canonical_news_payload_rows(articles)
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def canonical_pit_membership_rows(identities: Iterable[StableSecurityIdentity]) -> list[str]:
    rows: list[str] = []
    for ident in identities:
        rows.append(
            json.dumps(
                ident.to_dict(), sort_keys=True, separators=(",", ":")
            )
        )
    return sorted(rows)


def hash_pit_membership_payload(identities: Iterable[StableSecurityIdentity]) -> str:
    """SHA-256 over canonical security-identity rows (PIT membership source)."""
    canonical = "\n".join(canonical_pit_membership_rows(identities)).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()