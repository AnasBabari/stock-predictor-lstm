"""Immutable, availability-timestamped historical news lake and causal exchange alignment.

Ensures that every news article is timestamped with:
  available_at = max(published_at, first_seen_at, provider_delivery_time)
and aligned causally to trading sessions based on exchange close times.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class HistoricalNewsRecord:
    article_id: str
    canonical_url_hash: str
    provider: str
    source: str
    published_at: str
    first_seen_at: str
    provider_delivery_time: str
    retrieved_at: str
    language: str
    title: str
    body_hash: str
    security_ids: list[str]
    sector_ids: list[str]
    country_ids: list[str]
    event_types: list[str]
    license_id: str
    snapshot_id: str

    @property
    def available_at(self) -> str:
        """Effective causal availability timestamp."""
        ts_pub = pd.to_datetime(self.published_at, utc=True)
        ts_seen = pd.to_datetime(self.first_seen_at, utc=True)
        ts_deliv = pd.to_datetime(self.provider_delivery_time, utc=True)
        max_ts = max(ts_pub, ts_seen, ts_deliv)
        return max_ts.strftime("%Y-%m-%dT%H:%M:%SZ")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["available_at"] = self.available_at
        return d


def align_article_to_trading_session(
    available_at_utc: str,
    exchange_session_date: str,
    exchange_close_time_utc: str,
    next_session_date: str | None,
) -> str | None:
    """Align an article to its causal trading session.

    If available_at > exchange_close_time on session date, the article cannot
    influence the session's close-to-close features and belongs strictly to
    next_session_date.
    """
    ts_available = pd.to_datetime(available_at_utc, utc=True)
    ts_close = pd.to_datetime(f"{exchange_session_date}T{exchange_close_time_utc}", utc=True)

    if ts_available <= ts_close:
        return exchange_session_date
    return next_session_date
