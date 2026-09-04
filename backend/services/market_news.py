"""Institutional market news service with multi-provider fallbacks.

Tries in order:
1. Alpaca news API (if credentials configured)
2. Yahoo Finance live news (headlines, summaries, publisher metadata)
3. U.S. SEC EDGAR material corporate filings (8-K, 10-Q, 10-K)
4. Curated local news archive cache (JSONL)

Returns rich headline objects with title, source, publication time, summary,
URL, and sentiment metrics (compound, polarity breakdown, bullish/bearish badges).
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from cachetools import TTLCache
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

_cache: TTLCache = TTLCache(maxsize=30, ttl=10 * 60)
_lock = threading.RLock()
_sentiment = SentimentIntensityAnalyzer()
_sentiment.lexicon.update(
    {
        # Bullish / Positive market terminology
        "surge": 2.5,
        "surges": 2.5,
        "surging": 2.5,
        "surged": 2.5,
        "soar": 2.8,
        "soars": 2.8,
        "soaring": 2.8,
        "soared": 2.8,
        "rally": 2.4,
        "rallies": 2.4,
        "rallying": 2.4,
        "rallied": 2.4,
        "jump": 2.0,
        "jumps": 2.0,
        "jumping": 2.0,
        "jumped": 2.0,
        "gain": 2.0,
        "gains": 2.0,
        "gaining": 2.0,
        "gained": 2.0,
        "outperform": 2.6,
        "outperforms": 2.6,
        "outperforming": 2.6,
        "outperformed": 2.6,
        "upgrade": 2.7,
        "upgrades": 2.7,
        "upgrading": 2.7,
        "upgraded": 2.7,
        "beat": 2.2,
        "beats": 2.2,
        "beating": 2.2,
        "bullish": 2.5,
        "bull": 2.2,
        "boom": 2.4,
        "booming": 2.4,
        "rebound": 2.0,
        "rebounds": 2.0,
        "breakout": 2.3,
        "upside": 2.0,
        "growth": 1.8,
        "record": 1.6,
        "buy": 1.8,
        "overweight": 2.0,
        # Bearish / Negative market terminology
        "plunge": -2.8,
        "plunges": -2.8,
        "plunging": -2.8,
        "plunged": -2.8,
        "slump": -2.5,
        "slumps": -2.5,
        "slumping": -2.5,
        "slumped": -2.5,
        "drop": -2.2,
        "drops": -2.2,
        "dropping": -2.2,
        "dropped": -2.2,
        "fall": -2.0,
        "falls": -2.0,
        "falling": -2.0,
        "fell": -2.0,
        "tumble": -2.6,
        "tumbles": -2.6,
        "tumbling": -2.6,
        "tumbled": -2.6,
        "crash": -3.0,
        "crashes": -3.0,
        "crashing": -3.0,
        "crashed": -3.0,
        "dive": -2.4,
        "dives": -2.4,
        "diving": -2.4,
        "dived": -2.4,
        "probe": -2.0,
        "probes": -2.0,
        "probing": -2.0,
        "probed": -2.0,
        "investigation": -2.0,
        "investigates": -2.0,
        "downgrade": -2.7,
        "downgrades": -2.7,
        "downgrading": -2.7,
        "downgraded": -2.7,
        "underperform": -2.4,
        "underperforms": -2.4,
        "miss": -2.2,
        "misses": -2.2,
        "missing": -2.2,
        "missed": -2.2,
        "shortfall": -2.2,
        "selloff": -2.5,
        "sell-off": -2.5,
        "selling": -1.8,
        "bearish": -2.5,
        "bear": -2.0,
        "loss": -2.3,
        "losses": -2.3,
        "lawsuit": -2.3,
        "warning": -2.0,
        "warns": -2.0,
        "warned": -2.0,
        "downside": -2.0,
        "underweight": -2.0,
    }
)

CIK_MAPPING: dict[str, str] = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "GOOGL": "0001652044",
    "GOOG": "0001652044",
    "NVDA": "0001045810",
    "TSLA": "0001318605",
}

SEC_MATERIAL_FORMS = {"8-K", "8-K/A", "10-Q", "10-Q/A", "10-K", "10-K/A", "6-K"}

SEC_ITEM_DESCRIPTIONS: dict[str, str] = {
    "1.01": "Material Agreement",
    "1.02": "Termination of Agreement",
    "2.01": "Acquisition or Disposition",
    "2.02": "Results of Operations and Financial Condition",
    "2.03": "Financial Obligation",
    "3.02": "Unregistered Sales of Equity",
    "4.01": "Change in Certifying Accountant",
    "5.01": "Change in Control",
    "5.02": "Departure/Appointment of Principal Officers",
    "7.01": "Regulation FD Disclosure",
    "8.01": "Other Material Events",
}


def _safe_url(value: object) -> str | None:
    candidate = str(value or "").strip()
    parsed = urlparse(candidate)
    return candidate if parsed.scheme in ("https", "http") and parsed.netloc else None


def _format_news_item(
    *,
    identifier: str | None = None,
    title: str,
    summary: str = "",
    source: str = "Financial Wire",
    published_at: str | datetime = "",
    url: str | None = None,
    provider: str = "unknown",
) -> dict[str, Any] | None:
    clean_title = str(title or "").strip()
    if not clean_title:
        return None

    clean_summary = str(summary or "").strip()
    if clean_summary == clean_title:
        clean_summary = ""

    pub_iso = ""
    if isinstance(published_at, datetime):
        if published_at.tzinfo is None:
            pub_iso = published_at.replace(tzinfo=UTC).isoformat().replace("+00:00", "Z")
        else:
            pub_iso = published_at.astimezone(UTC).isoformat().replace("+00:00", "Z")
    elif published_at:
        raw = str(published_at).strip()
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            pub_iso = dt.astimezone(UTC).isoformat().replace("+00:00", "Z")
        except ValueError:
            pub_iso = raw

    if not pub_iso:
        pub_iso = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    sentiment_target = f"{clean_title}. {clean_summary}" if clean_summary else clean_title
    scores = _sentiment.polarity_scores(sentiment_target)
    compound = float(scores["compound"])

    if compound > 0.15:
        label = "positive"
        badge = "Bullish"
    elif compound < -0.15:
        label = "negative"
        badge = "Bearish"
    else:
        label = "neutral"
        badge = "Neutral"

    if not identifier:
        identifier = hashlib.sha256(f"{clean_title}\0{pub_iso}".encode()).hexdigest()[:16]

    return {
        "id": identifier,
        "title": clean_title[:300],
        "headline": clean_title[:300],
        "source": str(source or "Financial Wire")[:100],
        "published_at": pub_iso,
        "summary": clean_summary[:1000],
        "url": _safe_url(url),
        "sentiment": compound,
        "sentiment_label": label,
        "sentiment_badge": badge,
        "sentiment_pos": float(scores["pos"]),
        "sentiment_neg": float(scores["neg"]),
        "sentiment_neu": float(scores["neu"]),
        "provider": provider,
    }


def _fetch_from_alpaca(
    symbol: str,
    key_id: str,
    secret_key: str,
    base_url: str,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    end = datetime.now(UTC)
    start = end - timedelta(days=14)
    response = httpx.get(
        f"{base_url.rstrip('/')}/v1beta1/news",
        params={
            "symbols": symbol,
            "start": start.isoformat().replace("+00:00", "Z"),
            "end": end.isoformat().replace("+00:00", "Z"),
            "sort": "desc",
            "limit": 10,
            "include_content": "false",
        },
        headers={
            "APCA-API-KEY-ID": key_id,
            "APCA-API-SECRET-KEY": secret_key,
            "Accept": "application/json",
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    raw_items = payload.get("news", []) if isinstance(payload, dict) else []
    items: list[dict[str, Any]] = []
    for article in raw_items:
        if not isinstance(article, dict):
            continue
        headline = str(article.get("headline") or "").strip()
        summary = str(article.get("summary") or "").strip()
        item = _format_news_item(
            identifier=str(article.get("id") or ""),
            title=headline,
            summary=summary,
            source=str(article.get("source") or "Alpaca Feed"),
            published_at=str(article.get("created_at") or article.get("updated_at") or ""),
            url=article.get("url"),
            provider="alpaca",
        )
        if item is not None:
            items.append(item)
    return items


def _fetch_from_yahoo(symbol: str) -> list[dict[str, Any]]:
    import yfinance as yf

    ticker_obj = yf.Ticker(symbol)
    raw_news: list[dict[str, Any]] = list(ticker_obj.news or [])
    try:
        search_obj = yf.Search(symbol, news_count=15)
        if search_obj.news:
            raw_news.extend(search_obj.news)
    except Exception:
        pass

    items: list[dict[str, Any]] = []
    seen_titles: set[str] = set()

    for article in raw_news:
        if not isinstance(article, dict):
            continue
        content = article.get("content") if isinstance(article.get("content"), dict) else {}
        title = str(
            content.get("title") or article.get("headline") or article.get("title") or ""
        ).strip()
        if not title or title.lower() in seen_titles:
            continue
        seen_titles.add(title.lower())

        summary = str(
            content.get("summary")
            or content.get("description")
            or article.get("summary")
            or article.get("description")
            or ""
        ).strip()

        pub_raw = (
            content.get("pubDate")
            or content.get("displayTime")
            or article.get("published_at")
            or article.get("created_at")
        )
        if not pub_raw and article.get("providerPublishTime"):
            try:
                pub_raw = datetime.fromtimestamp(
                    int(article["providerPublishTime"]), tz=UTC
                ).isoformat()
            except Exception:
                pub_raw = ""

        source = str(
            (content.get("provider") or {}).get("displayName")
            or article.get("source")
            or article.get("publisher")
            or "Yahoo Finance"
        )
        url = (
            (content.get("canonicalUrl") or {}).get("url")
            or (content.get("clickThroughUrl") or {}).get("url")
            or content.get("previewUrl")
            or article.get("url")
            or article.get("link")
        )
        item_id = str(article.get("id") or content.get("id") or "")

        item = _format_news_item(
            identifier=item_id,
            title=title,
            summary=summary,
            source=source,
            published_at=str(pub_raw or ""),
            url=url,
            provider="yahoo",
        )
        if item is not None:
            items.append(item)
    return items


def _fetch_from_sec_edgar(symbol: str) -> list[dict[str, Any]]:
    cik = CIK_MAPPING.get(symbol)
    if not cik:
        return []
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    headers = {"User-Agent": "SignalSeven/1.0 (institutional@signalseven.terminal)"}
    with httpx.Client(timeout=12.0, headers=headers) as client:
        response = client.get(url)
        response.raise_for_status()
        payload = response.json()

    recent = payload.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    acceptance_times = recent.get("acceptanceDateTime", [])
    descriptions = recent.get("primaryDocDescription", [])
    accession_numbers = recent.get("accessionNumber", [])
    items_list = recent.get("items", [])

    items: list[dict[str, Any]] = []
    for i in range(min(len(forms), 40)):
        form = str(forms[i]).strip()
        if form not in SEC_MATERIAL_FORMS:
            continue
        acc_time = str(acceptance_times[i] if i < len(acceptance_times) else "").strip()
        acc_num = str(accession_numbers[i] if i < len(accession_numbers) else "").strip()
        desc = str(descriptions[i] if i < len(descriptions) else form).strip() or form
        raw_item_code = str(items_list[i] if i < len(items_list) else "").strip()

        item_detail = ""
        if raw_item_code:
            matched = [
                SEC_ITEM_DESCRIPTIONS[code.strip()]
                for code in raw_item_code.split(",")
                if code.strip() in SEC_ITEM_DESCRIPTIONS
            ]
            if matched:
                item_detail = f" ({', '.join(matched)})"

        title = f"{symbol} Form {form}{item_detail}: {desc}"
        summary = (
            f"Official U.S. SEC EDGAR filing by {symbol}. Form {form}{item_detail}. "
            f"Accession number: {acc_num}."
        )
        filing_url = f"https://www.sec.gov/edgar/browse/?CIK={cik}"

        item = _format_news_item(
            identifier=f"sec-{acc_num}" if acc_num else None,
            title=title,
            summary=summary,
            source="U.S. SEC EDGAR",
            published_at=acc_time,
            url=filing_url,
            provider="sec_edgar",
        )
        if item is not None:
            items.append(item)
    return items


def _fetch_from_cache(symbol: str) -> list[dict[str, Any]]:
    candidates = [
        Path.cwd() / "data" / "news" / "alpaca" / f"{symbol}.jsonl",
        Path(__file__).resolve().parents[2] / "data" / "news" / "alpaca" / f"{symbol}.jsonl",
        Path(__file__).resolve().parents[1] / "data" / "news" / "alpaca" / f"{symbol}.jsonl",
    ]
    target_file = None
    for cand in candidates:
        if cand.is_file():
            target_file = cand
            break
    if not target_file:
        return []

    lines = [
        line.strip()
        for line in target_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    items: list[dict[str, Any]] = []
    for line in reversed(lines[-15:]):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        title = record.get("headline") or record.get("title") or ""
        item = _format_news_item(
            identifier=record.get("id"),
            title=title,
            summary=record.get("summary") or f"Market context record for {symbol}.",
            source=record.get("source") or "Market Wire",
            published_at=record.get("published_at") or record.get("collected_at") or "",
            url=record.get("url"),
            provider=record.get("provider") or "cached_archive",
        )
        if item is not None:
            items.append(item)
    return items


def get_exchange_market_close(symbol: str, session_date: str | None = None) -> datetime | None:
    """Resolve the regular market close UTC timestamp for the ticker's exchange."""
    try:
        import pandas as pd
        import pandas_market_calendars as mcal

        from calendars import latest_completed_trading_session, resolve_calendar

        cal_name, _ = resolve_calendar(symbol)
        mcal_name = "LSE" if cal_name == "LSE" else "NYSE"
        if not session_date:
            session_date = latest_completed_trading_session(exchange=mcal_name).date().isoformat()
        calendar = mcal.get_calendar(mcal_name)
        schedule = calendar.schedule(start_date=session_date, end_date=session_date)
        if not schedule.empty and "market_close" in schedule.columns:
            close_ts = schedule.iloc[0]["market_close"]
            if isinstance(close_ts, pd.Timestamp):
                return close_ts.to_pydatetime().astimezone(UTC)
    except Exception:
        pass
    return None


def fetch_recent_news(
    ticker: str,
    *,
    key_id: str | None = None,
    secret_key: str | None = None,
    base_url: str = "https://data.alpaca.markets",
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Fetch rich financial headlines with automatic fallback.

    Tries Alpaca -> Yahoo Finance -> SEC EDGAR -> Cached JSONL archive.
    Returns rich headline objects with title, source, publication time,
    summary, URL, and sentiment metrics.
    """
    symbol = ticker.strip().upper()
    with _lock:
        cached = _cache.get(symbol)
        if cached is not None:
            return dict(cached)

    items: list[dict[str, Any]] = []
    provider_used = "none"

    # 1. Try Alpaca if credentials configured and not a foreign exchange (e.g. UK .L)
    clean_key = (key_id or "").strip()
    clean_secret = (secret_key or "").strip()
    if clean_key and clean_secret and not symbol.endswith(".L"):
        try:
            items = _fetch_from_alpaca(symbol, clean_key, clean_secret, base_url, timeout_seconds)
            if items:
                provider_used = "alpaca"
        except Exception as exc:
            logger.info("Alpaca news fetch failed for %s (%s); falling back to Yahoo", symbol, exc)

    # 2. Try Yahoo Finance fallback
    if not items:
        try:
            items = _fetch_from_yahoo(symbol)
            if items:
                provider_used = "yahoo"
        except Exception as exc:
            logger.info(
                "Yahoo Finance news fetch failed for %s (%s); falling back to SEC EDGAR",
                symbol,
                exc,
            )

    # 3. Try SEC EDGAR fallback
    if not items:
        try:
            items = _fetch_from_sec_edgar(symbol)
            if items:
                provider_used = "sec_edgar"
        except Exception as exc:
            logger.info(
                "SEC EDGAR news fetch failed for %s (%s); falling back to cached archive",
                symbol,
                exc,
            )

    # 4. Try Cached JSONL news archive fallback
    if not items:
        try:
            items = _fetch_from_cache(symbol)
            if items:
                provider_used = "cached_archive"
        except Exception as exc:
            logger.warning("Cache news load failed for %s: %s", symbol, exc)

    items.sort(key=lambda it: str(it.get("published_at") or ""), reverse=True)

    market_close_utc = get_exchange_market_close(symbol)
    for it in items:
        pub_str = str(it.get("published_at") or "")
        try:
            pub_dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00")).astimezone(UTC)
            if market_close_utc is not None:
                is_after = pub_dt > market_close_utc
                it["after_market_close"] = is_after
                it["session_timing"] = "after_hours" if is_after else "regular_hours"
        except Exception:
            it["after_market_close"] = False
            it["session_timing"] = "regular_hours"

    result = {
        "status": "available" if items else "unavailable",
        "items": items[:10],
        "role": "context_only",
        "used_by_model": False,
        "provider": provider_used,
        "exchange_close_utc": (
            market_close_utc.isoformat().replace("+00:00", "Z") if market_close_utc else None
        ),
        "as_of": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    with _lock:
        _cache[symbol] = result
    return dict(result)


def clear_news_cache() -> None:
    with _lock:
        _cache.clear()
