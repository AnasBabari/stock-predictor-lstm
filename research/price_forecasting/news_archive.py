"""Timestamp-preserving Alpaca, Yahoo, and SEC EDGAR news acquisition for offline model research."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

try:
    from datetime import UTC, datetime
except ImportError:
    from datetime import datetime, timezone

    UTC = timezone.utc  # noqa: UP017

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

NEWS_SCHEMA_VERSION = "alpaca-headline-archive-v1"
_sentiment = SentimentIntensityAnalyzer()

# Revision-lookahead gating policy for daily close-to-close pipeline.
# Tier 1 (<=15min): ingest, I_delayed=0. Tier 2 (15min,6h]: ingest, I_delayed=1.
# Tier 3 (>6h): discard from training/validation (retroactive syndication edits).
REVISION_FLAG_THRESHOLD_S = 900.0
REVISION_DISCARD_THRESHOLD_S = 6 * 3600.0
REVISION_POLICY_VERSION = "revision-gate-6h-v1"

NEWS_FEATURE_NAMES: tuple[str, ...] = (
    "news_headline_count_1d",
    "news_headline_count_3d",
    "news_headline_count_7d",
    "news_negative_sentiment_mean",
    "news_positive_sentiment_mean",
    "news_sentiment_dispersion",
    "news_negative_news_intensity",
    "news_absolute_sentiment_intensity",
    "news_hours_since_latest_article",
    "news_volume_zscore",
)

# Systematic macro block (e.g. SPY): identical rolling semantics to the
# ticker block, evaluated over the same 16:00 ET NYSE session boundaries.
# Separate names keep ticker and macro signals as complementary features
# instead of substituting macro values into ticker columns.
MACRO_NEWS_FEATURE_NAMES: tuple[str, ...] = (
    "macro_headline_count_1d",
    "macro_headline_count_3d",
    "macro_headline_count_7d",
    "macro_negative_sentiment_mean",
    "macro_positive_sentiment_mean",
    "macro_sentiment_dispersion",
    "macro_negative_news_intensity",
    "macro_absolute_sentiment_intensity",
    "macro_hours_since_latest_article",
    "macro_volume_zscore",
)

CIK_MAPPING: dict[str, str] = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "GOOGL": "0001652044",
    "GOOG": "0001652044",
    "NVDA": "0001045810",
    "TSLA": "0001318605",
}


def _normalise_article(
    article: dict[str, Any],
    ticker: str,
    collected_at: str,
    default_provider: str = "alpaca",
) -> dict[str, Any] | None:
    content = article.get("content") if isinstance(article.get("content"), dict) else {}
    headline = str(
        content.get("title")
        or article.get("headline")
        or article.get("title")
        or article.get("primaryDocDescription")
        or ""
    ).strip()
    summary = str(
        content.get("summary")
        or content.get("description")
        or article.get("summary")
        or article.get("description")
        or ""
    ).strip()
    published_at = str(
        content.get("pubDate")
        or article.get("created_at")
        or article.get("published_at")
        or article.get("acceptanceDateTime")
        or article.get("pubDate")
        or ""
    ).strip()
    if not headline or not published_at:
        return None
    try:
        timestamp = datetime.fromisoformat(published_at.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None
    sentiment_target = f"{headline}. {summary}" if summary and summary != headline else headline
    scores = _sentiment.polarity_scores(sentiment_target)
    raw_symbols = article.get("symbols", [])
    symbols = sorted(
        {
            str(symbol).strip().upper()
            for symbol in (raw_symbols if isinstance(raw_symbols, list) else [])
            if str(symbol).strip()
        }
        | {ticker}
    )
    identifier = str(article.get("id") or article.get("accessionNumber") or "").strip()
    if not identifier:
        identifier = hashlib.sha256(
            f"{ticker}\0{timestamp.isoformat()}\0{headline}".encode()
        ).hexdigest()

    provider = str(
        article.get("provider_name")
        or (
            "sec_edgar"
            if "acceptanceDateTime" in article or "accessionNumber" in article
            else "yahoo"
            if isinstance(article.get("content"), dict) or "publisher" in article
            else default_provider
        )
    )
    source = str(
        (content.get("provider") or {}).get("displayName")
        or article.get("source")
        or article.get("publisher")
        or ("U.S. SEC EDGAR" if provider == "sec_edgar" else "unknown")
    )[:200]
    url = str(
        (content.get("canonicalUrl") or {}).get("url")
        or content.get("previewUrl")
        or article.get("url")
        or article.get("link")
        or ""
    )[:2000]

    return {
        "schema_version": NEWS_SCHEMA_VERSION,
        "provider": provider,
        "id": identifier,
        "ticker": ticker,
        "symbols": symbols,
        "headline": headline[:1000],
        "source": source,
        "url": url,
        "published_at": timestamp.isoformat().replace("+00:00", "Z"),
        "collected_at": collected_at,
        "provider_updated_at": article.get("updated_at"),
        "availability_basis": "retrieved_version_publication_time_unverified",
        "sentiment_pos": float(scores["pos"]),
        "sentiment_neg": float(scores["neg"]),
        "sentiment_neu": float(scores["neu"]),
        "sentiment_compound": float(scores["compound"]),
        **_revision_gate_fields(
            timestamp.isoformat().replace("+00:00", "Z"),
            article.get("updated_at"),
        ),
    }


def _parse_optional_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


def _revision_gate_fields(
    published_at: str,
    provider_updated_at: Any,
    *,
    flag_threshold_s: float = REVISION_FLAG_THRESHOLD_S,
    discard_threshold_s: float = REVISION_DISCARD_THRESHOLD_S,
) -> dict[str, Any]:
    """Derive conservative availability fields for one record.

    t_available = max(published_at, provider_updated_at); unparsable or
    missing updates fall back to published_at with zero delay. Negative
    clock-skew deltas are clamped to zero.
    """
    published = _parse_optional_timestamp(published_at)
    updated = _parse_optional_timestamp(provider_updated_at)
    if published is None:
        raise ValueError("published_at is required for revision gating")
    if updated is None or updated <= published:
        delay_s = 0.0
        available = published
    else:
        delay_s = float((updated - published).total_seconds())
        available = updated
    if delay_s > discard_threshold_s:
        tier = 3
    elif delay_s > flag_threshold_s:
        tier = 2
    else:
        tier = 1
    return {
        "revision_delay_s": float(delay_s),
        "t_available": available.isoformat().replace("+00:00", "Z"),
        "is_delayed_ingest": 1 if tier == 2 else 0,
        "revision_tier": tier,
        "revision_policy": REVISION_POLICY_VERSION,
    }


def _is_pre_gated(
    df_news: pd.DataFrame,
    *,
    flag_threshold_s: float,
    discard_threshold_s: float,
) -> bool:
    """Detect records already annotated by apply_revision_policy under defaults.

    Lets broadcast workloads (e.g. one SPY macro stream joined to hundreds of
    tickers) gate once and reuse, instead of re-deriving per-ticker delays.
    Only trusted when the caller uses the default thresholds and every record
    carries the current revision policy version.
    """
    if (
        flag_threshold_s != REVISION_FLAG_THRESHOLD_S
        or discard_threshold_s != REVISION_DISCARD_THRESHOLD_S
    ):
        return False
    required = {"t_available", "revision_tier", "revision_policy"}
    if not required.issubset(df_news.columns):
        return False
    try:
        policies = pd.Series(df_news["revision_policy"]).unique()
    except Exception:
        return False
    return len(policies) == 1 and str(policies[0]) == REVISION_POLICY_VERSION


def apply_revision_policy(
    records: list[dict[str, Any]] | pd.DataFrame,
    *,
    flag_threshold_s: float = REVISION_FLAG_THRESHOLD_S,
    discard_threshold_s: float = REVISION_DISCARD_THRESHOLD_S,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Split records into Tier 1+2 ingest set and Tier 3 discards.

    Returns (kept, diagnostics). Kept records are annotated with
    t_available / revision_delay_s / is_delayed_ingest / revision_tier and
    sorted by (t_available, id) so downstream joins implement
    session(t_available) = min{s in S | s >= t_available}.
    """
    items: list[dict[str, Any]]
    if isinstance(records, pd.DataFrame):
        items = records.to_dict(orient="records")
    else:
        items = list(records or [])
    kept: list[dict[str, Any]] = []
    tier_counts = {1: 0, 2: 0, 3: 0}
    for item in items:
        if not isinstance(item, dict) or not item.get("published_at"):
            continue
        try:
            gate = _revision_gate_fields(
                str(item["published_at"]),
                item.get("provider_updated_at"),
                flag_threshold_s=flag_threshold_s,
                discard_threshold_s=discard_threshold_s,
            )
        except ValueError:
            continue
        tier_counts[int(gate["revision_tier"])] += 1
        if int(gate["revision_tier"]) == 3:
            continue
        annotated = dict(item)
        annotated.update(gate)
        # Backfill t_available for callers that only read published_at.
        kept.append(annotated)
    kept.sort(key=lambda r: (str(r.get("t_available", r.get("published_at", ""))), str(r.get("id", ""))))
    total = sum(tier_counts.values())
    diagnostics = {
        "revision_policy": REVISION_POLICY_VERSION,
        "flag_threshold_s": flag_threshold_s,
        "discard_threshold_s": discard_threshold_s,
        "total": total,
        "kept": len(kept),
        "discarded": tier_counts[3],
        "tier_counts": tier_counts,
        "discard_rate": (tier_counts[3] / total) if total else 0.0,
    }
    return kept, diagnostics


def collect_alpaca_news(
    ticker: str,
    *,
    key_id: str,
    secret_key: str,
    start: str,
    end: str,
    client: httpx.Client | None = None,
    maximum_pages: int = 20_000,
) -> list[dict[str, Any]]:
    """Download every Alpaca news page for one ticker and a bounded interval."""
    symbol = ticker.strip().upper()
    if not symbol or not key_id or not secret_key:
        raise ValueError("Ticker and Alpaca credentials are required")
    collected_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    owned_client = client is None
    session = client or httpx.Client(timeout=30.0)
    page_token: str | None = None
    records: dict[str, dict[str, Any]] = {}
    try:
        for _page in range(maximum_pages):
            params = {
                "symbols": symbol,
                "start": start,
                "end": end,
                "sort": "asc",
                "limit": 50,
                "include_content": "false",
            }
            if page_token:
                params["page_token"] = page_token
            for attempt in range(5):
                response = session.get(
                    "https://data.alpaca.markets/v1beta1/news",
                    params=params,
                    headers={
                        "APCA-API-KEY-ID": key_id,
                        "APCA-API-SECRET-KEY": secret_key,
                        "Accept": "application/json",
                    },
                )
                if response.status_code != 429:
                    break
                time.sleep(min(2**attempt, 16))
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("Alpaca news response must be an object")
            for raw in payload.get("news", []):
                if not isinstance(raw, dict):
                    continue
                normalised = _normalise_article(
                    raw, symbol, collected_at, default_provider="alpaca"
                )
                if normalised is not None:
                    normalised["provider"] = "alpaca"
                    publication = pd.Timestamp(normalised["published_at"])
                    start_time, end_time = pd.Timestamp(start), pd.Timestamp(end)
                    if start_time.tzinfo is None:
                        start_time = start_time.tz_localize("UTC")
                    if end_time.tzinfo is None:
                        end_time = end_time.tz_localize("UTC")
                    if not start_time <= publication < end_time:
                        continue
                    records[normalised["id"]] = normalised
            next_token = payload.get("next_page_token")
            if not next_token:
                break
            if next_token == page_token:
                raise ValueError("Alpaca news pagination token did not advance")
            page_token = str(next_token)
        else:
            raise RuntimeError("Alpaca news pagination exceeded the safety limit")
    finally:
        if owned_client:
            session.close()
    return sorted(records.values(), key=lambda item: (item["published_at"], item["id"]))


def collect_public_recent_news(
    ticker: str,
    *,
    api_base: str,
    client: httpx.Client | None = None,
) -> list[dict[str, Any]]:
    """Collect recent Alpaca-backed headlines through the public context endpoint."""
    symbol = ticker.strip().upper()
    base = api_base.strip().rstrip("/")
    if not symbol or not base.startswith("https://"):
        raise ValueError("A ticker and HTTPS API base are required")
    collected_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    owned_client = client is None
    session = client or httpx.Client(timeout=30.0)
    try:
        response = session.get(f"{base}/api/v1/news", params={"ticker": symbol})
        response.raise_for_status()
        payload = response.json()
    finally:
        if owned_client:
            session.close()
    if not isinstance(payload, dict) or payload.get("status") != "available":
        return []
    records = []
    for item in payload.get("items", []):
        if not isinstance(item, dict):
            continue
        normalised = _normalise_article(item, symbol, collected_at, default_provider="alpaca")
        if normalised is not None:
            normalised["acquisition_route"] = "public_context_api"
            records.append(normalised)
    return sorted(records, key=lambda item: (item["published_at"], item["id"]))


def collect_yahoo_news(ticker: str) -> list[dict[str, Any]]:
    """Collect recent headlines for a ticker using yfinance with timestamp verification."""
    import yfinance as yf

    symbol = ticker.strip().upper()
    if not symbol:
        raise ValueError("Ticker is required")
    collected_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    ticker_obj = yf.Ticker(symbol)
    raw_news = list(ticker_obj.news or [])
    try:
        search_obj = yf.Search(symbol, news_count=25)
        if search_obj.news:
            raw_news.extend(search_obj.news)
    except Exception:
        pass
    records: dict[str, dict[str, Any]] = {}
    for item in raw_news:
        if not isinstance(item, dict):
            continue
        normalised = _normalise_article(item, symbol, collected_at, default_provider="yahoo")
        if normalised is not None:
            normalised["acquisition_route"] = "yahoo_finance_live"
            records[normalised["id"]] = normalised
    return sorted(records.values(), key=lambda item: (item["published_at"], item["id"]))


def collect_sec_edgar_filings(
    ticker: str,
    *,
    cik: str | None = None,
    client: httpx.Client | None = None,
    start: str = "2015-01-01",
    end: str = "2026-12-31",
) -> list[dict[str, Any]]:
    """Collect timestamped material corporate filings (8-K, 10-Q, 10-K) from SEC EDGAR."""
    symbol = ticker.strip().upper()
    resolved_cik = cik or CIK_MAPPING.get(symbol)
    if not resolved_cik:
        raise ValueError(f"No CIK mapping available for {symbol}")
    collected_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    owned_client = client is None
    session = client or httpx.Client(
        timeout=30.0,
        headers={"User-Agent": "StockPredictorResearch/1.0 (research@stockpredictor.local)"},
    )
    material_forms = {"8-K", "8-K/A", "10-Q", "10-Q/A", "10-K", "10-K/A", "6-K"}
    item_labels: dict[str, str] = {
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
    records: dict[str, dict[str, Any]] = {}
    try:
        url = f"https://data.sec.gov/submissions/CIK{resolved_cik}.json"
        response = session.get(url)
        response.raise_for_status()
        payload = response.json()
        recent = payload.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        acceptance_times = recent.get("acceptanceDateTime", [])
        accession_numbers = recent.get("accessionNumber", [])
        descriptions = recent.get("primaryDocDescription", [])
        report_dates = recent.get("reportDate", [])
        items_list = recent.get("items", [])

        for i in range(len(forms)):
            form = str(forms[i]).strip()
            if form not in material_forms:
                continue
            acc_time = str(acceptance_times[i] if i < len(acceptance_times) else "").strip()
            acc_num = str(accession_numbers[i] if i < len(accession_numbers) else "").strip()
            desc = str(descriptions[i] if i < len(descriptions) else form).strip() or form
            rep_date = str(report_dates[i] if i < len(report_dates) else "").strip()
            raw_item_code = str(items_list[i] if i < len(items_list) else "").strip()
            if not acc_time or not acc_num:
                continue
            if not (start <= acc_time[:10] <= end):
                continue
            item_detail = ""
            if raw_item_code:
                matched_items = [
                    item_labels[code.strip()]
                    for code in raw_item_code.split(",")
                    if code.strip() in item_labels
                ]
                if matched_items:
                    item_detail = f" ({', '.join(matched_items)})"
            if desc != form and desc:
                headline = f"{symbol} Form {form}: {desc}{item_detail}"
            elif item_detail:
                headline = f"{symbol} Form {form}{item_detail}"
            else:
                headline = f"{symbol} Form {form} filing"
            if rep_date:
                headline += f" (period: {rep_date})"
            raw_record = {
                "id": f"sec-{acc_num}",
                "accessionNumber": f"sec-{acc_num}",
                "acceptanceDateTime": acc_time,
                "primaryDocDescription": headline,
                "symbols": [symbol],
                "source": "U.S. SEC EDGAR",
                "url": f"https://www.sec.gov/edgar/browse/?CIK={resolved_cik}",
                "provider_name": "sec_edgar",
            }
            normalised = _normalise_article(
                raw_record, symbol, collected_at, default_provider="sec_edgar"
            )
            if normalised is not None:
                normalised["acquisition_route"] = "sec_edgar_material"
                records[normalised["id"]] = normalised
    finally:
        if owned_client:
            session.close()
    return sorted(records.values(), key=lambda item: (item["published_at"], item["id"]))


def load_news_archive(path: str | Path) -> list[dict[str, Any]]:
    """Read and validate deduplicated JSONL news archive."""
    archive_path = Path(path)
    if not archive_path.is_file():
        raise FileNotFoundError(f"News archive not found: {archive_path}")
    records = []
    for line_num, line in enumerate(archive_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict) or "published_at" not in item or "headline" not in item:
            raise ValueError(f"Invalid record at {archive_path}:{line_num}")
        records.append(item)
    return sorted(records, key=lambda item: (item["published_at"], str(item.get("id", ""))))


def merge_news_archive(path: str | Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Atomically merge deduplicated records and write a checksummed manifest."""
    archive_path = Path(path)
    existing: dict[tuple[str, str], dict[str, Any]] = {}
    if archive_path.is_file():
        for line in archive_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            existing[(str(item.get("provider", "unknown")), str(item["id"]))] = item
    for item in records:
        key = (str(item.get("provider", "unknown")), str(item["id"]))
        if key in existing and existing[key] != item:
            raise ValueError("Conflicting news record: preserve revisions in a separate archive")
        existing[key] = item
    ordered = sorted(existing.values(), key=lambda item: (item["published_at"], item["id"]))
    payload = "".join(
        json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n" for item in ordered
    )
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive_path.with_suffix(archive_path.suffix + ".tmp")
    temporary.write_bytes(payload.encode("utf-8"))
    temporary.replace(archive_path)
    providers = sorted(
        {
            str(item.get("provider", "unknown")).strip()
            for item in ordered
            if item.get("provider") and str(item.get("provider", "")).strip()
        }
    )
    provider_str = ",".join(providers) if providers else "alpaca"
    manifest = {
        "schema_version": NEWS_SCHEMA_VERSION,
        "provider": provider_str,
        "archive": archive_path.name,
        "article_count": len(ordered),
        "published_start": ordered[0]["published_at"] if ordered else None,
        "published_end": ordered[-1]["published_at"] if ordered else None,
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "hash_convention": "sha256_exact_utf8_lf_bytes_v1",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    manifest_path = archive_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _resolve_session_closes(sessions: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Efficiently resolve UTC session close timestamps for an entire DatetimeIndex."""
    try:
        import pandas_market_calendars as mcal

        cal = mcal.get_calendar("NYSE")
        start_str = sessions.min().strftime("%Y-%m-%d")
        end_str = sessions.max().strftime("%Y-%m-%d")
        sched = cal.schedule(start_date=start_str, end_date=end_str)
        closes: list[pd.Timestamp] = []
        for s_date in sessions:
            date_str = s_date.strftime("%Y-%m-%d")
            if date_str in sched.index and "market_close" in sched.columns:
                close_val = sched.loc[date_str, "market_close"]
                closes.append(pd.Timestamp(close_val).tz_convert("UTC").tz_localize(None))
            else:
                ts_et = s_date.tz_localize("America/New_York").replace(hour=16, minute=0, second=0)
                closes.append(ts_et.tz_convert("UTC").tz_localize(None))
        return closes
    except Exception:
        closes = []
        for s_date in sessions:
            ts_et = s_date.tz_localize("America/New_York").replace(hour=16, minute=0, second=0)
            closes.append(ts_et.tz_convert("UTC").tz_localize(None))
        return closes


def validate_news_archive(
    records: list[dict[str, Any]],
    sessions: pd.DatetimeIndex,
    min_article_count: int = 1,
    *,
    enforce_revision_gate: bool = True,
    flag_threshold_s: float = REVISION_FLAG_THRESHOLD_S,
    discard_threshold_s: float = REVISION_DISCARD_THRESHOLD_S,
) -> dict[str, Any]:
    """Compute point-in-time coverage diagnostics for an evaluated news corpus.

    When enforce_revision_gate is True (default), coverage uses conservative
    t_available = max(published_at, provider_updated_at) and excludes Tier 3
    retroactive revisions (delay > discard_threshold_s). This implements
    session(t_available) = min{s in S | s >= t_available}.
    """
    sessions = pd.DatetimeIndex(sessions)
    if sessions.tz is not None:
        sessions = sessions.tz_convert(None)

    raw_items = [r for r in (records or []) if isinstance(r, dict) and r.get("published_at")]
    gate_diagnostics: dict[str, Any] | None = None
    if enforce_revision_gate:
        kept, gate_diagnostics = apply_revision_policy(
            raw_items,
            flag_threshold_s=flag_threshold_s,
            discard_threshold_s=discard_threshold_s,
        )
        active_times = [
            str(r.get("t_available") or r["published_at"]) for r in kept
        ]
        tier_counts = gate_diagnostics["tier_counts"]
    else:
        active_times = [str(r["published_at"]) for r in raw_items]
        tier_counts = None
    if not active_times:
        empty: dict[str, Any] = {
            "article_count": 0,
            "published_start": None,
            "published_end": None,
            "coverage_1d": 0.0,
            "coverage_3d": 0.0,
            "coverage_7d": 0.0,
            "is_valid": False,
            "reason": "News record list is empty",
        }
        if gate_diagnostics is not None:
            empty["revision_gate"] = gate_diagnostics
        return empty

    cutoff_timestamps = _resolve_session_closes(sessions)
    pub_dt = pd.to_datetime(active_times, utc=True).tz_localize(None).sort_values()
    pub_times = pub_dt.to_numpy(dtype="datetime64[ns]")

    has_1d: list[bool] = []
    has_3d: list[bool] = []
    has_7d: list[bool] = []

    for i in range(len(sessions)):
        cutoff = np.datetime64(cutoff_timestamps[i], "ns")
        idx_end = int(np.searchsorted(pub_times, cutoff, side="right"))
        idx_1d = int(np.searchsorted(pub_times, cutoff - np.timedelta64(24, "h"), side="right"))
        idx_3d = int(np.searchsorted(pub_times, cutoff - np.timedelta64(72, "h"), side="right"))
        idx_7d = int(np.searchsorted(pub_times, cutoff - np.timedelta64(168, "h"), side="right"))

        has_1d.append(idx_end > idx_1d)
        has_3d.append(idx_end > idx_3d)
        has_7d.append(idx_end > idx_7d)

    cov_1d = float(np.mean(has_1d)) if has_1d else 0.0
    cov_3d = float(np.mean(has_3d)) if has_3d else 0.0
    cov_7d = float(np.mean(has_7d)) if has_7d else 0.0

    is_valid = len(active_times) >= min_article_count
    reason = (
        "Valid timestamped corpus"
        if is_valid
        else f"Article count ({len(active_times)}) below minimum requirement ({min_article_count})"
    )

    result: dict[str, Any] = {
        "article_count": len(active_times),
        "published_start": min(active_times),
        "published_end": max(active_times),
        "coverage_1d": cov_1d,
        "coverage_3d": cov_3d,
        "coverage_7d": cov_7d,
        "is_valid": is_valid,
        "reason": reason,
    }
    if gate_diagnostics is not None:
        result["revision_gate"] = gate_diagnostics
        result["tier_counts"] = tier_counts
    return result


def build_causal_news_features(
    sessions: pd.DatetimeIndex,
    ticker: str,
    news_events: list[dict[str, Any]] | pd.DataFrame,
    *,
    enforce_revision_gate: bool = True,
    flag_threshold_s: float = REVISION_FLAG_THRESHOLD_S,
    discard_threshold_s: float = REVISION_DISCARD_THRESHOLD_S,
) -> pd.DataFrame:
    """Extract causal point-in-time news features with strict exchange market close cutoff.

    Cutoff at trading session t is resolved from the NYSE exchange calendar.
    Only articles with t_available <= cutoff participate in session t, where
    t_available = max(published_at, provider_updated_at). Tier 3 records
    (delay > discard_threshold_s, default 6h) are discarded; Tier 2 records
    carry is_delayed_ingest=1 at record level. A 15:30 ET publish with a
    16:30 ET update therefore shifts to the next session rather than leaking
    backward. NEWS_FEATURE_NAMES is unchanged for checkpoint compatibility.
    """
    sessions = pd.DatetimeIndex(sessions)
    if sessions.tz is not None:
        sessions = sessions.tz_convert(None)

    out = pd.DataFrame(index=sessions)
    if news_events is None:
        raise ValueError("news_events are required; synthetic news is prohibited.")

    df_news = pd.DataFrame(news_events) if isinstance(news_events, list) else news_events.copy()
    if df_news.empty or "published_at" not in df_news.columns:
        for col in NEWS_FEATURE_NAMES:
            out[col] = 168.0 if col == "news_hours_since_latest_article" else 0.0
        return out

    if enforce_revision_gate:
        if _is_pre_gated(
            df_news,
            flag_threshold_s=flag_threshold_s,
            discard_threshold_s=discard_threshold_s,
        ):
            df_news = df_news[
                pd.to_numeric(df_news["revision_tier"], errors="coerce").fillna(3).astype(int)
                != 3
            ]
            if df_news.empty:
                for col in NEWS_FEATURE_NAMES:
                    out[col] = 168.0 if col == "news_hours_since_latest_article" else 0.0
                return out
        else:
            gated_records, _gate_diag = apply_revision_policy(
                df_news.to_dict(orient="records"),
                flag_threshold_s=flag_threshold_s,
                discard_threshold_s=discard_threshold_s,
            )
            if not gated_records:
                for col in NEWS_FEATURE_NAMES:
                    out[col] = 168.0 if col == "news_hours_since_latest_article" else 0.0
                return out
            df_news = pd.DataFrame(gated_records)
        # Point-in-time join key: conservative availability, not creation time.
        df_news["available_at"] = pd.to_datetime(
            df_news["t_available"], utc=True
        ).dt.tz_localize(None)
    else:
        df_news["available_at"] = pd.to_datetime(
            df_news["published_at"], utc=True
        ).dt.tz_localize(None)
    df_news = df_news.sort_values("available_at").reset_index(drop=True)

    pub_times = df_news["available_at"].to_numpy(dtype="datetime64[ns]")
    neg_scores = (
        df_news["sentiment_neg"].to_numpy(dtype=float)
        if "sentiment_neg" in df_news
        else np.zeros(len(df_news))
    )
    pos_scores = (
        df_news["sentiment_pos"].to_numpy(dtype=float)
        if "sentiment_pos" in df_news
        else np.zeros(len(df_news))
    )
    compound_scores = (
        df_news["sentiment_compound"].to_numpy(dtype=float)
        if "sentiment_compound" in df_news
        else (pos_scores - neg_scores)
    )

    counts_1d: list[float] = []
    counts_3d: list[float] = []
    counts_7d: list[float] = []
    neg_means: list[float] = []
    pos_means: list[float] = []
    sent_disps: list[float] = []
    neg_intens: list[float] = []
    abs_intens: list[float] = []
    hours_since: list[float] = []

    cutoff_timestamps = _resolve_session_closes(sessions)

    for i, _s_date in enumerate(sessions):
        cutoff_ts = cutoff_timestamps[i]
        cutoff = np.datetime64(cutoff_ts, "ns")
        cutoff_1d = cutoff - np.timedelta64(24, "h")
        cutoff_3d = cutoff - np.timedelta64(72, "h")
        cutoff_7d = cutoff - np.timedelta64(168, "h")

        idx_end = int(np.searchsorted(pub_times, cutoff, side="right"))
        idx_start_1d = int(np.searchsorted(pub_times[:idx_end], cutoff_1d, side="right"))
        idx_start_3d = int(np.searchsorted(pub_times[:idx_end], cutoff_3d, side="right"))
        idx_start_7d = int(np.searchsorted(pub_times[:idx_end], cutoff_7d, side="right"))

        c1 = float(idx_end - idx_start_1d)
        c3 = float(idx_end - idx_start_3d)
        c7 = float(idx_end - idx_start_7d)

        counts_1d.append(c1)
        counts_3d.append(c3)
        counts_7d.append(c7)

        if c3 > 0:
            sub_neg = neg_scores[idx_start_3d:idx_end]
            sub_pos = pos_scores[idx_start_3d:idx_end]
            sub_comp = compound_scores[idx_start_3d:idx_end]
            n_mean = float(np.mean(sub_neg))
            p_mean = float(np.mean(sub_pos))
            s_disp = float(np.std(sub_comp, ddof=1)) if c3 > 1 else 0.0
            a_int = float(np.mean(np.abs(sub_comp)))
            n_int = float((c3 / 3.0) * n_mean)
        else:
            n_mean = 0.0
            p_mean = 0.0
            s_disp = 0.0
            a_int = 0.0
            n_int = 0.0

        neg_means.append(n_mean)
        pos_means.append(p_mean)
        sent_disps.append(s_disp)
        neg_intens.append(n_int)
        abs_intens.append(a_int)

        if idx_end > 0:
            latest_time = pub_times[idx_end - 1]
            elapsed_h = float((cutoff - latest_time) / np.timedelta64(1, "h"))
            hours_since.append(min(max(elapsed_h, 0.0), 168.0))
        else:
            hours_since.append(168.0)

    out["news_headline_count_1d"] = counts_1d
    out["news_headline_count_3d"] = counts_3d
    out["news_headline_count_7d"] = counts_7d
    out["news_negative_sentiment_mean"] = neg_means
    out["news_positive_sentiment_mean"] = pos_means
    out["news_sentiment_dispersion"] = sent_disps
    out["news_negative_news_intensity"] = neg_intens
    out["news_absolute_sentiment_intensity"] = abs_intens
    out["news_hours_since_latest_article"] = hours_since

    c1_series = pd.Series(counts_1d, index=sessions)
    roll_mean = c1_series.rolling(22, min_periods=1).mean()
    roll_std = c1_series.rolling(22, min_periods=1).std().fillna(1.0)
    out["news_volume_zscore"] = ((c1_series - roll_mean) / np.maximum(roll_std, 1.0)).to_numpy()

    return out


def build_macro_news_features(
    sessions: pd.DatetimeIndex,
    macro_events: list[dict[str, Any]] | pd.DataFrame,
    *,
    enforce_revision_gate: bool = True,
    flag_threshold_s: float = REVISION_FLAG_THRESHOLD_S,
    discard_threshold_s: float = REVISION_DISCARD_THRESHOLD_S,
) -> pd.DataFrame:
    """Build the systematic macro block (e.g. SPY) for one session index.

    Identical rolling semantics and 6-hour revision gating to the ticker
    block, evaluated over the same 16:00 ET NYSE session boundaries, with
    output columns under MACRO_NEWS_FEATURE_NAMES. Accepts pre-gated
    records (see _is_pre_gated) so one macro stream can be broadcast to
    hundreds of tickers after a single gating pass. US-session semantics
    only; do not join same-date macro output to UK closes without a lag.
    """
    frame = build_causal_news_features(
        sessions,
        ticker="MACRO",
        news_events=macro_events,
        enforce_revision_gate=enforce_revision_gate,
        flag_threshold_s=flag_threshold_s,
        discard_threshold_s=discard_threshold_s,
    )
    rename = dict(zip(NEWS_FEATURE_NAMES, MACRO_NEWS_FEATURE_NAMES))
    return frame.rename(columns=rename).loc[:, list(MACRO_NEWS_FEATURE_NAMES)]
