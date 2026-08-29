"""v8 historical-news provenance and numeric-fallback manifests.

The underlying event store is the immutable point-in-time snapshot in
``news_snapshot.py``. This module binds that store to the v8 universe, market
snapshot, alias map, provider coverage, and taxonomy. A complete snapshot is
only *eligible for ablation*; it is never called certified until a frozen
market-plus-news candidate passes the paired promotion and sealed-test gates.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from .gdelt_snapshot import load_ticker_aliases
from .news_snapshot import load_news_snapshot
from .universe_v8 import verify_universe_manifest
from .v8_protocol import V8_NEWS_TAXONOMY, V8_NEWS_TAXONOMY_VERSION

V8_NEWS_SNAPSHOT_SCHEMA = 2
V8_NEWS_FEATURE_LOOKBACK_DAYS = 20
V8_NEWS_STATUS_NUMERIC = "not_certified"
V8_NEWS_STATUS_READY = "snapshot_ready_uncertified"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _canonical_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        {key: value for key, value in payload.items() if key != "sha256"},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def news_snapshot_id(provider: str, content_checksum: str) -> str:
    raw = f"{provider}:{content_checksum}".encode()
    return f"news-{hashlib.sha256(raw).hexdigest()[:16]}"


def build_numeric_fallback_news_snapshot(
    *,
    provider: str = "none",
    license_id: str = "not_applicable_numeric_only",
) -> dict[str, Any]:
    """Build the honest numeric-only snapshot manifest (news not certified)."""

    empty_checksum = hashlib.sha256(b"no_articles").hexdigest()
    manifest: dict[str, Any] = {
        "schema_version": V8_NEWS_SNAPSHOT_SCHEMA,
        "snapshot_id": news_snapshot_id(provider, empty_checksum),
        "provider": provider,
        "license_id": license_id,
        "article_count": 0,
        "deduped_article_count": 0,
        "coverage_start": None,
        "coverage_end_exclusive": None,
        "missing_archive_dates": [],
        "content_checksum": f"sha256:{empty_checksum}",
        "news_enabled": False,
        "news_status": V8_NEWS_STATUS_NUMERIC,
        "coverage_complete": False,
        "model_certified": False,
        "note": (
            "Historical news is absent. This identity is numeric-only and must "
            "never be described as a news-enhanced model."
        ),
    }
    manifest["sha256"] = _canonical_digest(manifest)
    return manifest


def _market_coverage(market_manifest: dict[str, Any]) -> tuple[pd.Timestamp, pd.Timestamp]:
    tickers = market_manifest.get("tickers")
    if not isinstance(tickers, dict) or not tickers:
        raise ValueError("v8 market manifest has no ticker coverage")
    try:
        starts = [pd.Timestamp(metadata["start"], tz="UTC") for metadata in tickers.values()]
        ends = [pd.Timestamp(metadata["end"], tz="UTC") for metadata in tickers.values()]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("v8 market coverage dates are invalid") from error
    return min(starts), max(ends)


def build_v8_news_manifest(
    *,
    news_snapshot_dir: Path,
    universe_manifest: dict[str, Any],
    market_manifest: dict[str, Any],
    ticker_aliases_path: Path,
    provider_license_id: str,
    allow_provider_gaps: bool = False,
) -> dict[str, Any]:
    """Bind a verified point-in-time event lake to the v8 research identity."""

    universe = verify_universe_manifest(universe_manifest)
    if not isinstance(provider_license_id, str) or not provider_license_id.strip():
        raise ValueError("provider_license_id is required")
    v8_market = market_manifest.get("v8_market")
    if not isinstance(v8_market, dict):
        raise ValueError("news binding requires a v8 market manifest")
    if v8_market.get("universe_manifest_sha256") != universe["sha256"]:
        raise ValueError("market and universe identities differ")
    if v8_market.get("coverage_certifiable") is not True:
        raise ValueError("news cannot be bound to a diagnostic-only market snapshot")

    events, base = load_news_snapshot(news_snapshot_dir)
    aliases = load_ticker_aliases(ticker_aliases_path)
    universe_tickers = {str(member["ticker"]).strip().upper() for member in universe["members"]}
    missing_aliases = sorted(universe_tickers - set(aliases))
    extra_aliases = sorted(set(aliases) - universe_tickers)
    if missing_aliases or extra_aliases:
        raise ValueError(
            f"ticker alias coverage mismatch: missing={missing_aliases}, extra={extra_aliases}"
        )
    unknown_event_tickers = sorted(
        {ticker for event in events for ticker in event.tickers if ticker not in universe_tickers}
    )
    if unknown_event_tickers:
        raise ValueError(
            "news events reference securities outside the v8 universe: "
            + ", ".join(unknown_event_tickers)
        )

    provenance = base.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("news snapshot provenance is missing")
    raw_gaps = provenance.get("missing_archive_dates", [])
    if not isinstance(raw_gaps, list) or any(not isinstance(item, str) for item in raw_gaps):
        raise ValueError("news provider gap metadata is invalid")
    missing_archive_dates = sorted(set(raw_gaps))
    market_start, market_end = _market_coverage(market_manifest)
    required_start = market_start - pd.Timedelta(days=V8_NEWS_FEATURE_LOOKBACK_DAYS)
    required_end = market_end + pd.Timedelta(days=1)
    coverage_start = pd.Timestamp(base["coverage_start"])
    coverage_end = pd.Timestamp(base["coverage_end_exclusive"])
    if coverage_start.tzinfo is None or coverage_end.tzinfo is None:
        raise ValueError("news snapshot coverage must be timezone-aware")
    coverage_start = coverage_start.tz_convert("UTC")
    coverage_end = coverage_end.tz_convert("UTC")

    reasons: list[str] = []
    if coverage_start > required_start:
        reasons.append("initial_news_lookback_incomplete")
    if coverage_end <= required_end:
        reasons.append("news_coverage_ends_before_market_snapshot")
    if missing_archive_dates:
        reasons.append("provider_archive_gaps")
    ambiguous = sum(event.eligible_at is None for event in events)
    eligible = len(events) - ambiguous
    coverage_complete = not reasons
    if reasons and not allow_provider_gaps:
        raise ValueError("v8 news snapshot is incomplete: " + "; ".join(reasons))

    manifest: dict[str, Any] = {
        "schema_version": V8_NEWS_SNAPSHOT_SCHEMA,
        "base_snapshot_id": base["snapshot_id"],
        "provider": base["provider"],
        "provider_license_id": provider_license_id,
        "universe_manifest_sha256": universe["sha256"],
        "market_panel_checksum": market_manifest["pooled_checksum"],
        "ticker_aliases_sha256": _sha256_file(ticker_aliases_path),
        "content_checksum": "sha256:" + str(base["events_sha256"]),
        "provenance_checksum": "sha256:" + str(base["provenance_sha256"]),
        "article_count": int(base["event_count"]),
        "deduped_article_count": len({event.cluster_id for event in events}),
        "eligible_event_count": eligible,
        "quarantined_ambiguous_timestamp_count": ambiguous,
        "coverage_start": coverage_start.isoformat(),
        "coverage_end_exclusive": coverage_end.isoformat(),
        "required_coverage_start": required_start.isoformat(),
        "required_coverage_end_exclusive": required_end.isoformat(),
        "missing_archive_dates": missing_archive_dates,
        "coverage_complete": coverage_complete,
        "coverage_reasons": reasons,
        "news_enabled": True,
        "news_status": V8_NEWS_STATUS_READY,
        "model_certified": False,
        "taxonomy_version": V8_NEWS_TAXONOMY_VERSION,
        "taxonomy": list(V8_NEWS_TAXONOMY),
        "feature_lookback_days": V8_NEWS_FEATURE_LOOKBACK_DAYS,
        "available_at_policy": "max(published_at,first_seen_at); date_only_next_utc_day",
    }
    manifest["snapshot_id"] = news_snapshot_id(str(base["provider"]), str(base["events_sha256"]))
    manifest["sha256"] = _canonical_digest(manifest)
    return manifest


def verify_v8_news_manifest(
    payload: object,
    *,
    news_snapshot_dir: Path,
    universe_manifest: dict[str, Any],
    market_manifest: dict[str, Any],
    ticker_aliases_path: Path,
) -> dict[str, Any]:
    """Rebuild a persisted v8 news manifest and compare every bound field."""

    if not isinstance(payload, dict):
        raise ValueError("v8 news manifest must be a JSON object")
    if payload.get("news_enabled") is not True:
        raise ValueError("use the numeric fallback verifier for a no-news manifest")
    expected = build_v8_news_manifest(
        news_snapshot_dir=news_snapshot_dir,
        universe_manifest=universe_manifest,
        market_manifest=market_manifest,
        ticker_aliases_path=ticker_aliases_path,
        provider_license_id=str(payload.get("provider_license_id", "")),
        allow_provider_gaps=payload.get("coverage_complete") is not True,
    )
    normalized = json.loads(json.dumps(payload, sort_keys=True))
    if normalized != expected:
        raise ValueError("v8 news manifest content or checksum does not match")
    return expected


def _write_manifest_atomic(path: Path, manifest: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"news manifest already exists at {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=".news-v8-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def write_numeric_fallback_news_snapshot(out_dir: Path, **kwargs: Any) -> Path:
    """Atomically write the numeric fallback manifest."""

    return _write_manifest_atomic(
        out_dir / "news-v8-manifest.json",
        build_numeric_fallback_news_snapshot(**kwargs),
    )


def write_v8_news_manifest(out_dir: Path, **kwargs: Any) -> Path:
    """Atomically write a verified news snapshot binding without article text."""

    return _write_manifest_atomic(
        out_dir / "news-v8-manifest.json",
        build_v8_news_manifest(**kwargs),
    )
