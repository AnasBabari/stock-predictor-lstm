"""Point-in-time four-market universe builder for v8.

This module is deliberately separate from panel fetching. It defines
how a universe is *declared*, *deduplicated*, *versioned*, and
*checksummed* before any panel download. No feature/target code belongs here.

The builder is deterministic given a seed and source snapshots; the
resulting manifest is the only legitimate identifier for the universe.
Ticker-alone is never a primary key.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

# Deterministic selection seed for the initial v8 cohort
V8_UNIVERSE_SEED = 42

# Allowed MIC / exchange identifiers for v8 (see docs/VOLATILITY_V8_PREREGISTRATION.md)
V8_EXCHANGE_MICS: dict[str, str] = {
    "SP500": "SP500_PIT",  # not an exchange but a constituent list
    "NASDAQ": "XNAS",
    "NYSE": "XNYS",
    "LSE": "XLON",
}

V8_EXCLUDED_SECURITY_TYPES = {
    "ETF",
    "ETN",
    "WARRANT",
    "PREFERRED",
    "RIGHT",
    "FUND",
    "ADR",  # unless explicitly preregistered
    "GDR",
}


@dataclass(frozen=True)
class UniverseMember:
    """One security’s point-in-time identity."""

    security_id: str  # stable provider id or ISIN:FIGI:CIK composite
    ticker: str
    company_name: str
    isin: str | None
    figi: str | None
    cik: str | None
    primary_exchange: str  # MIC
    currency: str
    timezone: str
    sector: str | None
    industry: str | None
    security_type: str  # e.g. COMMON
    membership_start: str | None  # YYYY-MM-DD, None = beginning of history
    membership_end: str | None  # None = still member
    source: str
    source_snapshot_id: str
    required_history_sessions: int = 756
    point_in_time_liquidity_ok: bool = True


@dataclass(frozen=True)
class UniverseManifest:
    manifest_version: str = "universe-v8-v1"
    protocol_version: str = "global-volatility-distribution-v8-news-transfer"
    created_at: str | None = None
    seed: int = V8_UNIVERSE_SEED
    members: tuple[UniverseMember, ...] = field(default_factory=tuple)
    source_checksums: dict[str, str] = field(default_factory=dict)
    selection_policy: dict[str, Any] = field(default_factory=dict)
    total_members: int = 0
    per_exchange_counts: dict[str, int] = field(default_factory=dict)
    per_type_counts: dict[str, int] = field(default_factory=dict)
    sha256: str | None = None


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_member(member: UniverseMember) -> dict[str, Any]:
    # Deterministic field order for content addressing
    d = asdict(member)
    # Normalize ticker/ISIN casing
    d["ticker"] = d["ticker"].upper()
    if d.get("isin"):
        d["isin"] = str(d["isin"]).upper()
    return d


def _manifest_canonical_bytes(manifest: dict[str, Any]) -> bytes:
    # Exclude sha256 field itself from the digest
    payload = {k: v for k, v in manifest.items() if k != "sha256"}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def build_universe_manifest(
    members: list[UniverseMember],
    *,
    source_checksums: dict[str, str] | None = None,
    selection_policy: dict[str, Any] | None = None,
    seed: int = V8_UNIVERSE_SEED,
    protocol_version: str = "global-volatility-distribution-v8-news-transfer",
) -> dict[str, Any]:
    """Build a deterministic, content-addressed universe manifest (no I/O)."""

    if not members:
        raise ValueError("universe requires at least one member")
    # Validate exchange mics and dedupe by security_id (not ticker)
    seen_ids: set[str] = set()
    per_exchange: dict[str, int] = {}
    per_type: dict[str, int] = {}
    for m in members:
        if not m.security_id or not m.ticker:
            raise ValueError("member missing security_id or ticker")
        if m.security_id in seen_ids:
            raise ValueError(f"duplicate security_id {m.security_id}")
        seen_ids.add(m.security_id)
        if m.security_type.upper() in V8_EXCLUDED_SECURITY_TYPES:
            raise ValueError(f"excluded security type {m.security_type} for {m.ticker}")
        if m.primary_exchange not in set(V8_EXCHANGE_MICS.values()).union({"SP500_PIT"}):
            raise ValueError(f"unknown exchange MIC {m.primary_exchange} for {m.ticker}")
        per_exchange[m.primary_exchange] = per_exchange.get(m.primary_exchange, 0) + 1
        per_type[m.security_type] = per_type.get(m.security_type, 0) + 1
        # Validate dates
        for field_name in ("membership_start", "membership_end"):
            value = getattr(m, field_name)
            if value is not None:
                try:
                    date.fromisoformat(value)
                except ValueError as error:
                    raise ValueError(f"invalid {field_name} {value} for {m.ticker}") from error

    canonical_members = sorted(
        (_canonical_member(m) for m in members), key=lambda x: x["security_id"]
    )
    manifest: dict[str, Any] = {
        "manifest_version": "universe-v8-v1",
        "protocol_version": protocol_version,
        "seed": seed,
        "members": canonical_members,
        "total_members": len(canonical_members),
        "per_exchange_counts": dict(sorted(per_exchange.items())),
        "per_type_counts": dict(sorted(per_type.items())),
        "source_checksums": dict(sorted((source_checksums or {}).items())),
        "selection_policy": dict(sorted((selection_policy or {}).items())),
    }
    digest = _sha256_bytes(_manifest_canonical_bytes(manifest))
    manifest["sha256"] = digest
    return manifest


def write_universe_manifest(out_dir: Path, members: list[UniverseMember], **kwargs: Any) -> Path:
    """Atomically write ``universe-v8-manifest.json``; refuses overwrites."""
    manifest = build_universe_manifest(members, **kwargs)
    out_dir.mkdir(parents=True, exist_ok=False)
    target = out_dir / "universe-v8-manifest.json"
    if target.exists():
        raise FileExistsError(f"universe manifest already exists at {target} – immutable")
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Verify
    written = json.loads(target.read_text(encoding="utf-8"))
    if written.get("sha256") != manifest["sha256"]:
        raise RuntimeError("universe manifest checksum mismatch after write")
    return target


def initial_v8_selection_policy(seed: int = V8_UNIVERSE_SEED) -> dict[str, Any]:
    """Recommended initial v8 cohort policy (preregistered, deterministic)."""
    return {
        "description": "All PIT S&P500 + liquidity-stratified Nasdaq/NYSE/LSE, >=25-50 per exchange",
        "seed": seed,
        "sp500_mode": "point_in_time_membership",
        "nasdaq": {
            "primary_mic": V8_EXCHANGE_MICS["NASDAQ"],
            "security_type": "COMMON",
            "exclude_types": sorted(V8_EXCLUDED_SECURITY_TYPES),
            "min_history_sessions": 756,
            "liquidity_filter": "point_in_time_median_dollar_volume_top_quintile",
            "include_delisted_where_available": True,
        },
        "nyse": {
            "primary_mic": V8_EXCHANGE_MICS["NYSE"],
            "security_type": "COMMON",
            "exclude_types": sorted(V8_EXCLUDED_SECURITY_TYPES),
            "min_history_sessions": 756,
            "liquidity_filter": "point_in_time_median_dollar_volume_top_quintile",
            "include_delisted_where_available": True,
        },
        "lse": {
            "primary_mic": V8_EXCHANGE_MICS["LSE"],
            "security_type": "COMMON",
            "exclude_types_sorted": sorted(V8_EXCLUDED_SECURITY_TYPES),
            "currency": "GBX",
            "timezone": "Europe/London",
            "include_secondary_listings": False,
        },
        "holdout_seed": seed,
        "holdout_fraction": 0.20,
        "required_holdouts": ["NMM", "MSFT"],
        "dedupe_keys": ["isin", "figi", "cik", "provider_security_id"],
    }
