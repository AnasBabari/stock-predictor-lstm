"""Point-in-time four-market universe builder for v8.

This module is deliberately separate from panel fetching. It defines
how a universe is *declared*, *deduplicated*, *versioned*, and
*checksummed* before any panel download. No feature/target code belongs here.

The builder is deterministic given a seed and source snapshots; the
resulting manifest is the only legitimate identifier for the universe.
Ticker-alone is never a primary key. Index membership is never an
exchange MIC — see ``primary_exchange_mic`` vs ``index_memberships``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

# Deterministic selection seed for the initial v8 cohort
V8_UNIVERSE_SEED = 42

# Allowed MIC / exchange identifiers for v8 (see docs/VOLATILITY_V8_PREREGISTRATION.md)
V8_EXCHANGE_MICS: dict[str, str] = {
    "NASDAQ": "XNAS",
    "NYSE": "XNYS",
    "LSE": "XLON",
}

V8_INDEX_IDS: dict[str, str] = {
    "SP500": "SP500",
}

V8_ALLOWED_SECURITY_TYPES = {
    "COMMON",
    "ORDINARY_SHARE",
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

# Valid currencies for v8 (GBP/GBX distinguished for LSE)
V8_VALID_CURRENCIES = {"USD", "GBP", "GBX", "GBp"}

# Valid timezones for v8
V8_VALID_TIMEZONES = {"America/New_York", "Europe/London", "UTC"}

# Minimum counts per exchange for a certifiable four-market model
V8_MIN_PER_EXCHANGE = 25
V8_MIN_HOLDOUTS = 2
V8_MIN_SECTOR_COVERAGE = 3  # at least 3 distinct sectors


@dataclass(frozen=True)
class UniverseMember:
    """One security's point-in-time identity.

    ``primary_exchange_mic`` is the listing venue (XNAS/XNYS/XLON).
    S&P 500 membership is stored in ``index_memberships`` and never in
    ``primary_exchange_mic``. ``SP500_PIT`` is not a MIC.
    """

    security_id: str  # stable provider id or ISIN:FIGI:CIK composite
    ticker: str
    company_name: str
    isin: str | None
    figi: str | None
    cik: str | None
    primary_exchange_mic: str  # MIC: XNAS, XNYS, XLON
    index_memberships: tuple[dict[str, str], ...] = field(default_factory=tuple)
    currency: str = "USD"
    timezone: str = "America/New_York"
    sector: str | None = None
    industry: str | None = None
    security_type: str = "COMMON"  # must be in V8_ALLOWED_SECURITY_TYPES
    membership_start: str | None = None  # YYYY-MM-DD, None = beginning of history
    membership_end: str | None = None  # None = still member
    source: str = ""
    source_snapshot_id: str = ""
    required_history_sessions: int = 756
    point_in_time_liquidity_ok: bool = True

    # Backwards compat: allow ``primary_exchange`` as alias for ``primary_exchange_mic``
    def __init__(
        self,
        security_id: str,
        ticker: str,
        company_name: str,
        isin: str | None,
        figi: str | None,
        cik: str | None,
        primary_exchange_mic: str | None = None,
        index_memberships: tuple[dict[str, str], ...] | list[dict[str, str]] | None = None,
        currency: str = "USD",
        timezone: str = "America/New_York",
        sector: str | None = None,
        industry: str | None = None,
        security_type: str = "COMMON",
        membership_start: str | None = None,
        membership_end: str | None = None,
        source: str = "",
        source_snapshot_id: str = "",
        required_history_sessions: int = 756,
        point_in_time_liquidity_ok: bool = True,
        # deprecated alias
        primary_exchange: str | None = None,
    ) -> None:
        # Resolve MIC via explicit or alias
        mic = primary_exchange_mic if primary_exchange_mic is not None else primary_exchange
        if mic is None:
            raise ValueError("primary_exchange_mic is required")
        # Normalize index_memberships
        if index_memberships is None:
            idx = ()
        elif isinstance(index_memberships, list):
            idx = tuple(index_memberships)
        else:
            idx = tuple(index_memberships)
        object.__setattr__(self, "security_id", security_id)
        object.__setattr__(self, "ticker", ticker)
        object.__setattr__(self, "company_name", company_name)
        object.__setattr__(self, "isin", isin)
        object.__setattr__(self, "figi", figi)
        object.__setattr__(self, "cik", cik)
        object.__setattr__(self, "primary_exchange_mic", mic)
        object.__setattr__(self, "index_memberships", idx)
        object.__setattr__(self, "currency", currency)
        object.__setattr__(self, "timezone", timezone)
        object.__setattr__(self, "sector", sector)
        object.__setattr__(self, "industry", industry)
        object.__setattr__(self, "security_type", security_type)
        object.__setattr__(self, "membership_start", membership_start)
        object.__setattr__(self, "membership_end", membership_end)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "source_snapshot_id", source_snapshot_id)
        object.__setattr__(self, "required_history_sessions", required_history_sessions)
        object.__setattr__(self, "point_in_time_liquidity_ok", point_in_time_liquidity_ok)


@dataclass(frozen=True)
class UniverseManifest:
    manifest_version: str = "universe-v8-v1"
    protocol_version: str = "global-volatility-distribution-v8-news-transfer"
    created_at: str | None = None
    seed: int = V8_UNIVERSE_SEED
    members: tuple[UniverseMember, ...] = field(default_factory=tuple)
    source_checksums: dict[str, str] = field(default_factory=dict)
    source_attestations: dict[str, dict[str, Any]] = field(default_factory=dict)
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
    d["ticker"] = d["ticker"].upper().strip()
    if d.get("isin"):
        d["isin"] = str(d["isin"]).upper().strip()
    # Normalize exchange MIC
    d["primary_exchange_mic"] = str(d["primary_exchange_mic"]).upper().strip()
    return d


def _manifest_canonical_bytes(manifest: dict[str, Any]) -> bytes:
    # Exclude sha256 field itself from the digest
    payload = {k: v for k, v in manifest.items() if k != "sha256"}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _validate_source_evidence(
    members: list[UniverseMember],
    *,
    source_checksums: dict[str, str],
    source_attestations: dict[str, dict[str, Any]],
    require_certifiable: bool,
) -> list[str]:
    """Validate immutable source identities and return non-certifiable reasons.

    A checksum proves which bytes were used, not that those bytes are a
    point-in-time security master.  Certifiable manifests therefore require a
    separate operator-supplied attestation for every member source.  The
    attestation is content-addressed as part of the universe manifest and is
    deliberately declarative: the repository never fabricates provider or
    licensing claims from a current-constituent web page.
    """

    reasons: list[str] = []
    for name, checksum in source_checksums.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("source checksum keys must be non-empty strings")
        if not isinstance(checksum, str) or not checksum.startswith("sha256:"):
            raise ValueError(f"source checksum for {name!r} must use sha256:<hex>")
        digest = checksum.removeprefix("sha256:")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest.lower()):
            raise ValueError(f"source checksum for {name!r} is not a SHA-256 digest")

    member_sources = {member.source.strip() for member in members}
    missing_attestations = sorted(member_sources - set(source_attestations))
    if missing_attestations:
        reasons.append("missing_source_attestations:" + ",".join(missing_attestations))

    required_true = (
        "license_acknowledged",
        "point_in_time_membership",
        "historical_listing_status",
        "includes_delisted_where_available",
    )
    for source_name, attestation in sorted(source_attestations.items()):
        if not isinstance(source_name, str) or not source_name.strip():
            raise ValueError("source attestation keys must be non-empty strings")
        if not isinstance(attestation, dict):
            raise ValueError(f"source attestation for {source_name!r} must be an object")
        for field_name in ("source_snapshot_id", "license_id", "retrieved_at"):
            value = attestation.get(field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"source attestation {source_name!r} has empty {field_name}")
        try:
            retrieved_at = datetime.fromisoformat(
                str(attestation["retrieved_at"]).replace("Z", "+00:00")
            )
        except ValueError as error:
            raise ValueError(
                f"source attestation {source_name!r} has invalid retrieved_at"
            ) from error
        if retrieved_at.tzinfo is None:
            raise ValueError(
                f"source attestation {source_name!r} retrieved_at must include a timezone"
            )
        for field_name in required_true:
            if attestation.get(field_name) is not True:
                reasons.append(f"{source_name}:{field_name}=false")
        evidence_files = attestation.get("evidence_files")
        if not isinstance(evidence_files, list) or not evidence_files:
            reasons.append(f"{source_name}:evidence_files=missing")
        else:
            missing_evidence = sorted(set(evidence_files) - set(source_checksums))
            if missing_evidence:
                reasons.append(f"{source_name}:unhashed_evidence=" + ",".join(missing_evidence))

    for member in members:
        attestation = source_attestations.get(member.source.strip())
        if isinstance(attestation, dict) and member.source_snapshot_id != attestation.get(
            "source_snapshot_id"
        ):
            reasons.append(f"{member.ticker}:source_snapshot_id_mismatch")

    if require_certifiable and reasons:
        raise ValueError("universe source evidence is not certifiable: " + "; ".join(reasons))
    return reasons


def _validate_member(m: UniverseMember) -> None:
    # IDs and tickers must be non-empty after stripping
    if not isinstance(m.security_id, str) or not m.security_id.strip():
        raise ValueError("member missing or whitespace-only security_id")
    if not isinstance(m.ticker, str) or not m.ticker.strip():
        raise ValueError(f"member {m.security_id!r} has empty or whitespace-only ticker")
    if not isinstance(m.company_name, str) or not m.company_name.strip():
        raise ValueError(f"member {m.ticker!r} has empty company_name")
    if not isinstance(m.source, str) or not m.source.strip():
        raise ValueError(f"member {m.ticker!r} has empty source")
    if not isinstance(m.source_snapshot_id, str) or not m.source_snapshot_id.strip():
        raise ValueError(f"member {m.ticker!r} has empty source_snapshot_id")
    # Security type allowlist (strict, not just excluded)
    if m.security_type.upper().strip() not in V8_ALLOWED_SECURITY_TYPES:
        raise ValueError(
            f"member {m.ticker!r} has non-allowlisted security_type {m.security_type!r}; "
            f"allowed: {sorted(V8_ALLOWED_SECURITY_TYPES)}"
        )
    # Exchange MIC must be known, never SP500_PIT
    if m.primary_exchange_mic.upper().strip() == "SP500_PIT":
        raise ValueError(
            f"member {m.ticker!r} uses SP500_PIT as MIC; use index_memberships instead"
        )
    if m.primary_exchange_mic not in set(V8_EXCHANGE_MICS.values()):
        raise ValueError(f"unknown exchange MIC {m.primary_exchange_mic!r} for {m.ticker!r}")
    # Currency / timezone
    if m.currency not in V8_VALID_CURRENCIES:
        raise ValueError(f"member {m.ticker!r} has invalid currency {m.currency!r}")
    if m.timezone not in V8_VALID_TIMEZONES:
        raise ValueError(f"member {m.ticker!r} has invalid timezone {m.timezone!r}")
    # Index memberships: must not contain duplicate index entries, must have start/end if present
    seen_idx: set[str] = set()
    for entry in m.index_memberships:
        if not isinstance(entry, dict):
            raise ValueError(f"member {m.ticker!r} has non-dict index_memberships entry")
        idx_name = entry.get("index")
        if idx_name not in V8_INDEX_IDS.values():
            raise ValueError(f"member {m.ticker!r} has unknown index {idx_name!r}")
        if idx_name in seen_idx:
            raise ValueError(f"member {m.ticker!r} has duplicate index {idx_name!r}")
        seen_idx.add(str(idx_name))
        for date_field in ("membership_start", "membership_end"):
            v = entry.get(date_field)
            if v is not None:
                try:
                    date.fromisoformat(str(v))
                except ValueError as error:
                    raise ValueError(f"invalid {date_field} {v!r} for {m.ticker!r}") from error
        # Check reversed dates within entry
        s_val = entry.get("membership_start")
        e_val = entry.get("membership_end")
        if (
            s_val is not None
            and e_val is not None
            and date.fromisoformat(str(s_val)) > date.fromisoformat(str(e_val))
        ):
            raise ValueError(f"index membership start after end for {m.ticker!r}")
    # Validate top-level membership dates
    for field_name in ("membership_start", "membership_end"):
        value = getattr(m, field_name)
        if value is not None:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"invalid {field_name} whitespace for {m.ticker!r}")
            try:
                date.fromisoformat(value)
            except ValueError as error:
                raise ValueError(f"invalid {field_name} {value!r} for {m.ticker!r}") from error
    s_top = m.membership_start
    e_top = m.membership_end
    if (
        s_top is not None
        and e_top is not None
        and date.fromisoformat(s_top) > date.fromisoformat(e_top)
    ):
        raise ValueError(f"membership_start after membership_end for {m.ticker!r}")
    # ISIN/FIGI normalization: if present must be non-whitespace
    for field_name in ("isin", "figi", "cik"):
        v = getattr(m, field_name)
        if v is not None and (not isinstance(v, str) or not v.strip()):
            raise ValueError(f"member {m.ticker!r} has whitespace-only {field_name}")


def build_universe_manifest(
    members: list[UniverseMember],
    *,
    source_checksums: dict[str, str] | None = None,
    source_attestations: dict[str, dict[str, Any]] | None = None,
    selection_policy: dict[str, Any] | None = None,
    seed: int = V8_UNIVERSE_SEED,
    protocol_version: str = "global-volatility-distribution-v8-news-transfer",
) -> dict[str, Any]:
    """Build a deterministic, content-addressed universe manifest (no I/O)."""

    if not members:
        raise ValueError("universe requires at least one member")
    if not source_checksums or not isinstance(source_checksums, dict) or not source_checksums:
        raise ValueError("source_checksums is required and must be non-empty (prove provenance)")
    # selection_policy must have uniform keys per exchange
    policy = dict(selection_policy or {})
    allow_sparse = bool(policy.get("allow_sparse", False))
    evidence_reasons = _validate_source_evidence(
        members,
        source_checksums=dict(source_checksums),
        source_attestations=dict(source_attestations or {}),
        require_certifiable=not allow_sparse,
    )
    # Validate dedupe by security_id (not ticker) and per-exchange minimums
    seen_ids: set[str] = set()
    seen_isins: dict[str, str] = {}
    seen_tickers: dict[str, str] = {}
    per_exchange: dict[str, int] = {}
    per_type: dict[str, int] = {}
    per_sector: dict[str, int] = {}
    for m in members:
        _validate_member(m)
        # Normalize security_id for dedupe
        norm_id = m.security_id.strip()
        if norm_id in seen_ids:
            raise ValueError(f"duplicate security_id {m.security_id!r}")
        seen_ids.add(norm_id)
        normalized_ticker = m.ticker.strip().upper()
        if normalized_ticker in seen_tickers:
            raise ValueError(
                f"duplicate active ticker {normalized_ticker!r} for {m.security_id!r} "
                f"and {seen_tickers[normalized_ticker]!r}; the current panel contract "
                "requires a unique provider symbol per security"
            )
        seen_tickers[normalized_ticker] = m.security_id
        # Also dedupe by normalized ISIN under different forms
        if m.isin:
            norm_isin = m.isin.strip().upper()
            if norm_isin in seen_isins:
                raise ValueError(
                    f"duplicate ISIN {norm_isin!r} for {m.ticker!r} and {seen_isins[norm_isin]!r}"
                )
            seen_isins[norm_isin] = m.ticker
        per_exchange[m.primary_exchange_mic] = per_exchange.get(m.primary_exchange_mic, 0) + 1
        per_type[m.security_type] = per_type.get(m.security_type, 0) + 1
        if m.sector:
            per_sector[m.sector] = per_sector.get(m.sector, 0) + 1

    # Enforce minimum coverage for a certifiable four-market model
    # We fail closed unless explicitly overridden via selection_policy allow_sparse=True
    if not allow_sparse:
        low_history = sorted(
            member.ticker for member in members if member.required_history_sessions < 756
        )
        if low_history:
            raise ValueError(
                "certifiable universe members require at least 756 history sessions: "
                + ", ".join(low_history)
            )
        failed_liquidity = sorted(
            member.ticker for member in members if not member.point_in_time_liquidity_ok
        )
        if failed_liquidity:
            raise ValueError(
                "certifiable universe members failed point-in-time liquidity: "
                + ", ".join(failed_liquidity)
            )
        for mic in (V8_EXCHANGE_MICS["NASDAQ"], V8_EXCHANGE_MICS["NYSE"], V8_EXCHANGE_MICS["LSE"]):
            count = per_exchange.get(mic, 0)
            if count < V8_MIN_PER_EXCHANGE:
                raise ValueError(
                    f"universe has {count} members for {mic}, need >= {V8_MIN_PER_EXCHANGE} for certifiable four-market model "
                    f"(set selection_policy.allow_sparse=true to record non-certifiable state explicitly)"
                )
        # Also require some SP500-tagged members if policy expects them
        sp500_tagged = sum(
            1 for m in members if any(e.get("index") == "SP500" for e in m.index_memberships)
        )
        if sp500_tagged == 0 and not policy.get("allow_no_sp500", False):
            raise ValueError("certifiable universe requires point-in-time S&P 500 membership rows")
        required_holdouts = {
            str(ticker).strip().upper() for ticker in policy.get("required_holdouts", ())
        }
        member_tickers = {member.ticker.strip().upper() for member in members}
        missing_holdouts = sorted(required_holdouts - member_tickers)
        if missing_holdouts:
            raise ValueError(
                "certifiable universe is missing required holdouts: " + ", ".join(missing_holdouts)
            )
        if len(per_sector) < V8_MIN_SECTOR_COVERAGE:
            raise ValueError(
                f"universe has only {len(per_sector)} sectors, need >= {V8_MIN_SECTOR_COVERAGE}"
            )

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
        "per_sector_counts": dict(sorted(per_sector.items())),
        "source_checksums": dict(sorted((source_checksums or {}).items())),
        "source_attestations": {
            key: dict(sorted(value.items()))
            for key, value in sorted((source_attestations or {}).items())
        },
        "selection_policy": dict(sorted((selection_policy or {}).items())),
        "coverage_certifiable": not allow_sparse,
        "coverage_reasons": evidence_reasons,
    }
    digest = _sha256_bytes(_manifest_canonical_bytes(manifest))
    manifest["sha256"] = digest
    return manifest


def verify_universe_manifest(payload: object) -> dict[str, Any]:
    """Rebuild and verify a persisted universe manifest, failing closed.

    A caller must not trust the embedded ``sha256`` alone: every member and
    coverage count is reconstructed through the same validation path used at
    creation time, then the complete canonical payload is compared.
    """
    if not isinstance(payload, dict):
        raise ValueError("universe manifest must be a JSON object")
    # Builders may still contain tuple-valued dataclass fields in memory;
    # manifests on disk contain the JSON-normalized list representation.
    normalized_payload = json.loads(json.dumps(payload, sort_keys=True))
    payload = normalized_payload
    rows = payload.get("members")
    if not isinstance(rows, list) or not rows:
        raise ValueError("universe manifest has no members")
    try:
        members = [UniverseMember(**row) for row in rows if isinstance(row, dict)]
    except (TypeError, ValueError) as error:
        raise ValueError("universe manifest member table is invalid") from error
    if len(members) != len(rows):
        raise ValueError("universe manifest contains a non-object member")
    expected = build_universe_manifest(
        members,
        source_checksums=payload.get("source_checksums"),
        source_attestations=payload.get("source_attestations"),
        selection_policy=payload.get("selection_policy"),
        seed=int(payload.get("seed", V8_UNIVERSE_SEED)),
        protocol_version=str(payload.get("protocol_version", "")),
    )
    normalized_expected = json.loads(json.dumps(expected, sort_keys=True))
    if normalized_payload != normalized_expected:
        raise ValueError("universe manifest content or checksum does not match")
    return normalized_expected


def universe_identity_maps(
    verified_manifest: dict[str, Any],
) -> tuple[dict[str, str], dict[str, str]]:
    """Return ticker-to-MIC and ticker-to-security-ID maps for split binding.

    The input must already have passed :func:`verify_universe_manifest`.  The
    helper still validates its local assumptions so a caller cannot silently
    collapse two securities onto one provider symbol before hashing sealed
    split assignments.
    """

    rows = verified_manifest.get("members")
    if not isinstance(rows, list) or not rows:
        raise ValueError("verified universe manifest has no members")
    exchange_map: dict[str, str] = {}
    security_id_map: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("verified universe manifest contains a non-object member")
        ticker = str(row.get("ticker", "")).strip().upper()
        mic = str(row.get("primary_exchange_mic", "")).strip().upper()
        security_id = str(row.get("security_id", "")).strip()
        if not ticker or not mic or not security_id:
            raise ValueError("verified universe member has incomplete split identity")
        if ticker in exchange_map or ticker in security_id_map:
            raise ValueError(f"verified universe contains duplicate ticker identity {ticker!r}")
        exchange_map[ticker] = mic
        security_id_map[ticker] = security_id
    return exchange_map, security_id_map


def write_universe_manifest(out_dir: Path, members: list[UniverseMember], **kwargs: Any) -> Path:
    """Atomically write ``universe-v8-manifest.json``; refuses overwrites (fsync+replace)."""
    manifest = build_universe_manifest(members, **kwargs)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "universe-v8-manifest.json"
    if target.exists():
        raise FileExistsError(f"universe manifest already exists at {target} – immutable")
    # Atomic write via temp file in same directory
    tmp_fd, tmp_path_str = tempfile.mkstemp(prefix=".universe-v8-", dir=str(out_dir))
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Ensure temp is on same filesystem, then atomic replace
        tmp_path.replace(target)
        # Fsync directory to ensure durability (best effort, Unix only)
        try:
            if hasattr(os, "O_DIRECTORY"):
                dir_fd = os.open(str(out_dir), os.O_DIRECTORY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
        except (OSError, AttributeError):
            pass
    finally:
        if tmp_path.exists():
            with contextlib.suppress(OSError):
                tmp_path.unlink()
    # Readback verification
    written = json.loads(target.read_text(encoding="utf-8"))
    if written.get("sha256") != manifest["sha256"]:
        raise RuntimeError("universe manifest checksum mismatch after write")
    return target


def initial_v8_selection_policy(seed: int = V8_UNIVERSE_SEED) -> dict[str, Any]:
    """Recommended initial v8 cohort policy (preregistered, deterministic, uniform keys)."""
    base = {
        "description": "All PIT S&P500 + liquidity-stratified Nasdaq/NYSE/LSE, >=25-50 per exchange",
        "seed": seed,
        "sp500_mode": "point_in_time_membership",
        "nasdaq": {
            "primary_mic": V8_EXCHANGE_MICS["NASDAQ"],
            "security_type": "COMMON",
            "exclude_types": sorted(V8_EXCLUDED_SECURITY_TYPES),
            "allowed_types": sorted(V8_ALLOWED_SECURITY_TYPES),
            "min_history_sessions": 756,
            "liquidity_filter": "point_in_time_median_dollar_volume_top_quintile",
            "include_delisted_where_available": True,
        },
        "nyse": {
            "primary_mic": V8_EXCHANGE_MICS["NYSE"],
            "security_type": "COMMON",
            "exclude_types": sorted(V8_EXCLUDED_SECURITY_TYPES),
            "allowed_types": sorted(V8_ALLOWED_SECURITY_TYPES),
            "min_history_sessions": 756,
            "liquidity_filter": "point_in_time_median_dollar_volume_top_quintile",
            "include_delisted_where_available": True,
        },
        "lse": {
            "primary_mic": V8_EXCHANGE_MICS["LSE"],
            "security_type": "COMMON",
            "exclude_types": sorted(V8_EXCLUDED_SECURITY_TYPES),
            "allowed_types": sorted(V8_ALLOWED_SECURITY_TYPES),
            "currency": "GBX",
            "timezone": "Europe/London",
            "include_secondary_listings": False,
            "min_history_sessions": 756,
        },
        "holdout_seed": seed,
        "holdout_fraction": 0.20,
        "required_holdouts": ["NMM", "MSFT"],
        "dedupe_keys": ["isin", "figi", "cik", "provider_security_id"],
        "coverage_requirements": {
            "min_per_exchange": V8_MIN_PER_EXCHANGE,
            "min_holdouts": V8_MIN_HOLDOUTS,
            "min_sectors": V8_MIN_SECTOR_COVERAGE,
        },
    }
    # Schema test: every exchange sub-policy must have identical field names
    exchange_keys = set(base["nasdaq"].keys())
    assert set(base["nyse"].keys()) == exchange_keys, "NYSE policy keys must match Nasdaq"
    assert set(base["lse"].keys()).issuperset(
        {"primary_mic", "security_type", "exclude_types", "allowed_types"}
    )
    return base
