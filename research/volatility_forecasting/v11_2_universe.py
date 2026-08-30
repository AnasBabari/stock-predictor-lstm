"""Audited point-in-time 64-security universe for V11.2.

The module deliberately does not download or infer constituents.  Callers must
provide an audited manifest with stable identifiers, alias intervals, and
membership provenance.  A current constituent list is never accepted as a
historical substitute.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .v11_2_protocol import V11_2_UNIVERSE_SIZE, canonical_json_digest


def _date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid ISO date: {value!r}") from exc


@dataclass(frozen=True)
class MembershipInterval:
    start_date: str
    end_date: str
    source: str
    source_digest: str

    def __post_init__(self) -> None:
        if _date(self.start_date) > _date(self.end_date):
            raise ValueError("membership interval is not chronological")
        if (
            not self.source.strip()
            or len(self.source_digest) != 64
            or any(value not in "0123456789abcdef" for value in self.source_digest)
        ):
            raise ValueError("membership provenance requires a source and content digest")


@dataclass(frozen=True)
class TickerInterval:
    ticker: str
    start_date: str
    end_date: str

    def __post_init__(self) -> None:
        if not self.ticker.strip():
            raise ValueError("ticker alias cannot be empty")
        if _date(self.start_date) > _date(self.end_date):
            raise ValueError("ticker interval is not chronological")


@dataclass(frozen=True)
class PITSecurity:
    security_id: str
    cik: str
    figi: str
    exchange_mic: str
    sector: str
    industry: str
    volatility_stratum: str
    market_cap_stratum: str
    ticker_intervals: tuple[TickerInterval, ...]
    membership_intervals: tuple[MembershipInterval, ...]
    provider_aliases: tuple[str, ...] = ()
    corporate_actions: tuple[dict[str, Any], ...] = ()
    ohlcv_coverage: dict[str, Any] | None = None
    provenance: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        required = {
            "security_id": self.security_id,
            "cik": self.cik,
            "figi": self.figi,
            "exchange_mic": self.exchange_mic,
            "sector": self.sector,
            "industry": self.industry,
            "volatility_stratum": self.volatility_stratum,
            "market_cap_stratum": self.market_cap_stratum,
        }
        if any(not str(value).strip() for value in required.values()):
            raise ValueError("security identity and stratification fields are required")
        if not self.ticker_intervals or not self.membership_intervals:
            raise ValueError(f"{self.security_id}: ticker and membership intervals are required")
        aliases = [interval.ticker.upper() for interval in self.ticker_intervals]
        if len(aliases) != len(set(aliases)):
            raise ValueError(f"{self.security_id}: duplicate ticker interval")
        for left, right in zip(self.ticker_intervals, self.ticker_intervals[1:], strict=False):
            if _date(left.start_date) > _date(right.start_date):
                raise ValueError(f"{self.security_id}: ticker intervals are not chronological")
            if _date(left.end_date) >= _date(right.start_date):
                raise ValueError(f"{self.security_id}: overlapping ticker intervals")
        for left, right in zip(
            self.membership_intervals, self.membership_intervals[1:], strict=False
        ):
            if _date(left.start_date) > _date(right.start_date):
                raise ValueError(f"{self.security_id}: membership intervals are not chronological")
            if _date(left.end_date) >= _date(right.start_date):
                raise ValueError(f"{self.security_id}: overlapping membership intervals")

    def is_member(self, date_str: str) -> bool:
        value = _date(date_str)
        return any(
            _date(i.start_date) <= value <= _date(i.end_date) for i in self.membership_intervals
        )

    def ticker_at(self, date_str: str) -> str | None:
        value = _date(date_str)
        for interval in self.ticker_intervals:
            if _date(interval.start_date) <= value <= _date(interval.end_date):
                return interval.ticker
        return None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ticker_intervals"] = [asdict(i) for i in self.ticker_intervals]
        payload["membership_intervals"] = [asdict(i) for i in self.membership_intervals]
        return payload


@dataclass(frozen=True)
class V112UniverseManifest:
    protocol_id: str
    universe_version: str
    securities: tuple[PITSecurity, ...]
    selection_method: str
    membership_sources: tuple[str, ...]
    certification_eligible: bool
    manifest_sha256: str

    @property
    def universe_size(self) -> int:
        return len(self.securities)

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_id": self.protocol_id,
            "universe_version": self.universe_version,
            "universe_size": self.universe_size,
            "selection_method": self.selection_method,
            "membership_sources": list(self.membership_sources),
            "certification_eligible": self.certification_eligible,
            "securities": [security.to_dict() for security in self.securities],
            "manifest_sha256": self.manifest_sha256,
        }


def _validate_strata(securities: list[PITSecurity]) -> None:
    counts: dict[str, int] = {}
    for security in securities:
        counts[security.sector] = counts.get(security.sector, 0) + 1
    if len(counts) != 8 or any(count != 8 for count in counts.values()):
        raise ValueError(
            "V11.2 universe must contain eight sector strata with eight securities each"
        )


def build_universe_manifest(
    securities: Iterable[PITSecurity],
    *,
    protocol_id: str,
    universe_version: str = "v11.2-pit64-r1",
    membership_sources: Iterable[str] = (),
    selection_method: str = "curated_stratified_pit64",
    certification_eligible: bool = True,
) -> V112UniverseManifest:
    """Validate and hash the exact 64-security curated manifest."""
    ordered = sorted(list(securities), key=lambda item: item.security_id)
    if len(ordered) != V11_2_UNIVERSE_SIZE:
        raise ValueError(f"V11.2 requires exactly {V11_2_UNIVERSE_SIZE} accepted securities")
    ids = [item.security_id for item in ordered]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate permanent security_id")
    if len(set(item.exchange_mic for item in ordered)) < 1:
        raise ValueError("at least one exchange MIC is required")
    if not universe_version.strip():
        raise ValueError("universe version is required")
    _validate_strata(ordered)
    source_list = tuple(sorted({source.strip() for source in membership_sources if source.strip()}))
    if not source_list:
        raise ValueError("membership sources must be recorded")
    payload = {
        "protocol_id": protocol_id,
        "universe_version": universe_version,
        "selection_method": selection_method,
        "membership_sources": list(source_list),
        "certification_eligible": certification_eligible,
        "securities": [item.to_dict() for item in ordered],
    }
    return V112UniverseManifest(
        protocol_id=protocol_id,
        universe_version=universe_version,
        securities=tuple(ordered),
        selection_method=selection_method,
        membership_sources=source_list,
        certification_eligible=certification_eligible,
        manifest_sha256=canonical_json_digest(payload),
    )


class PITUniverseResolver:
    """Resolve point-in-time aliases without accepting out-of-interval rows."""

    def __init__(self, manifest: V112UniverseManifest) -> None:
        self.manifest = manifest
        self._by_id = {item.security_id: item for item in manifest.securities}

    def resolve(self, ticker: str, date_str: str) -> PITSecurity | None:
        query = ticker.strip().upper()
        if not query:
            return None
        for security in self.manifest.securities:
            if security.ticker_at(date_str) == query and security.is_member(date_str):
                return security
        return None

    def require_member(self, security_id: str, date_str: str) -> PITSecurity:
        security = self._by_id.get(security_id)
        if security is None or not security.is_member(date_str):
            raise ValueError(f"{security_id} is not an active PIT constituent at {date_str}")
        return security


def load_universe_manifest(path: Path) -> V112UniverseManifest:
    """Load and re-hash a canonical PIT manifest before it is used for data."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("universe manifest must contain a JSON object")
    try:
        securities = [
            PITSecurity(
                security_id=str(item["security_id"]),
                cik=str(item["cik"]),
                figi=str(item["figi"]),
                exchange_mic=str(item["exchange_mic"]),
                sector=str(item["sector"]),
                industry=str(item["industry"]),
                volatility_stratum=str(item["volatility_stratum"]),
                market_cap_stratum=str(item["market_cap_stratum"]),
                ticker_intervals=tuple(
                    TickerInterval(**interval) for interval in item["ticker_intervals"]
                ),
                membership_intervals=tuple(
                    MembershipInterval(**interval) for interval in item["membership_intervals"]
                ),
                provider_aliases=tuple(str(value) for value in item.get("provider_aliases", ())),
                corporate_actions=tuple(item.get("corporate_actions", ())),
                ohlcv_coverage=item.get("ohlcv_coverage"),
                provenance=item.get("provenance"),
            )
            for item in payload["securities"]
        ]
        certification_eligible = payload["certification_eligible"]
        if not isinstance(certification_eligible, bool):
            raise ValueError("certification_eligible must be a JSON boolean")
        manifest = build_universe_manifest(
            securities,
            protocol_id=str(payload["protocol_id"]),
            universe_version=str(payload["universe_version"]),
            membership_sources=tuple(str(value) for value in payload["membership_sources"]),
            selection_method=str(payload["selection_method"]),
            certification_eligible=certification_eligible,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("universe manifest is malformed") from exc
    canonical_payload = json.loads(json.dumps(manifest.to_dict(), sort_keys=True))
    if canonical_payload != payload:
        raise ValueError("universe manifest is not canonical or its digest is invalid")
    return manifest


def save_universe_manifest(manifest: V112UniverseManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
