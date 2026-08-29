"""Certification-grade point-in-time security master and universe definitions for V10.

Manages permanent security IDs (independent of ticker renames), corporate actions,
point-in-time index memberships (S&P 500, Nasdaq-100), and causal liquidity bounding
for NYSE and LSE ordinary equities.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SecurityRecord:
    security_id: str
    ticker: str
    exchange: str
    primary_share_class: bool
    is_etf_or_fund: bool
    is_spac_or_warrant: bool
    listing_date: str
    delisting_date: str | None
    figi: str | None = None
    cik: str | None = None


@dataclass(frozen=True)
class PointInTimeUniverseManifest:
    universe_id: str
    as_of_date: str
    securities: tuple[SecurityRecord, ...]
    data_provider: str
    license_id: str
    certification_eligible: bool
    checksum_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "universe_id": self.universe_id,
            "as_of_date": self.as_of_date,
            "securities": [asdict(s) for s in self.securities],
            "data_provider": self.data_provider,
            "license_id": self.license_id,
            "certification_eligible": self.certification_eligible,
            "checksum_sha256": self.checksum_sha256,
        }

    def verify_licensing(self) -> bool:
        """Verify that provider and license grant certification eligibility."""
        if not self.certification_eligible:
            return False
        if not self.data_provider or not self.license_id:
            return False
        # Free public web scraping or unverified development sources cannot be certification eligible
        if "yfinance" in self.data_provider.lower() or "unverified" in self.license_id.lower():
            return False
        return True


def filter_eligible_equities(
    records: list[SecurityRecord],
    *,
    require_primary: bool = True,
    exclude_derivatives: bool = True,
) -> list[SecurityRecord]:
    """Filter ordinary primary shares meeting certification eligibility criteria."""
    filtered = []
    for r in records:
        if require_primary and not r.primary_share_class:
            continue
        if exclude_derivatives and (r.is_etf_or_fund or r.is_spac_or_warrant):
            continue
        filtered.append(r)
    return filtered
