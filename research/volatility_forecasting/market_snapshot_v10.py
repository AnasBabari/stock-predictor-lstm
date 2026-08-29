"""Immutable licensed market panel snapshots and data eligibility gates for V10.

Ensures that training and evaluation panels are bound to verified provider receipts,
license IDs, corporate-action adjustment logs, and cryptographic checksums.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class DataIneligibilityError(PermissionError):
    """Raised when an uncertified data snapshot attempts to run in a certifying pipeline."""


@dataclass(frozen=True)
class MarketPanelSnapshotV10:
    snapshot_id: str
    provider: str
    license_id: str
    as_of_utc: str
    security_count: int
    session_count: int
    row_count: int
    checksums: dict[str, str]
    certification_eligible: bool
    adjustment_method: str = "point_in_time_split_and_dividend_adjusted"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def verify_certification_eligibility(self) -> None:
        """Strict fail-closed check on data eligibility."""
        if not self.certification_eligible:
            raise DataIneligibilityError(
                f"Data snapshot '{self.snapshot_id}' is marked certification_eligible=False. "
                "Certification on unverified/development data is strictly prohibited."
            )
        if not self.provider or "yfinance" in self.provider.lower():
            raise DataIneligibilityError(
                f"Data provider '{self.provider}' is not licensed for certified model production."
            )
        if not self.license_id or "unverified" in self.license_id.lower():
            raise DataIneligibilityError(
                f"License ID '{self.license_id}' is invalid or unverified."
            )
        if not self.checksums:
            raise DataIneligibilityError("Data snapshot has no integrity checksums.")

    def save_manifest(self, directory: Path) -> Path:
        target_dir = Path(directory) / self.snapshot_id
        target_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = target_dir / "manifest.json"
        manifest_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return manifest_path

    @classmethod
    def from_manifest(cls, manifest_path: Path) -> MarketPanelSnapshotV10:
        data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        return cls(
            snapshot_id=str(data["snapshot_id"]),
            provider=str(data["provider"]),
            license_id=str(data["license_id"]),
            as_of_utc=str(data["as_of_utc"]),
            security_count=int(data["security_count"]),
            session_count=int(data["session_count"]),
            row_count=int(data["row_count"]),
            checksums=dict(data.get("checksums", {})),
            certification_eligible=bool(data.get("certification_eligible", False)),
            adjustment_method=str(data.get("adjustment_method", "point_in_time_split_and_dividend_adjusted")),
        )
