"""Single-use sealed test certification execution for StockLSTM V10.

Enforces:
- Strict fail-closed data eligibility checks (no certification on ineligible data)
- Single-use test-opening audit record written before target access
- Strict rejection of second test-opening attempts
- Per-horizon Holm-adjusted significance and non-degradation criteria
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from research.volatility_forecasting.market_snapshot_v10 import DataIneligibilityError

logger = logging.getLogger("certification_v10")


class SealedTestReopenError(PermissionError):
    """Raised when an attempt is made to re-open an already evaluated sealed test."""


@dataclass(frozen=True)
class SealedTestOpeningRecordV10:
    test_opened_at_utc: str
    candidate_package_sha256: str
    protocol_sha256: str
    split_manifest_sha256: str
    operator: str
    attempt: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, output_dir: Path) -> Path:
        target = Path(output_dir) / "test_opening_record.json"
        if target.exists():
            raise SealedTestReopenError(
                f"Test opening record already exists at {target}. Reopening forbidden."
            )
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target


@dataclass(frozen=True)
class HorizonCertificationDecision:
    horizon: int
    family: str
    outcome: str  # "certified_learned_model" | "certified_baseline" | "abstention"
    relative_qlike: float
    ratio_upper_95: float
    dm_p_value: float
    holm_adjusted_p_value: float
    transfer_relative_qlike: float
    passed_all_gates: bool
    reason: str


@dataclass(frozen=True)
class CertificationReportV10:
    report_id: str
    protocol_id: str
    protocol_sha256: str
    candidate_package_sha256: str
    data_snapshot_sha256: str
    evaluated_at_utc: str
    decisions: tuple[HorizonCertificationDecision, ...]
    certified_horizons: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "protocol_id": self.protocol_id,
            "protocol_sha256": self.protocol_sha256,
            "candidate_package_sha256": self.candidate_package_sha256,
            "data_snapshot_sha256": self.data_snapshot_sha256,
            "evaluated_at_utc": self.evaluated_at_utc,
            "decisions": [asdict(d) for d in self.decisions],
            "certified_horizons": list(self.certified_horizons),
        }

    def save(self, output_dir: Path) -> Path:
        target = Path(output_dir) / "certification_record_v10.json"
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target


def verify_certification_prerequisites(
    protocol_data: dict[str, Any],
    candidate_manifest: dict[str, Any],
    output_dir: Path,
) -> None:
    """Strict pre-certification gate validation."""
    eligibility = protocol_data.get("data_eligibility", {})
    if not eligibility.get("universe_certification_eligible", False) or not eligibility.get(
        "market_panel_certification_eligible", False
    ):
        raise DataIneligibilityError(
            f"Cannot certify: {eligibility.get('blocker', 'Data is not certification-eligible.')}"
        )

    if protocol_data.get("protocol_status") != "frozen":
        raise ValueError("Protocol must be in 'frozen' status for certification.")

    # Check if test opening record already exists in output_dir
    opening_file = Path(output_dir) / "test_opening_record.json"
    if opening_file.exists():
        raise SealedTestReopenError(f"Sealed test already opened in {output_dir}")
