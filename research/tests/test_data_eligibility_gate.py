"""Tests for Data Eligibility Gate and Fail-Closed Certification Prevention."""

from __future__ import annotations

import pytest

from research.volatility_forecasting.market_snapshot_v10 import (
    DataIneligibilityError,
    MarketPanelSnapshotV10,
)


def test_ineligible_yfinance_snapshot_fails_closed() -> None:
    snapshot = MarketPanelSnapshotV10(
        snapshot_id="dev-yfinance-ndx100",
        provider="yfinance_development_cache",
        license_id="unverified",
        as_of_utc="2026-08-29T00:00:00Z",
        security_count=24,
        session_count=2764,
        row_count=66288,
        checksums={"panel.parquet": "0" * 64},
        certification_eligible=False,
    )
    with pytest.raises(DataIneligibilityError, match="certification_eligible=False"):
        snapshot.verify_certification_eligibility()


def test_certified_licensed_snapshot_passes_gate() -> None:
    snapshot = MarketPanelSnapshotV10(
        snapshot_id="prod-licensed-ndx100-v1",
        provider="attested_market_data_vendor",
        license_id="lic_prod_volatility_2026",
        as_of_utc="2026-08-29T00:00:00Z",
        security_count=100,
        session_count=2764,
        row_count=276400,
        checksums={"panel.parquet": "a" * 64},
        certification_eligible=True,
    )
    # Should not raise
    snapshot.verify_certification_eligibility()
