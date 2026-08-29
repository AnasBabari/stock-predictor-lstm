"""Tests for V10 Point-in-Time Universe and Security Master."""

from __future__ import annotations

import pytest

from research.volatility_forecasting.universe_v10 import (
    PointInTimeUniverseManifest,
    SecurityRecord,
    filter_eligible_equities,
)


@pytest.fixture
def sample_securities() -> list[SecurityRecord]:
    return [
        SecurityRecord(
            security_id="SEC_AAPL_001",
            ticker="AAPL",
            exchange="NASDAQ",
            primary_share_class=True,
            is_etf_or_fund=False,
            is_spac_or_warrant=False,
            listing_date="1980-12-12",
            delisting_date=None,
        ),
        SecurityRecord(
            security_id="SEC_SPY_001",
            ticker="SPY",
            exchange="NYSE_ARCA",
            primary_share_class=True,
            is_etf_or_fund=True,
            is_spac_or_warrant=False,
            listing_date="1993-01-22",
            delisting_date=None,
        ),
        SecurityRecord(
            security_id="SEC_SPAC_001",
            ticker="TESTW",
            exchange="NASDAQ",
            primary_share_class=False,
            is_etf_or_fund=False,
            is_spac_or_warrant=True,
            listing_date="2021-01-01",
            delisting_date="2022-01-01",
        ),
    ]


def test_filter_eligible_equities_excludes_funds_and_spacs(
    sample_securities: list[SecurityRecord],
) -> None:
    filtered = filter_eligible_equities(sample_securities)
    assert len(filtered) == 1
    assert filtered[0].ticker == "AAPL"


def test_universe_manifest_licensing_verification() -> None:
    manifest_unverified = PointInTimeUniverseManifest(
        universe_id="u1",
        as_of_date="2024-01-01",
        securities=(),
        data_provider="yfinance_scraper",
        license_id="unverified",
        certification_eligible=False,
        checksum_sha256="0" * 64,
    )
    assert manifest_unverified.verify_licensing() is False

    manifest_valid = PointInTimeUniverseManifest(
        universe_id="u2",
        as_of_date="2024-01-01",
        securities=(),
        data_provider="attested_institutional_feed",
        license_id="lic_commercial_prod_2026",
        certification_eligible=True,
        checksum_sha256="1" * 64,
    )
    assert manifest_valid.verify_licensing() is True
