"""Tests for exact PIT64 universe curation and identity resolution."""

from __future__ import annotations

import pytest

from research.volatility_forecasting.v11_2_universe import (
    MembershipInterval,
    PITSecurity,
    PITUniverseResolver,
    TickerInterval,
    build_universe_manifest,
)

SECTORS = (
    "semiconductors_hardware",
    "software_cloud",
    "communications_media",
    "healthcare_biotech",
    "industrials_transportation",
    "consumer_discretionary",
    "consumer_staples_defensive",
    "energy_utilities_financial_adjacent",
)


def _security(index: int, sector: str) -> PITSecurity:
    ticker = f"T{index:03d}"
    return PITSecurity(
        security_id=f"US.TEST{index:03d}",
        cik=f"{index + 1:010d}",
        figi=f"BBG00TEST{index:03d}",
        exchange_mic="XNAS",
        sector=sector,
        industry="test",
        volatility_stratum="low" if index % 2 else "high",
        market_cap_stratum="large" if index % 3 else "mid",
        ticker_intervals=(TickerInterval(ticker, "2020-01-01", "2030-12-31"),),
        membership_intervals=(
            MembershipInterval("2020-01-01", "2030-12-31", "test-source", "a" * 64),
        ),
    )


def test_manifest_requires_exact_eight_by_eight_strata() -> None:
    securities = [_security(i, SECTORS[i // 8]) for i in range(64)]
    manifest = build_universe_manifest(
        securities,
        protocol_id="stocklstm-volatility-v11.2-numeric-pit64",
        membership_sources=["test-source"],
    )
    assert manifest.universe_size == 64
    assert len(manifest.manifest_sha256) == 64
    assert PITUniverseResolver(manifest).resolve("t000", "2025-01-01") is not None


def test_manifest_rejects_missing_security() -> None:
    securities = [_security(i, SECTORS[i // 8]) for i in range(63)]
    with pytest.raises(ValueError, match="exactly 64"):
        build_universe_manifest(
            securities,
            protocol_id="stocklstm-volatility-v11.2-numeric-pit64",
            membership_sources=["test-source"],
        )
