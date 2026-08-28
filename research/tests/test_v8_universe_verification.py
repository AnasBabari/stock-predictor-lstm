from __future__ import annotations

import copy

import pytest
from volatility_forecasting.universe_v8 import (
    UniverseMember,
    build_universe_manifest,
    verify_universe_manifest,
)


def _member(ticker: str, mic: str, sector: str) -> UniverseMember:
    return UniverseMember(
        security_id=f"fixture:{ticker}",
        ticker=ticker,
        company_name=f"{ticker} Plc",
        isin=None,
        figi=None,
        cik=None,
        primary_exchange_mic=mic,
        currency="GBX" if mic == "XLON" else "USD",
        timezone="Europe/London" if mic == "XLON" else "America/New_York",
        sector=sector,
        source="fixture",
        source_snapshot_id="fixture-v1",
    )


def _manifest() -> dict:
    members = [
        _member("AAA", "XNAS", "Technology"),
        _member("BBB", "XNYS", "Industrials"),
        _member("CCC.L", "XLON", "Energy"),
    ]
    return build_universe_manifest(
        members,
        source_checksums={"fixture.csv": "sha256:" + "a" * 64},
        selection_policy={"allow_sparse": True},
    )


def test_verify_universe_manifest_round_trips_json_shape() -> None:
    manifest = _manifest()
    # Persisted JSON normalizes tuples to lists.
    import json

    persisted = json.loads(json.dumps(manifest))
    assert verify_universe_manifest(persisted) == persisted


def test_verify_universe_manifest_rejects_tampered_members_and_counts() -> None:
    manifest = _manifest()
    tampered_member = copy.deepcopy(manifest)
    tampered_member["members"][0]["ticker"] = "ZZZ"
    with pytest.raises(ValueError, match="content or checksum"):
        verify_universe_manifest(tampered_member)

    tampered_count = copy.deepcopy(manifest)
    tampered_count["total_members"] += 1
    with pytest.raises(ValueError, match="content or checksum"):
        verify_universe_manifest(tampered_count)
