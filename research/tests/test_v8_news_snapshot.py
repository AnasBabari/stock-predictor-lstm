from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest
from volatility_forecasting.news import NewsEvent
from volatility_forecasting.news_snapshot import save_news_snapshot
from volatility_forecasting.news_snapshot_v8 import (
    V8_NEWS_STATUS_NUMERIC,
    V8_NEWS_STATUS_READY,
    build_numeric_fallback_news_snapshot,
    build_v8_news_manifest,
    verify_v8_news_manifest,
)
from volatility_forecasting.universe_v8 import UniverseMember, build_universe_manifest


def _attestation() -> dict[str, object]:
    return {
        "source_snapshot_id": "pit-v1",
        "license_id": "research-v1",
        "retrieved_at": "2026-08-27T10:00:00+00:00",
        "license_acknowledged": True,
        "point_in_time_membership": True,
        "historical_listing_status": True,
        "includes_delisted_where_available": True,
        "evidence_files": ["archive"],
    }


def _universe() -> dict:
    members: list[UniverseMember] = []
    for mic in ("XNAS", "XNYS", "XLON"):
        for index in range(25):
            ticker = f"{mic[1:3]}{index:02d}" + (".L" if mic == "XLON" else "")
            if mic == "XNAS" and index == 0:
                ticker = "MSFT"
            elif mic == "XNYS" and index == 0:
                ticker = "NMM"
            members.append(
                UniverseMember(
                    security_id=f"pit:{mic}:{index}",
                    ticker=ticker,
                    company_name=f"Company {mic} {index}",
                    isin=None,
                    figi=None,
                    cik=None,
                    primary_exchange_mic=mic,
                    index_memberships=(
                        ({"index": "SP500", "membership_start": "2019-01-01"},)
                        if mic != "XLON" and index < 5
                        else ()
                    ),
                    currency="GBX" if mic == "XLON" else "USD",
                    timezone="Europe/London" if mic == "XLON" else "America/New_York",
                    sector=("Technology", "Energy", "Industrials")[index % 3],
                    source="pit-provider",
                    source_snapshot_id="pit-v1",
                )
            )
    return build_universe_manifest(
        members,
        source_checksums={"archive": "sha256:" + "a" * 64},
        source_attestations={"pit-provider": _attestation()},
        selection_policy={"required_holdouts": ["NMM", "MSFT"]},
    )


def _market(universe: dict) -> dict:
    return {
        "pooled_checksum": "sha256:" + "b" * 64,
        "tickers": {
            member["ticker"]: {"start": "2020-01-02", "end": "2020-12-30"}
            for member in universe["members"]
        },
        "v8_market": {
            "universe_manifest_sha256": universe["sha256"],
            "coverage_certifiable": True,
        },
    }


def _event() -> NewsEvent:
    return NewsEvent(
        event_id="gdelt:1",
        cluster_id="cluster:1",
        source="example.com",
        first_seen_at=pd.Timestamp("2020-06-01T12:00:00Z"),
        published_at=pd.Timestamp("2020-06-01T11:00:00Z"),
        timestamp_quality="precise",
        tickers=("MSFT",),
        topics=("regulation",),
        positive_probability=0.1,
        neutral_probability=0.2,
        negative_probability=0.7,
        novelty=0.5,
        severity=0.8,
        confidence=0.9,
        source_reliability=0.7,
    )


def _snapshot_and_aliases(
    tmp_path: Path, universe: dict, *, missing_dates: list[str] | None = None
) -> tuple[Path, Path]:
    snapshot = tmp_path / "news"
    save_news_snapshot(
        snapshot,
        [_event()],
        provider="GDELT fixture",
        license_acknowledged=True,
        coverage_start="2019-12-01T00:00:00Z",
        coverage_end_exclusive="2021-01-02T00:00:00Z",
        provenance={"missing_archive_dates": missing_dates or []},
    )
    aliases = tmp_path / "aliases.json"
    aliases.write_text(
        json.dumps(
            {
                member["ticker"]: [
                    member["company_name"] if len(member["company_name"]) >= 3 else member["ticker"]
                ]
                for member in universe["members"]
            }
        ),
        encoding="utf-8",
    )
    return snapshot, aliases


def test_numeric_fallback_is_never_news_certified() -> None:
    manifest = build_numeric_fallback_news_snapshot()
    assert manifest["news_enabled"] is False
    assert manifest["news_status"] == V8_NEWS_STATUS_NUMERIC
    assert manifest["model_certified"] is False
    assert len(manifest["sha256"]) == 64


def test_complete_event_lake_is_ready_for_ablation_not_certified(tmp_path: Path) -> None:
    universe = _universe()
    market = _market(universe)
    snapshot, aliases = _snapshot_and_aliases(tmp_path, universe)

    manifest = build_v8_news_manifest(
        news_snapshot_dir=snapshot,
        universe_manifest=universe,
        market_manifest=market,
        ticker_aliases_path=aliases,
        provider_license_id="gdelt-terms-ack-2026-08-27",
    )

    assert manifest["coverage_complete"] is True
    assert manifest["news_status"] == V8_NEWS_STATUS_READY
    assert manifest["model_certified"] is False
    assert manifest["available_at_policy"].startswith("max(published_at,first_seen_at)")
    assert (
        verify_v8_news_manifest(
            json.loads(json.dumps(manifest)),
            news_snapshot_dir=snapshot,
            universe_manifest=universe,
            market_manifest=market,
            ticker_aliases_path=aliases,
        )
        == manifest
    )


def test_provider_gap_fails_closed_or_is_explicitly_diagnostic(tmp_path: Path) -> None:
    universe = _universe()
    market = _market(universe)
    snapshot, aliases = _snapshot_and_aliases(tmp_path, universe, missing_dates=["2020-05-01"])
    kwargs = {
        "news_snapshot_dir": snapshot,
        "universe_manifest": universe,
        "market_manifest": market,
        "ticker_aliases_path": aliases,
        "provider_license_id": "gdelt-terms-ack-2026-08-27",
    }
    with pytest.raises(ValueError, match="provider_archive_gaps"):
        build_v8_news_manifest(**kwargs)
    manifest = build_v8_news_manifest(**kwargs, allow_provider_gaps=True)
    assert manifest["coverage_complete"] is False
    assert manifest["coverage_reasons"] == ["provider_archive_gaps"]


def test_alias_and_manifest_tampering_are_rejected(tmp_path: Path) -> None:
    universe = _universe()
    market = _market(universe)
    snapshot, aliases = _snapshot_and_aliases(tmp_path, universe)
    manifest = build_v8_news_manifest(
        news_snapshot_dir=snapshot,
        universe_manifest=universe,
        market_manifest=market,
        ticker_aliases_path=aliases,
        provider_license_id="gdelt-terms-ack-2026-08-27",
    )
    aliases_payload = json.loads(aliases.read_text(encoding="utf-8"))
    aliases_payload.pop("MSFT")
    aliases.write_text(json.dumps(aliases_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="alias coverage mismatch"):
        verify_v8_news_manifest(
            copy.deepcopy(manifest),
            news_snapshot_dir=snapshot,
            universe_manifest=universe,
            market_manifest=market,
            ticker_aliases_path=aliases,
        )
