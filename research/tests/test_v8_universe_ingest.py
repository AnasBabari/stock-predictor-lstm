from __future__ import annotations

import json
from pathlib import Path

import pytest
from volatility_forecasting.universe_ingest_v8 import (
    load_source_attestations,
    load_universe_members_csv,
    sha256_file,
    validate_attestation_evidence_files,
)
from volatility_forecasting.universe_v8 import (
    UniverseMember,
    build_universe_manifest,
    initial_v8_selection_policy,
    verify_universe_manifest,
)


def _attestation() -> dict[str, object]:
    return {
        "source_snapshot_id": "pit-master-2026-08-27",
        "license_id": "provider-research-license-v1",
        "retrieved_at": "2026-08-27T10:00:00+00:00",
        "license_acknowledged": True,
        "point_in_time_membership": True,
        "historical_listing_status": True,
        "includes_delisted_where_available": True,
        "evidence_files": ["membership_archive"],
    }


def test_csv_ingestion_preserves_point_in_time_identity(tmp_path: Path) -> None:
    csv_path = tmp_path / "members.csv"
    csv_path.write_text(
        "security_id,ticker,company_name,primary_exchange_mic,currency,timezone,"
        "sector,security_type,source,source_snapshot_id,index_memberships_json,"
        "point_in_time_liquidity_ok\n"
        "figi:MSFT,msft,Microsoft Corp,XNAS,USD,America/New_York,Technology,COMMON,"
        'pit-provider,pit-2026,"[{""index"":""SP500"",""membership_start"":""1994-06-01"",""membership_end"":null}]",true\n',
        encoding="utf-8",
    )

    members = load_universe_members_csv(csv_path)

    assert len(members) == 1
    assert members[0].ticker == "MSFT"
    assert members[0].index_memberships[0]["index"] == "SP500"
    assert sha256_file(csv_path).startswith("sha256:")


def test_csv_ingestion_rejects_ambiguous_boolean(tmp_path: Path) -> None:
    csv_path = tmp_path / "members.csv"
    csv_path.write_text(
        "security_id,ticker,company_name,primary_exchange_mic,currency,timezone,"
        "sector,security_type,source,source_snapshot_id,point_in_time_liquidity_ok\n"
        "id:X,X,X Corp,XNAS,USD,America/New_York,Technology,COMMON,source,snapshot,maybe\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must be true or false"):
        load_universe_members_csv(csv_path)


def test_attestation_loader_requires_hashed_evidence(tmp_path: Path) -> None:
    path = tmp_path / "attestations.json"
    path.write_text(
        json.dumps({"schema_version": 1, "sources": {"pit-provider": _attestation()}}),
        encoding="utf-8",
    )
    attestations = load_source_attestations(path)
    with pytest.raises(ValueError, match="unhashed evidence files"):
        validate_attestation_evidence_files(attestations, {"members_csv": "sha256:" + "a" * 64})


def _certifiable_members() -> list[UniverseMember]:
    rows: list[UniverseMember] = []
    counters = {"XNAS": 0, "XNYS": 0, "XLON": 0}
    for mic in counters:
        for index in range(25):
            ticker = f"{mic[1:3]}{index:02d}" + (".L" if mic == "XLON" else "")
            if mic == "XNAS" and index == 0:
                ticker = "MSFT"
            elif mic == "XNYS" and index == 0:
                ticker = "NMM"
            rows.append(
                UniverseMember(
                    security_id=f"pit:{mic}:{index}",
                    ticker=ticker,
                    company_name=f"Company {mic} {index}",
                    isin=None,
                    figi=None,
                    cik=None,
                    primary_exchange_mic=mic,
                    index_memberships=(
                        (
                            {
                                "index": "SP500",
                                "membership_start": "2020-01-01",
                                "membership_end": None,
                            },
                        )
                        if mic != "XLON" and index < 5
                        else ()
                    ),
                    currency="GBX" if mic == "XLON" else "USD",
                    timezone="Europe/London" if mic == "XLON" else "America/New_York",
                    sector=("Technology", "Energy", "Industrials")[index % 3],
                    source="pit-provider",
                    source_snapshot_id="pit-master-2026-08-27",
                )
            )
    return rows


def test_certifiable_manifest_requires_and_binds_source_attestation() -> None:
    manifest = build_universe_manifest(
        _certifiable_members(),
        source_checksums={"membership_archive": "sha256:" + "a" * 64},
        source_attestations={"pit-provider": _attestation()},
        selection_policy=initial_v8_selection_policy(),
    )

    assert manifest["coverage_certifiable"] is True
    assert manifest["coverage_reasons"] == []
    persisted = json.loads(json.dumps(manifest))
    assert verify_universe_manifest(persisted) == persisted


def test_certifiable_manifest_rejects_unattested_current_list() -> None:
    with pytest.raises(ValueError, match="missing_source_attestations"):
        build_universe_manifest(
            _certifiable_members(),
            source_checksums={"membership_archive": "sha256:" + "a" * 64},
            selection_policy=initial_v8_selection_policy(),
        )


def test_manifest_rejects_duplicate_provider_ticker_identity() -> None:
    first = UniverseMember(
        security_id="pit:first",
        ticker="DUP",
        company_name="First Corp",
        isin=None,
        figi=None,
        cik=None,
        primary_exchange_mic="XNAS",
        sector="Technology",
        source="pit-provider",
        source_snapshot_id="pit-master-2026-08-27",
    )
    second = UniverseMember(
        security_id="pit:second",
        ticker="dup",
        company_name="Second Corp",
        isin=None,
        figi=None,
        cik=None,
        primary_exchange_mic="XNYS",
        sector="Energy",
        source="pit-provider",
        source_snapshot_id="pit-master-2026-08-27",
    )
    with pytest.raises(ValueError, match="duplicate active ticker"):
        build_universe_manifest(
            [first, second],
            source_checksums={"membership_archive": "sha256:" + "a" * 64},
            source_attestations={"pit-provider": _attestation()},
            selection_policy={"allow_sparse": True},
        )
