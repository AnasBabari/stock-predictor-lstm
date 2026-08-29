from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from volatility_forecasting.market_snapshot_v8 import (
    V8_MARKET_COLUMNS,
    build_v8_market_snapshot,
    canonical_v8_market_csv,
    normalize_v8_provider_frame,
    verify_v8_market_snapshot,
    write_v8_market_snapshot,
)
from volatility_forecasting.universe_v8 import UniverseMember, build_universe_manifest

from backend.panel.snapshots import PanelValidationError


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


def _member(ticker: str, mic: str = "XNAS") -> UniverseMember:
    return UniverseMember(
        security_id=f"pit:{ticker}",
        ticker=ticker,
        company_name=f"{ticker} Corp",
        isin=None,
        figi=None,
        cik=None,
        primary_exchange_mic=mic,
        currency="GBX" if mic == "XLON" else "USD",
        timezone="Europe/London" if mic == "XLON" else "America/New_York",
        sector="Technology",
        source="pit-provider",
        source_snapshot_id="pit-v1",
        required_history_sessions=60,
    )


def _diagnostic_universe(*tickers: str) -> dict:
    members = [
        _member(ticker, ("XNAS", "XNYS", "XLON")[index % 3]) for index, ticker in enumerate(tickers)
    ]
    return build_universe_manifest(
        members,
        source_checksums={"archive": "sha256:" + "a" * 64},
        source_attestations={"pit-provider": _attestation()},
        selection_policy={"allow_sparse": True},
    )


def _certifiable_universe() -> dict:
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
                        ({"index": "SP500", "membership_start": "2020-01-01"},)
                        if mic != "XLON" and index < 5
                        else ()
                    ),
                    currency="GBX" if mic == "XLON" else "USD",
                    timezone="Europe/London" if mic == "XLON" else "America/New_York",
                    sector=("Technology", "Energy", "Industrials")[index % 3],
                    source="pit-provider",
                    source_snapshot_id="pit-v1",
                    required_history_sessions=756,
                )
            )
    return build_universe_manifest(
        members,
        source_checksums={"archive": "sha256:" + "a" * 64},
        source_attestations={"pit-provider": _attestation()},
        selection_policy={"required_holdouts": ["NMM", "MSFT"]},
    )


def _raw_provider_frame(rows: int = 80) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-02", periods=rows)
    raw_close = 100.0 * np.exp(np.linspace(0.0, 0.1, rows))
    adjustment = np.ones(rows)
    adjustment[: rows // 2] = 0.5
    return pd.DataFrame(
        {
            "Open": raw_close * 0.99,
            "High": raw_close * 1.01,
            "Low": raw_close * 0.98,
            "Close": raw_close,
            "Adj Close": raw_close * adjustment,
            "Volume": np.linspace(1_000_000, 2_000_000, rows),
            "Dividends": np.zeros(rows),
            "Stock Splits": np.where(np.arange(rows) == rows // 2, 2.0, 0.0),
        },
        index=index,
    )


def test_normalization_preserves_raw_and_adjusted_prices() -> None:
    raw = _raw_provider_frame()
    normalized = normalize_v8_provider_frame(raw)

    assert tuple(normalized.columns) == V8_MARKET_COLUMNS
    np.testing.assert_allclose(normalized["RawClose"], raw["Close"])
    np.testing.assert_allclose(normalized["Close"], raw["Adj Close"])
    np.testing.assert_allclose(normalized["Open"], raw["Open"] * raw["Adj Close"] / raw["Close"])
    assert "RawClose" in canonical_v8_market_csv(normalized).splitlines()[0]


def test_incomplete_snapshot_is_diagnostic_and_cannot_verify_as_certifiable(
    tmp_path: Path,
) -> None:
    universe = _diagnostic_universe("MSFT", "NMM", "VOD.L")
    frames = {"MSFT": normalize_v8_provider_frame(_raw_provider_frame())}
    output = write_v8_market_snapshot(
        tmp_path,
        frames,
        universe_manifest=universe,
        provider="fixture",
        provider_snapshot_id="fixture-2026-08-27",
        provider_license_id="fixture-license",
        license_acknowledged=True,
        allow_incomplete_diagnostic=True,
    )

    manifest, loaded = verify_v8_market_snapshot(
        output, universe_manifest=universe, require_certifiable=False
    )
    assert set(loaded) == {"MSFT"}
    assert manifest["v8_market"]["coverage_certifiable"] is False
    assert "missing_universe_tickers" in manifest["v8_market"]["coverage_reasons"]
    with pytest.raises(PanelValidationError, match="not certifiable|diagnostic-only"):
        verify_v8_market_snapshot(output, universe_manifest=universe)


def test_legacy_adjusted_only_panel_cannot_be_certifiable() -> None:
    universe = _diagnostic_universe("MSFT")
    complete = normalize_v8_provider_frame(_raw_provider_frame())
    legacy = complete[["Open", "High", "Low", "Close", "Volume"]]

    manifest = build_v8_market_snapshot(
        {"MSFT": legacy},
        universe_manifest=universe,
        provider="fixture",
        provider_snapshot_id="fixture-v1",
        provider_license_id="fixture-license",
        license_acknowledged=True,
        allow_incomplete_diagnostic=True,
    )

    assert manifest["v8_market"]["coverage_certifiable"] is False
    assert "raw_and_adjusted_history_not_preserved" in manifest["v8_market"]["coverage_reasons"]


def test_complete_attested_four_market_snapshot_is_certifiable() -> None:
    universe = _certifiable_universe()
    frame = normalize_v8_provider_frame(_raw_provider_frame(rows=756))
    frames = {member["ticker"]: frame for member in universe["members"]}

    manifest = build_v8_market_snapshot(
        frames,
        universe_manifest=universe,
        provider="fixture",
        provider_snapshot_id="fixture-v1",
        provider_license_id="fixture-license",
        license_acknowledged=True,
    )

    assert manifest["v8_market"]["coverage_certifiable"] is True
    assert manifest["ticker_count"] == 75
    assert manifest["v8_market"]["coverage_reasons"] == []


def test_snapshot_rejects_universe_identity_tampering(tmp_path: Path) -> None:
    universe = _diagnostic_universe("MSFT")
    output = write_v8_market_snapshot(
        tmp_path,
        {"MSFT": normalize_v8_provider_frame(_raw_provider_frame())},
        universe_manifest=universe,
        provider="fixture",
        provider_snapshot_id="fixture-v1",
        provider_license_id="fixture-license",
        license_acknowledged=True,
        allow_incomplete_diagnostic=True,
    )
    tampered = json.loads(json.dumps(universe))
    tampered["members"][0]["security_id"] = "different"
    with pytest.raises(ValueError, match="content or checksum"):
        verify_v8_market_snapshot(output, universe_manifest=tampered, require_certifiable=False)
