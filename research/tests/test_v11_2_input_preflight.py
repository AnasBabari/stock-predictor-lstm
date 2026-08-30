"""Tests for the real-input V11.2 preflight boundary."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from research.volatility_forecasting.v11_2_protocol import (
    V11_2_HORIZONS,
    V11_2_PROTOCOL_ID,
    V112Protocol,
    feature_schema_digest,
)
from research.volatility_forecasting.v11_2_universe import (
    MembershipInterval,
    PITSecurity,
    TickerInterval,
    V112UniverseManifest,
    build_universe_manifest,
    save_universe_manifest,
)
from scripts.check_v11_2_inputs import check_inputs

_SECTORS = (
    "semiconductors_hardware",
    "software_cloud",
    "communications_media",
    "healthcare_biotech",
    "industrials_transportation",
    "consumer_discretionary",
    "consumer_staples_defensive",
    "energy_utilities_financial_adjacent",
)


def _universe() -> V112UniverseManifest:
    securities = []
    for index in range(64):
        ticker = f"T{index:03d}"
        securities.append(
            PITSecurity(
                security_id=f"US.TEST{index:03d}",
                cik=f"{index + 1:010d}",
                figi=f"BBG00TEST{index:03d}",
                exchange_mic="XNAS",
                sector=_SECTORS[index // 8],
                industry="test",
                volatility_stratum="low" if index % 2 else "high",
                market_cap_stratum="large" if index % 3 else "mid",
                ticker_intervals=(TickerInterval(ticker, "2020-01-01", "2030-12-31"),),
                membership_intervals=(
                    MembershipInterval("2020-01-01", "2030-12-31", "test-source", "a" * 64),
                ),
            )
        )
    return build_universe_manifest(
        securities,
        protocol_id=V11_2_PROTOCOL_ID,
        membership_sources=["test-source"],
    )


def test_missing_real_inputs_fail_closed(tmp_path: Path) -> None:
    report = check_inputs(
        panel_path=tmp_path / "missing.npz",
        universe_path=tmp_path / "missing-universe.json",
        key_path=tmp_path / "private" / "missing.key",
        repository_root=tmp_path / "repository",
    )
    assert report["ready"] is False
    assert not all(item["passed"] for item in report["checks"])


def test_valid_panel_and_external_key_pass_preflight(tmp_path: Path) -> None:
    protocol = V112Protocol()
    universe = _universe()
    universe_path = tmp_path / "universe.json"
    save_universe_manifest(universe, universe_path)
    security_ids = [security.security_id for security in universe.securities]
    dates = ["2025-01-02"] * len(security_ids)
    panel_path = tmp_path / "pit64.npz"
    np.savez_compressed(
        panel_path,
        dates=np.asarray(dates, dtype="U10"),
        security_ids=np.asarray(security_ids, dtype="U128"),
        features=np.zeros(
            (64, protocol.window_size, len(protocol.feature_names)), dtype=np.float32
        ),
        returns=np.zeros((64, len(V11_2_HORIZONS)), dtype=np.float32),
        rv=np.ones((64, len(V11_2_HORIZONS)), dtype=np.float32),
        feature_names=np.asarray(protocol.feature_names, dtype="U64"),
        horizons=np.asarray(V11_2_HORIZONS, dtype=np.int64),
    )
    panel_manifest = {
        "protocol_id": protocol.protocol_id,
        "schema_sha256": feature_schema_digest(protocol),
        "universe_manifest_sha256": universe.manifest_sha256,
        "stock_origin_observations": 64,
        "unique_sessions": 1,
        "snapshot_manifest_sha256": "b" * 64,
    }
    (panel_path.with_suffix(panel_path.suffix + ".manifest.json")).write_text(
        json.dumps(panel_manifest), encoding="utf-8"
    )
    key_path = tmp_path / "private" / "holdout.key"
    key_path.parent.mkdir()
    key_path.write_bytes(b"k" * 32)
    report = check_inputs(
        panel_path=panel_path,
        universe_path=universe_path,
        key_path=key_path,
        repository_root=tmp_path / "repository",
    )
    assert report["ready"] is True
    assert report["panel_summary"]["security_count"] == 64


def test_secondary_ndx_cache_is_rejected(tmp_path: Path) -> None:
    cache_root = tmp_path / "data" / "ndx100" / "cache"
    panel = cache_root / "panel.npz"
    report = check_inputs(
        panel_path=panel,
        universe_path=tmp_path / "universe.json",
        key_path=tmp_path / "private.key",
        repository_root=tmp_path,
    )
    check = next(
        item for item in report["checks"] if item["name"] == "secondary_ndx100_cache_rejected"
    )
    assert check["passed"] is False
