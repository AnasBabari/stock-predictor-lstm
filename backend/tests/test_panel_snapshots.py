"""Slice-4 tests: immutable panel snapshot provenance."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from panel.snapshots import (
    LicenseNotAcknowledged,
    PanelValidationError,
    build_snapshot,
    canonical_csv,
    load_snapshot,
    validate_ohlcv,
    write_snapshot,
)


def make_frame(rows: int = 120, start: str = "2022-01-03", drift: float = 0.001) -> pd.DataFrame:
    index = pd.bdate_range(start, periods=rows)
    close = 100.0 * np.exp(np.cumsum(np.full(rows, drift)))
    return pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.005,
            "Low": close * 0.995,
            "Close": close,
            "Volume": np.linspace(1e6, 2e6, rows),
        },
        index=index,
    )


def test_license_gate_blocks_unacknowledged_builds() -> None:
    with pytest.raises(LicenseNotAcknowledged):
        build_snapshot({"MSFT": make_frame()}, license_acknowledged=False)


def test_structural_rejections() -> None:
    frame = make_frame()
    dup = pd.concat([frame, frame.iloc[[-1]]])
    with pytest.raises(PanelValidationError, match="duplicate"):
        validate_ohlcv("X", dup)

    shuffled = frame.iloc[::-1]
    with pytest.raises(PanelValidationError, match="chronological"):
        validate_ohlcv("X", shuffled)

    bad = frame.copy()
    bad.iloc[5, bad.columns.get_loc("Close")] = -1.0
    with pytest.raises(PanelValidationError, match="non-positive"):
        validate_ohlcv("X", bad)

    nan_frame = frame.copy()
    nan_frame.iloc[3, nan_frame.columns.get_loc("Open")] = np.nan
    with pytest.raises(PanelValidationError, match="non-finite"):
        validate_ohlcv("X", nan_frame)

    with pytest.raises(PanelValidationError, match="missing columns"):
        validate_ohlcv("X", frame.drop(columns=["Volume"]))


def test_suspicious_adjustment_rows_flagged_not_rejected() -> None:
    frame = make_frame()
    frame.iloc[60:, :] = frame.iloc[60:, :] * 1.35  # >20% overnight jump
    provenance = validate_ohlcv("SPLIT", frame)
    assert len(provenance.suspicious_adjustment_rows) >= 1


def test_manifest_is_deterministic_and_content_addressed() -> None:
    frames = {"MSFT": make_frame(), "AAPL": make_frame(start="2021-06-01")}
    a = build_snapshot(frames, license_acknowledged=True, extra_metadata={"universe": "test"})
    b = build_snapshot(frames, license_acknowledged=True)
    # Content address independent of retrieval timestamp/extra metadata.
    assert a["panel_id"] == b["panel_id"]
    assert a["pooled_checksum"].startswith("sha256:")
    assert a["provider"] == "yfinance"
    assert a["timezone"] == "America/New_York"
    assert a["adjust_mode"] == "auto_adjust=true"
    assert a["license"]["acknowledged"] is True
    assert set(a["tickers"]) == {"AAPL", "MSFT"}
    # Changing one input changes the pooled address.
    c = build_snapshot(
        {**frames, "AAPL": make_frame(start="2021-07-01")}, license_acknowledged=True
    )
    assert c["panel_id"] != a["panel_id"]


def test_canonical_csv_is_stable_under_column_noise() -> None:
    frame = make_frame(40)
    noisy = frame.copy()
    noisy["Extra"] = 1.0  # non-OHLCV column must not affect the address
    assert canonical_csv(frame) == canonical_csv(noisy)


def test_write_once_roundtrip_and_tamper_detection(tmp_path: Path) -> None:
    root = tmp_path / "panel"
    frames = {"MSFT": make_frame()}
    out = write_snapshot(root, frames, license_acknowledged=True)

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["ticker_count"] == 1

    loaded_manifest, loaded_frames = load_snapshot(out)
    assert loaded_manifest["panel_id"] == manifest["panel_id"]
    pd.testing.assert_frame_equal(loaded_frames["MSFT"], frames["MSFT"], check_freq=False)

    with pytest.raises(FileExistsError, match="immutable"):
        write_snapshot(root, frames, license_acknowledged=True)

    # Tampering is detected on reload.
    csv_path = out / "raw" / "MSFT.csv"
    text = csv_path.read_text(encoding="utf-8").replace("100.", "999.")
    csv_path.write_text(text, encoding="utf-8")
    with pytest.raises(PanelValidationError, match="checksum mismatch|non-positive"):
        load_snapshot(out)
