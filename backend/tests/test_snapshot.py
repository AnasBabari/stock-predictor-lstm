import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from snapshot import (
    create_market_snapshot,
    load_market_snapshot,
    normalise_ticker,
    validate_market_frame,
)


def _frame(rows=8):
    index = pd.date_range("2025-01-01", periods=rows, freq="B")
    close = np.arange(100.0, 100.0 + rows)
    return pd.DataFrame(
        {"Open": close, "High": close + 1, "Low": close - 1, "Close": close, "Volume": 10},
        index=index,
    )


def test_market_snapshot_is_hash_verified_and_loadable(tmp_path):
    manifest = create_market_snapshot(
        ["aapl", "msft"],
        start="2025-01-01",
        end="2025-02-01",
        output=tmp_path / "market-snapshot",
        downloader=lambda ticker, **_kwargs: _frame(),
    )
    assert manifest["requested"]["tickers"] == ["AAPL", "MSFT"]
    manifest_path = tmp_path / "market-snapshot" / "manifest.json"
    loaded, frames = load_market_snapshot(manifest_path)
    assert loaded["content_sha256"] == manifest["content_sha256"]
    assert set(frames) == {"AAPL", "MSFT"}
    assert list(frames["AAPL"].columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_market_snapshot_rejects_tampered_asset(tmp_path):
    destination = tmp_path / "market-snapshot"
    create_market_snapshot(
        ["AAPL"],
        start="2025-01-01",
        end="2025-02-01",
        output=destination,
        downloader=lambda ticker, **_kwargs: _frame(),
    )
    (destination / "market" / "AAPL.parquet").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="asset hash"):
        load_market_snapshot(destination / "manifest.json")


def test_market_snapshot_manifest_hash_is_verified(tmp_path):
    destination = tmp_path / "market-snapshot"
    create_market_snapshot(
        ["AAPL"],
        start="2025-01-01",
        end="2025-02-01",
        output=destination,
        downloader=lambda ticker, **_kwargs: _frame(),
    )
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["requested"]["start"] = "2000-01-01"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest hash"):
        load_market_snapshot(manifest_path)


def test_market_frame_rejects_duplicate_sessions():
    frame = _frame()
    frame.index = pd.DatetimeIndex([frame.index[0]] * len(frame))
    with pytest.raises(ValueError, match="duplicate"):
        validate_market_frame(frame, "AAPL")


def test_snapshot_rejects_unsafe_identity_and_date_range():
    with pytest.raises(ValueError, match="ticker"):
        normalise_ticker("../../secret")
    with pytest.raises(ValueError, match="precede"):
        create_market_snapshot(
            ["AAPL"],
            start="2025-02-01",
            end="2025-01-01",
            output=Path("unused"),
        )
