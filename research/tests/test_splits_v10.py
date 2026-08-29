"""Tests for 70/15/15 unique-origin splitter with purge, embargo and sealed target store."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from research.volatility_forecasting.splits_v10 import (
    SealedTargetStore,
    UniqueOriginSplitterV10,
)


def test_unique_origin_splitter_enforces_embargo_and_purge() -> None:
    dates = pd.date_range("2020-01-01", periods=500, freq="B").strftime("%Y-%m-%d").tolist()
    security_ids = ["SEC_AAPL_001", "SEC_AMZN_001", "SEC_MSFT_001", "SEC_PEP_001"]

    rows = []
    for d in dates:
        for sec in security_ids:
            rows.append({"Date": d, "SecurityID": sec, "Close": 100.0})
    df = pd.DataFrame(rows)

    splitter = UniqueOriginSplitterV10(
        train_fraction=0.70,
        val_fraction=0.15,
        test_fraction=0.15,
        embargo_sessions=30,
        max_label_horizon=30,
        required_transfer_security_ids=("SEC_MSFT_001", "SEC_PEP_001"),
    )
    train_dates, val_dates, test_dates = splitter.partition_sessions(dates)

    assert set(train_dates).isdisjoint(set(val_dates))
    assert set(val_dates).isdisjoint(set(test_dates))
    assert set(train_dates).isdisjoint(set(test_dates))

    df_assigned, assignment = splitter.build_assignment(df)
    assert assignment.transfer_rows > 0
    assert assignment.purged_rows > 0
    assert len(assignment.assignment_fingerprint_sha256) == 64


def test_splitter_rejects_missing_security_id_and_duplicate_identities() -> None:
    splitter = UniqueOriginSplitterV10()

    # Missing SecurityID
    bad_df1 = pd.DataFrame({"Date": ["2024-01-01"], "Ticker": ["AAPL"]})
    with pytest.raises(ValueError, match="Required column 'SecurityID' missing"):
        splitter.build_assignment(bad_df1)

    # Duplicate (SecurityID, Date)
    bad_df2 = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-01"],
            "SecurityID": ["SEC_AAPL_001", "SEC_AAPL_001"],
        }
    )
    with pytest.raises(ValueError, match="Duplicate"):
        splitter.build_assignment(bad_df2)


def test_assignment_fingerprint_is_row_order_invariant() -> None:
    dates = pd.date_range("2020-01-01", periods=300, freq="B").strftime("%Y-%m-%d").tolist()
    rows = []
    for d in dates:
        for sec in ["SEC_AAPL_001", "SEC_NVDA_001"]:
            rows.append({"Date": d, "SecurityID": sec, "Close": 100.0})

    df1 = pd.DataFrame(rows)
    df2 = df1.sample(frac=1.0, random_state=123).reset_index(drop=True)

    splitter = UniqueOriginSplitterV10(required_transfer_security_ids=("SEC_NVDA_001",))
    _, a1 = splitter.build_assignment(df1)
    _, a2 = splitter.build_assignment(df2)

    assert a1.assignment_fingerprint_sha256 == a2.assignment_fingerprint_sha256


def test_sealed_target_store_creation_and_tamper_detection(tmp_path: Path) -> None:
    target_data = pd.DataFrame(
        {
            "Date": ["2024-01-02", "2024-01-03"],
            "SecurityID": ["SEC_AAPL_001", "SEC_AAPL_001"],
            "target_h1": [0.0004, 0.0005],
        }
    )
    store_file = tmp_path / "sealed_targets.json"
    store, meta = SealedTargetStore.create_sealed_store(target_data, store_file)

    assert store_file.exists()
    assert meta.row_count == 2
    assert len(meta.checksum_sha256) == 64

    # Load successfully with verified checksum
    df_loaded = store.load_targets(expected_checksum=meta.checksum_sha256)
    assert len(df_loaded) == 2

    # Tamper with file
    store_file.write_bytes(b"[tampered]")
    with pytest.raises(ValueError, match="checksum mismatch"):
        store.load_targets(expected_checksum=meta.checksum_sha256)
