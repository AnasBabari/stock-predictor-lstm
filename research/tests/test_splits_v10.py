"""Tests for V10 unique-origin session 70/15/15 partitioning."""

from __future__ import annotations

import pandas as pd
import pytest

from research.volatility_forecasting.splits_v10 import (
    UniqueOriginSplitterV10,
)


@pytest.fixture
def synthetic_panel() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=1000, freq="B").strftime("%Y-%m-%d")
    tickers = ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "PEP", "TXN", "INTC"]
    records = []
    for d in dates:
        for t in tickers:
            records.append({"Date": d, "Ticker": t, "Feature": 1.0})
    return pd.DataFrame(records)


def test_unique_origin_partition_ratios_and_isolation(synthetic_panel: pd.DataFrame) -> None:
    splitter = UniqueOriginSplitterV10(
        train_fraction=0.70,
        val_fraction=0.15,
        test_fraction=0.15,
        embargo_sessions=30,
        required_transfer_tickers=("GOOGL", "MSFT", "NVDA", "PEP", "TXN"),
    )
    assignment = splitter.build_assignment(synthetic_panel)

    assert len(assignment.train_sessions) == 700
    assert len(assignment.val_sessions) == 150
    assert len(assignment.test_sessions) == 150
    assert (
        len(assignment.train_sessions)
        + len(assignment.val_sessions)
        + len(assignment.test_sessions)
        == 1000
    )

    # No overlap in session dates
    train_set = set(assignment.train_sessions)
    val_set = set(assignment.val_sessions)
    test_set = set(assignment.test_sessions)
    assert not (train_set & val_set)
    assert not (val_set & test_set)
    assert not (train_set & test_set)

    # Transfer rows are accounted for
    assert assignment.transfer_rows > 0
    assert len(assignment.assignment_fingerprint_sha256) == 64
