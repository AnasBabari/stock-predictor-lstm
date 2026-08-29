"""Tests for 70/15/15 unique-origin splitter with purge and embargo."""

from __future__ import annotations

import pandas as pd

from research.volatility_forecasting.splits_v10 import (
    UniqueOriginSplitterV10,
)


def test_unique_origin_splitter_enforces_embargo_and_purge() -> None:
    # 500 business days
    dates = pd.date_range("2020-01-01", periods=500, freq="B").strftime("%Y-%m-%d").tolist()
    tickers = ["AAPL", "AMZN", "MSFT", "PEP"]

    rows = []
    for d in dates:
        for t in tickers:
            rows.append({"Date": d, "SecurityID": t, "Close": 100.0})
    df = pd.DataFrame(rows)

    splitter = UniqueOriginSplitterV10(
        train_fraction=0.70,
        val_fraction=0.15,
        test_fraction=0.15,
        embargo_sessions=30,
        max_label_horizon=30,
    )
    train_dates, val_dates, test_dates = splitter.partition_sessions(dates)

    # 1. No overlap between train, val, test
    assert set(train_dates).isdisjoint(set(val_dates))
    assert set(val_dates).isdisjoint(set(test_dates))
    assert set(train_dates).isdisjoint(set(test_dates))

    # 2. Embargo gap between train end and val start
    train_end = pd.to_datetime(train_dates[-1])
    val_start = pd.to_datetime(val_dates[0])
    gap_days_1 = (val_start - train_end).days
    assert gap_days_1 >= 30, f"Gap between train and val must be >= 30 days, got {gap_days_1}"

    # 3. Embargo gap between val end and test start
    val_end = pd.to_datetime(val_dates[-1])
    test_start = pd.to_datetime(test_dates[0])
    gap_days_2 = (test_start - val_end).days
    assert gap_days_2 >= 30, f"Gap between val and test must be >= 30 days, got {gap_days_2}"

    # 4. Build assignment and verify transfer holdouts and purged rows
    df_assigned, assignment = splitter.build_assignment(df)
    assert assignment.transfer_rows > 0  # MSFT and PEP are required transfer assets
    assert assignment.purged_rows > 0  # Embargoed sessions are marked embargo_purged
    assert len(assignment.assignment_fingerprint_sha256) == 64
