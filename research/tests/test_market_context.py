import numpy as np
import pandas as pd
import pytest

from research.price_forecasting.gpu_pipeline import build_global_price_dataset
from research.price_forecasting.market_context import (
    CONTEXT_NAMES,
    append_context,
    build_market_context,
)
from research.price_forecasting.paired_validation import hac_mean, paired_tests, validation_table


def frames(rows=500):
    dates = pd.bdate_range("2020-01-01", periods=rows)
    result = {}
    for k, symbol in enumerate(("A", "B", "C", "D.L", "E.L", "F.L")):
        close = 100 * np.exp(np.cumsum(0.001 + 0.005 * np.sin(np.arange(rows) / (4 + k))))
        result[symbol] = pd.DataFrame(
            {
                "Open": close * 0.999,
                "High": close * 1.01,
                "Low": close * 0.99,
                "Close": close,
                "Volume": 1000 + np.arange(rows),
            },
            index=dates,
        )
    return result


def test_leave_one_out_market_isolation_and_future_causality():
    source = frames()
    original = build_market_context(source)
    changed = {s: f.copy() for s, f in source.items()}
    changed["A"].loc[:, "Close"] *= np.exp(np.arange(500) * 0.01)
    modified = build_market_context(changed)
    pd.testing.assert_series_equal(original["A"].market_return_1d, modified["A"].market_return_1d)
    pd.testing.assert_frame_equal(original["D.L"], modified["D.L"])
    changed = {s: f.copy() for s, f in source.items()}
    for f in changed.values():
        f.iloc[70:, f.columns.get_loc("Close")] *= 3
    modified = build_market_context(changed)
    for s in source:
        pd.testing.assert_frame_equal(original[s].iloc[:70], modified[s].iloc[:70])


def test_missing_breadth_carries_past_with_flag_no_backfill():
    source = frames()
    source["B"].iloc[60, source["B"].columns.get_loc("Volume")] = 0
    context = build_market_context(source)["A"]
    assert context.context_missing.iloc[0] == 1
    assert context.market_return_20d.iloc[0] == 0
    assert context.context_missing.iloc[60] == 1
    assert context.market_return_20d.iloc[60] == context.market_return_20d.iloc[59]
    assert context.context_peer_coverage.iloc[60] == 0.5
    assert context.context_stale_sessions.iloc[60] > 0


def test_append_preserves_every_row_target_and_split():
    source = frames(700)
    dataset = build_global_price_dataset(source)
    augmented = append_context(dataset, source, build_market_context(source))
    for key in (
        "targets",
        "split_train",
        "split_validation",
        "split_test",
        "origin_dates",
        "ticker_indices",
    ):
        np.testing.assert_array_equal(getattr(dataset, key), getattr(augmented, key))
    np.testing.assert_array_equal(augmented.sequences[:, :, :25], dataset.sequences)
    assert augmented.sequences.shape[-1] == 25 + len(CONTEXT_NAMES)
    assert np.isfinite(augmented.sequences).all()
    zeros = np.zeros_like(dataset.targets[dataset.split_validation])
    table = validation_table(dataset, zeros, zeros)
    assert len(table) == len(dataset.split_validation) * 7
    assert not table.duplicated(["sample_idx", "horizon"]).any()


def test_hac_matches_explicit_bartlett_covariance():
    values = np.array([0.1, -0.2, 0.05, 0.15, -0.04, 0.2])
    e = values - values.mean()
    n = len(e)
    weights = np.maximum(0, 1 - np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) / 3)
    expected = np.sqrt(e @ weights @ e / (n * (n - 1)))
    assert hac_mean(values, 2)["se"] == pytest.approx(expected)


def test_date_aggregation_does_not_inflate_count_for_duplicate_stocks():
    dates = pd.bdate_range("2020-01-01", periods=30)
    table = pd.DataFrame(
        {
            "date": dates,
            "market": "US",
            "horizon": 1,
            "y_true": np.sin(np.arange(30)) * 0.01,
            "y_naive": 0.0,
            "y_pred_ridge": 0.001,
            "y_pred_lstm": 0.002,
        }
    )
    one = paired_tests(table)
    two = paired_tests(pd.concat([table, table]))
    assert one == two
    assert {r["bandwidth"] for r in one["results"]} == {0, 1, 5}
    assert all(r["dates"] == 30 for r in one["results"])
