"""Tests for point-in-time Nasdaq-100 universe reconstruction and calendar alignment."""

from __future__ import annotations

import pandas as pd
import pytest

from research.ndx100.universe import (
    BASE_CONSTITUENTS_2021_12_31,
    assert_survivorship_bias_resistant,
    get_membership_changes,
    get_ndx100_constituents,
    get_ndx100_membership_timeline,
    get_weekly_origins,
    verify_membership_source,
)


def test_base_constituents_count_and_uniqueness() -> None:
    assert len(BASE_CONSTITUENTS_2021_12_31) == len(set(BASE_CONSTITUENTS_2021_12_31))
    assert len(BASE_CONSTITUENTS_2021_12_31) >= 100
    for ticker in BASE_CONSTITUENTS_2021_12_31:
        assert ticker.isupper()
        assert ticker.isalpha() or "-" in ticker or "." in ticker


def test_membership_changes_loading() -> None:
    df = get_membership_changes()
    assert not df.empty
    assert {"effective_date", "action", "ticker"}.issubset(df.columns)
    assert set(df["action"].unique()).issubset({"ADD", "REMOVE", "TICKER_CHANGE"})
    assert df["effective_date"].is_monotonic_increasing


def test_membership_source_is_content_addressed_and_development_only() -> None:
    manifest = verify_membership_source()
    assert manifest["row_count"] == 105
    assert manifest["certification_eligible"] is False


def test_meta_ticker_change_preserves_point_in_time_symbol() -> None:
    before = get_ndx100_constituents("2022-06-08")
    after = get_ndx100_constituents("2022-06-09")
    assert "FB" in before and "META" not in before
    assert "META" in after and "FB" not in after


def test_fiserv_uses_historical_symbol_until_nasdaq_removal() -> None:
    before = get_ndx100_constituents("2023-06-06")
    after = get_ndx100_constituents("2023-06-07")
    assert "FISV" in before and "FI" not in before
    assert "FISV" not in after and "FI" not in after


def test_point_in_time_constituents_early_2022() -> None:
    constituents = get_ndx100_constituents("2022-01-14")
    assert "PTON" in constituents
    assert "XLNX" in constituents
    assert "OKTA" in constituents
    assert "ODFL" not in constituents
    assert "AZN" not in constituents


def test_point_in_time_removal_pton() -> None:
    before = get_ndx100_constituents("2022-01-21")
    after = get_ndx100_constituents("2022-01-24")
    assert "PTON" in before
    assert "PTON" not in after
    assert "ODFL" not in before
    assert "ODFL" in after


def test_point_in_time_removal_xlnx() -> None:
    before = get_ndx100_constituents("2022-02-18")
    after = get_ndx100_constituents("2022-02-22")
    assert "XLNX" in before
    assert "XLNX" not in after
    assert "AZN" not in before
    assert "AZN" in after


def test_point_in_time_constituents_2026() -> None:
    constituents = get_ndx100_constituents("2026-07-24")
    # Historical removals should not be in 2026
    for removed in ("PTON", "XLNX", "OKTA", "SPLK", "SIRI", "WBA", "DLTR"):
        assert removed not in constituents, f"{removed} should be excluded in 2026"
    # Recent additions should be present in 2026
    for added in ("ARM", "APP", "PLTR", "MSTR", "AXON", "SHOP"):
        assert added in constituents, f"{added} should be present in 2026"


def test_membership_timeline() -> None:
    timeline = get_ndx100_membership_timeline()
    assert not timeline.empty
    assert len(timeline) >= 20
    assert "effective_date" in timeline.columns
    assert "additions" in timeline.columns
    assert "removals" in timeline.columns
    assert "active_count" in timeline.columns
    # Check that counts remain bounded near 100
    assert (timeline["active_count"] >= 100).all()
    assert (timeline["active_count"] <= 110).all()


def test_weekly_origins_calendar_alignment() -> None:
    origins = get_weekly_origins("2022-01-01", "2026-08-28")
    assert len(origins) >= 230
    for origin, targets in origins:
        assert isinstance(origin, pd.Timestamp)
        assert len(targets) == 5
        assert all(t > origin for t in targets)
        # Verify strictly increasing targets
        for i in range(len(targets) - 1):
            assert targets[i] < targets[i + 1]


def test_assert_survivorship_bias_resistant() -> None:
    origins = get_weekly_origins("2022-01-01", "2026-08-28")
    constituents_by_origin = {origin: get_ndx100_constituents(origin) for origin, _ in origins}
    # Should pass without raising
    assert_survivorship_bias_resistant(constituents_by_origin)

    # Biased universe (today's constituents everywhere) should fail
    today_constituents = get_ndx100_constituents("2026-07-24")
    biased_by_origin = {origin: today_constituents for origin, _ in origins}
    with pytest.raises(
        AssertionError, match="No historical removals observed|PTON must be present"
    ):
        assert_survivorship_bias_resistant(biased_by_origin)
