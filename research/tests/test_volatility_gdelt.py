from __future__ import annotations

from datetime import UTC, datetime

import pytest
from volatility_forecasting.gdelt import (
    gdelt_row_to_news_event,
    iter_gdelt_v2_archives,
    parse_gdelt_v2_export_line,
)
from volatility_forecasting.news import NewsValidationError


def _line() -> str:
    fields = [""] * 61
    fields[0] = "123456789"
    fields[1] = "20190101"
    fields[6] = "IRAN"
    fields[16] = "MICROSOFT CORPORATION"
    fields[26] = "190"
    fields[27] = "190"
    fields[28] = "19"
    fields[29] = "4"
    fields[30] = "-8.0"
    fields[31] = "12"
    fields[32] = "4"
    fields[33] = "8"
    fields[34] = "-6.5"
    fields[59] = "20200101001500"
    fields[60] = "https://example.com/news/microsoft-iran"
    return "\t".join(fields)


def test_gdelt_row_uses_first_seen_not_historical_event_date() -> None:
    row = parse_gdelt_v2_export_line(_line())
    assert row.sql_date == "20190101"
    assert row.date_added.isoformat() == "2020-01-01T00:15:00+00:00"
    event = gdelt_row_to_news_event(
        row,
        ticker_aliases={"MSFT": ("Microsoft Corporation",)},
    )
    assert event.eligible_at == row.date_added
    assert event.timestamp_quality == "first_seen_only"
    assert event.tickers == ("MSFT",)
    assert "military_conflict" in event.topics
    assert event.negative_probability > event.positive_probability
    assert sum(
        (
            event.positive_probability,
            event.neutral_probability,
            event.negative_probability,
        )
    ) == pytest.approx(1.0)


def test_gdelt_archive_plan_is_utc_aligned_and_half_open() -> None:
    files = iter_gdelt_v2_archives(
        datetime(2025, 1, 1, 0, 0, tzinfo=UTC),
        datetime(2025, 1, 1, 0, 45, tzinfo=UTC),
    )
    assert len(files) == 3
    assert files[0].url.endswith("20250101000000.export.CSV.zip")
    assert files[-1].url.endswith("20250101003000.export.CSV.zip")
    with pytest.raises(ValueError, match="timezone-aware"):
        iter_gdelt_v2_archives(datetime(2025, 1, 1), datetime(2025, 1, 2))


def test_gdelt_parser_rejects_schema_drift_and_short_aliases() -> None:
    with pytest.raises(NewsValidationError, match="61 columns"):
        parse_gdelt_v2_export_line("too\tshort")
    row = parse_gdelt_v2_export_line(_line())
    with pytest.raises(NewsValidationError, match="at least three"):
        gdelt_row_to_news_event(row, ticker_aliases={"C": ("C",)})
