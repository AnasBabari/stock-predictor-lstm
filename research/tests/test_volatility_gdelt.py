from __future__ import annotations

from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
import pytest
from volatility_forecasting.gdelt import (
    aggregate_gdelt_v1_daily_lines,
    gdelt_row_to_news_event,
    gdelt_v1_row_to_news_event,
    iter_gdelt_v1_daily_archives,
    iter_gdelt_v2_archives,
    parse_gdelt_v1_export_line,
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


def _v1_line() -> str:
    fields = [""] * 58
    fields[0] = "123456"
    fields[1] = "20250104"
    fields[6] = "MICROSOFT"
    fields[16] = "IRAN"
    fields[26] = "190"
    fields[27] = "190"
    fields[28] = "19"
    fields[29] = "4"
    fields[30] = "-9.0"
    fields[31] = "20"
    fields[32] = "5"
    fields[33] = "20"
    fields[34] = "-6.5"
    fields[56] = "20250105"
    fields[57] = "https://example.com/news/microsoft-iran"
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


def test_daily_archive_uses_next_day_noon_as_conservative_availability() -> None:
    archives = iter_gdelt_v1_daily_archives(date(2025, 1, 5), date(2025, 1, 7))
    assert len(archives) == 2
    assert archives[0].available_at == datetime(2025, 1, 6, 12, tzinfo=UTC)
    assert archives[0].url.endswith("20250105.export.CSV.zip")


def test_daily_event_never_uses_event_date_as_information_time() -> None:
    row = parse_gdelt_v1_export_line(_v1_line())
    archive = iter_gdelt_v1_daily_archives(date(2025, 1, 5), date(2025, 1, 6))[0]
    event = gdelt_v1_row_to_news_event(
        row,
        archive=archive,
        ticker_aliases={"MSFT": ("Microsoft",)},
    )
    assert event.first_seen_at == pd.Timestamp("2025-01-06T12:00:00Z")
    assert event.eligible_at == event.first_seen_at
    assert event.tickers == ("MSFT",)
    assert "military_conflict" in event.topics


def test_daily_archive_is_compressed_into_market_and_ticker_events() -> None:
    ticker_line = _v1_line()
    oil_fields = _v1_line().split("\t")
    oil_fields[0] = "987654"
    oil_fields[6] = "OPEC OIL"
    oil_fields[16] = "PETROLEUM MARKET"
    oil_fields[26] = "010"
    oil_fields[27] = "010"
    oil_fields[28] = "01"
    oil_fields[57] = "https://example.com/news/opec-supply"
    archive = iter_gdelt_v1_daily_archives(date(2025, 1, 5), date(2025, 1, 6))[0]
    events, stats = aggregate_gdelt_v1_daily_lines(
        [ticker_line, "\t".join(oil_fields)],
        archive=archive,
        ticker_aliases={"MSFT": ("Microsoft",)},
    )
    assert stats.total_rows == 2
    assert stats.retained_rows == 2
    assert stats.output_events == 2
    market = next(event for event in events if not event.tickers)
    ticker = next(event for event in events if event.tickers == ("MSFT",))
    assert "oil_supply" in market.topics
    assert ticker.volume == 1.0
    assert all(event.first_seen_at == pd.Timestamp("2025-01-06T12:00:00Z") for event in events)


def test_gdelt_parser_rejects_schema_drift_and_short_aliases() -> None:
    with pytest.raises(NewsValidationError, match="61 columns"):
        parse_gdelt_v2_export_line("too\tshort")
    row = parse_gdelt_v2_export_line(_line())
    with pytest.raises(NewsValidationError, match="at least three"):
        gdelt_row_to_news_event(row, ticker_aliases={"C": ("C",)})
    with pytest.raises(NewsValidationError, match="58 columns"):
        parse_gdelt_v1_export_line("too\tshort")


def test_daily_aggregate_survives_weighted_share_drift_near_the_simplex_edge() -> None:
    """Regression: weighted sentiment means must never escape [0, 1] by rounding."""
    from volatility_forecasting.gdelt import _clipped_weighted_share

    drifted = np.asarray([0.5 + 8e-17, 0.5 + 8e-17], dtype=np.float64)
    assert _clipped_weighted_share(drifted, [1.0, 1.0]) == 1.0
    assert _clipped_weighted_share(drifted, [-1.0, -1.0]) == 0.0

    archive = iter_gdelt_v1_daily_archives(date(2025, 1, 5), date(2025, 1, 6))[0]
    for rows in range(1, 41):
        events, stats = aggregate_gdelt_v1_daily_lines(
            [_v1_line()] * rows,
            archive=archive,
            ticker_aliases={"MSFT": ("Microsoft",)},
        )
        assert stats.output_events >= 1
        for event in events:
            probabilities = (
                event.positive_probability,
                event.neutral_probability,
                event.negative_probability,
            )
            assert all(0.0 <= value <= 1.0 for value in probabilities)
            assert sum(probabilities) == pytest.approx(1.0)
