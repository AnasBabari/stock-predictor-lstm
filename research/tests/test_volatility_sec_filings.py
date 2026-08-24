from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest
from volatility_forecasting.news import NewsValidationError
from volatility_forecasting.sec_filings import (
    parse_sec_submissions_recent,
    sec_filing_to_news_event,
)


def _payload() -> dict[str, object]:
    return {
        "cik": "789019",
        "filings": {
            "recent": {
                "accessionNumber": ["0000789019-25-000001", "0000789019-25-000002"],
                "filingDate": ["2025-01-02", "2025-01-03"],
                "acceptanceDateTime": ["2025-01-02T16:05:00-05:00", "2025-01-03T09:00:00-05:00"],
                "form": ["8-K", "4"],
                "items": ["2.02,5.02", ""],
                "primaryDocument": ["report.htm", "ownership.xml"],
            }
        },
    }


def test_sec_submissions_parser_filters_forms_and_preserves_acceptance_time() -> None:
    records = parse_sec_submissions_recent(_payload(), ticker="msft")
    assert len(records) == 1
    assert records[0].ticker == "MSFT"
    assert records[0].accepted_at == pd.Timestamp("2025-01-02T21:05:00Z")
    event = sec_filing_to_news_event(records[0])
    assert event.eligible_at == pd.Timestamp("2025-01-02T21:20:00Z")
    assert event.tickers == ("MSFT",)
    assert {"earnings", "management_change"}.issubset(event.topics)
    assert event.neutral_probability == 1.0


def test_sec_adapter_rejects_misaligned_arrays_and_negative_delay() -> None:
    payload = _payload()
    payload["filings"]["recent"]["form"].append("10-Q")
    with pytest.raises(NewsValidationError, match="not aligned"):
        parse_sec_submissions_recent(payload, ticker="MSFT")

    record = parse_sec_submissions_recent(_payload(), ticker="MSFT")[0]
    with pytest.raises(ValueError, match="cannot be negative"):
        sec_filing_to_news_event(record, conservative_availability_delay=-timedelta(minutes=1))

    invalid_cik = _payload()
    invalid_cik["cik"] = "not-a-cik"
    with pytest.raises(NewsValidationError, match="invalid CIK"):
        parse_sec_submissions_recent(invalid_cik, ticker="MSFT")
