from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from volatility_forecasting.news_alignment import market_close_news_origins, validate_news_coverage


def test_nyse_close_alignment_respects_daylight_saving_time() -> None:
    origins = market_close_news_origins(
        np.array(["MSFT", "NMM"]),
        np.array(["2025-01-02", "2025-07-01"], dtype="datetime64[D]"),
    )
    assert origins[0].cutoff_at == pd.Timestamp("2025-01-02T21:00:00Z")
    assert origins[1].cutoff_at == pd.Timestamp("2025-07-01T20:00:00Z")


def test_non_session_origin_fails_closed() -> None:
    with pytest.raises(ValueError, match="absent from NYSE schedule"):
        market_close_news_origins(
            np.array(["MSFT"]),
            np.array(["2025-01-04"], dtype="datetime64[D]"),
        )


def test_provider_coverage_must_span_every_origin_and_lookback() -> None:
    cutoffs = np.array(
        ["2025-01-08T21:00:00", "2025-01-10T21:00:00"],
        dtype="datetime64[ns]",
    )
    validate_news_coverage(
        {
            "coverage_start": "2025-01-01T00:00:00Z",
            "coverage_end_exclusive": "2025-01-11T00:00:00Z",
        },
        cutoffs,
    )
    with pytest.raises(ValueError, match="initial lookback"):
        validate_news_coverage(
            {
                "coverage_start": "2025-01-02T00:00:00Z",
                "coverage_end_exclusive": "2025-01-11T00:00:00Z",
            },
            cutoffs,
        )
    with pytest.raises(ValueError, match="final forecast"):
        validate_news_coverage(
            {
                "coverage_start": "2025-01-01T00:00:00Z",
                "coverage_end_exclusive": "2025-01-10T21:00:00Z",
            },
            cutoffs,
        )
