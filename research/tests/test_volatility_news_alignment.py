from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from volatility_forecasting.news_alignment import market_close_news_origins


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
