from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from config import FEATURES
from services.training_data import build_training_snapshot, validate_training_snapshot


def _market_snapshot(rows=140):
    dates = pd.date_range("2025-01-01", periods=rows, freq="B")
    values = np.arange(rows, dtype=float) + 100
    frame = pd.DataFrame({name: values + index for index, name in enumerate(FEATURES)}, index=dates)
    return frame, values, dates, {"snapshot_id": "upstream", "market_context": {}}


def test_training_snapshot_preserves_schema_and_has_stable_fingerprint():
    payload = _market_snapshot()
    with (
        patch("services.training_data.fetch_data", return_value=payload),
        patch(
            "services.training_data.future_trading_dates",
            return_value=(["2025-07-15"] * 30, "XNYS"),
        ),
    ):
        first = build_training_snapshot("MSFT")
        second = build_training_snapshot("MSFT")

    assert first["feature_names"] == list(FEATURES)
    assert first["close_index"] == FEATURES.index("Close")
    assert first["window_size"] == 60
    assert first["output_width"] == 30
    assert first["snapshot_id"] == second["snapshot_id"]
    assert len(first["features"]) == len(first["dates"]) == 140
    validate_training_snapshot(first)


def test_training_snapshot_rejects_non_finite_values():
    frame, closes, dates, metadata = _market_snapshot()
    frame.iloc[100, 0] = np.nan
    with (
        patch("services.training_data.fetch_data", return_value=(frame, closes, dates, metadata)),
        pytest.raises(ValueError, match="non-finite"),
    ):
        build_training_snapshot("MSFT")


def test_training_snapshot_bounds_rows_and_horizon():
    payload = _market_snapshot(rows=2200)
    with (
        patch("services.training_data.fetch_data", return_value=payload),
        patch("services.training_data.future_trading_dates", return_value=([], "XNYS")),
    ):
        result = build_training_snapshot("MSFT", days=1)
    assert len(result["features"]) == 2000
    assert len(result["future_dates"]) == 0

    with (
        patch("services.training_data.fetch_data", return_value=payload),
        pytest.raises(ValueError, match="horizon"),
    ):
        build_training_snapshot("MSFT", days=31)
