from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from config import FEATURES_V4, MAX_FORECAST_DAYS, SNAPSHOT_SCHEMA_VERSION, WINDOW_SIZE
from services.training_data import build_training_snapshot, validate_training_snapshot


def _market_snapshot(rows=140):
    dates = pd.date_range("2025-01-01", periods=rows, freq="B")
    values = np.arange(rows, dtype=float) + 100
    frame = pd.DataFrame(
        {name: values + index for index, name in enumerate(FEATURES_V4)}, index=dates
    )
    return (
        frame,
        values,
        dates,
        {
            "snapshot_id": "upstream",
            "adjusted_prices": True,
            "market_context": {"sources": {}},
            "quality": {"checks": {}, "issues": [], "status": "clean"},
        },
    )


def test_training_snapshot_preserves_schema_and_has_stable_fingerprint():
    payload = _market_snapshot()
    with (
        patch("services.training_data.fetch_browser_data", return_value=payload),
        patch(
            "services.training_data.future_trading_dates",
            return_value=(
                pd.date_range("2025-08-01", periods=30, freq="B").strftime("%Y-%m-%d").tolist(),
                "XNYS",
            ),
        ),
    ):
        first = build_training_snapshot("MSFT")
        second = build_training_snapshot("MSFT")

    assert first["feature_names"] == list(FEATURES_V4)
    assert first["schema_version"] == SNAPSHOT_SCHEMA_VERSION
    assert "close_index" not in first
    assert first["window_size"] == WINDOW_SIZE
    assert first["output_width"] == MAX_FORECAST_DAYS
    assert first["snapshot_id"] == second["snapshot_id"]
    assert len(first["features"]) == len(first["dates"]) == 140
    validate_training_snapshot(first)


def test_training_snapshot_discloses_source_metadata():
    payload = _market_snapshot(rows=300)
    with (
        patch("services.training_data.fetch_browser_data", return_value=payload),
        patch("services.training_data.future_trading_dates", return_value=([], "XNYS")),
    ):
        result = build_training_snapshot("MSFT")

    snapshot_metadata = result["data_snapshot"]
    assert snapshot_metadata["start_date"] == "2025-01-01"
    assert snapshot_metadata["end_date"] == result["dates"][-1]
    assert snapshot_metadata["raw_rows"] == 300
    assert snapshot_metadata["usable_sequences"] == 300 - WINDOW_SIZE - MAX_FORECAST_DAYS + 1
    assert snapshot_metadata["adjusted_prices"] is True
    assert "quality" in snapshot_metadata


def test_training_snapshot_rejects_non_finite_values():
    frame, closes, dates, metadata = _market_snapshot()
    frame.iloc[100, 0] = np.nan
    with (
        patch(
            "services.training_data.fetch_browser_data",
            return_value=(frame, closes, dates, metadata),
        ),
        pytest.raises(ValueError, match="non-finite"),
    ):
        build_training_snapshot("MSFT")


def test_training_snapshot_bounds_rows_and_horizon():
    payload = _market_snapshot(rows=2200)
    with (
        patch("services.training_data.fetch_browser_data", return_value=payload),
        patch("services.training_data.future_trading_dates", return_value=([], "XNYS")),
    ):
        result = build_training_snapshot("MSFT", days=1)
    assert len(result["features"]) == 2000
    assert result["data_snapshot"]["raw_rows"] == 2200
    assert len(result["future_dates"]) == 0

    with (
        patch("services.training_data.fetch_browser_data", return_value=payload),
        pytest.raises(ValueError, match="horizon"),
    ):
        build_training_snapshot("MSFT", days=31)


def test_training_snapshot_rejects_non_chronological_dates():
    frame, closes, dates, metadata = _market_snapshot()
    with (
        patch(
            "services.training_data.fetch_browser_data",
            return_value=(frame, closes, dates, metadata),
        ),
        patch(
            "services.training_data.future_trading_dates",
            return_value=(
                pd.date_range("2025-08-01", periods=30, freq="B").strftime("%Y-%m-%d").tolist(),
                "XNYS",
            ),
        ),
    ):
        result = build_training_snapshot("MSFT")

    original_dates = result["dates"]
    swapped = original_dates.copy()
    swapped[-1], swapped[0] = swapped[0], swapped[-1]
    with pytest.raises(ValueError, match="chronological"):
        validate_training_snapshot({**result, "dates": swapped})

    with pytest.raises(ValueError, match="chronological"):
        validate_training_snapshot({**result, "future_dates": [result["future_dates"][-1]] * 30})

    with pytest.raises(ValueError, match="do not follow"):
        validate_training_snapshot({**result, "future_dates": result["dates"][:30]})


def test_training_snapshot_annotates_quality_findings():
    frame, closes, dates, metadata = _market_snapshot()
    metadata["quality"] = {
        "checks": {"stale_latest_observation_days": 400},
        "issues": [
            {
                "code": "stale_latest_observation",
                "severity": "warning",
                "detail": "Latest observation is 400 days old.",
            }
        ],
        "status": "annotated",
    }
    with (
        patch(
            "services.training_data.fetch_browser_data",
            return_value=(frame, closes, dates, metadata),
        ),
        patch("services.training_data.future_trading_dates", return_value=([], "XNYS")),
    ):
        result = build_training_snapshot("MSFT")
    issues = result["data_snapshot"]["quality"]["issues"]
    assert any(issue["code"] == "stale_latest_observation" for issue in issues)
