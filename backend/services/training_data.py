"""Validated feature snapshots for browser-side model training.

The public backend is deliberately a data service. It builds the same feature
matrix used by the offline Python trainer, but never imports TensorFlow, loads
model artifacts, or writes training output to disk.
"""

from __future__ import annotations

import hashlib
import math
from datetime import UTC, datetime
from typing import Any

import numpy as np

from calendars import future_trading_dates
from config import FEATURES, MAX_FORECAST_DAYS, SCHEMA_VERSION, WINDOW_SIZE
from data_pipeline import fetch_data

MAX_SNAPSHOT_ROWS = 2000


def _snapshot_id(feature_values: np.ndarray, dates: list[str], ticker: str) -> str:
    """Return a stable fingerprint for the exact feature snapshot sent to a client."""

    hasher = hashlib.sha256()
    hasher.update(ticker.encode("utf-8"))
    hasher.update(str(SCHEMA_VERSION).encode("ascii"))
    hasher.update("|".join(FEATURES).encode("utf-8"))
    hasher.update("|".join(dates).encode("utf-8"))
    hasher.update(np.asarray(feature_values, dtype=np.float64).tobytes())
    return hasher.hexdigest()


def _finite_rows(values: np.ndarray) -> list[list[float]]:
    """Convert a matrix to JSON-safe finite floats and reject invalid data."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURES):
        raise ValueError("Feature matrix has an incompatible shape.")
    if not np.isfinite(matrix).all():
        raise ValueError("Feature matrix contains non-finite values.")
    return matrix.tolist()


def build_training_snapshot(ticker: str, days: int = MAX_FORECAST_DAYS) -> dict[str, Any]:
    """Fetch and serialize the bounded, validated snapshot used by browser training."""

    days = int(days)
    if not 1 <= days <= MAX_FORECAST_DAYS:
        raise ValueError("Forecast horizon is outside the supported range.")

    feature_df, closing_prices, dates, metadata = fetch_data(ticker)
    feature_values = feature_df[FEATURES].to_numpy(dtype=np.float64)
    if len(feature_values) > MAX_SNAPSHOT_ROWS:
        start = len(feature_values) - MAX_SNAPSHOT_ROWS
        feature_values = feature_values[start:]
        closing_prices = np.asarray(closing_prices)[start:]
        dates = dates[start:]
    date_values = dates.strftime("%Y-%m-%d").tolist()
    if len(feature_values) < WINDOW_SIZE + MAX_FORECAST_DAYS + 1:
        raise ValueError("Not enough feature rows for browser training.")

    future_dates, calendar_id = future_trading_dates(ticker, dates[-1], days)
    rows = _finite_rows(feature_values)
    close_values = np.asarray(closing_prices, dtype=np.float64).reshape(-1)
    if len(close_values) != len(rows) or not np.isfinite(close_values).all():
        raise ValueError("Close-price history is incompatible with the feature snapshot.")
    if np.any(close_values <= 0):
        raise ValueError("Close-price history contains invalid values.")

    snapshot_metadata = dict(metadata)
    snapshot_metadata.update(
        {
            "schema_version": SCHEMA_VERSION,
            "feature_schema_version": SCHEMA_VERSION,
            "feature_names": list(FEATURES),
            "snapshot_id": _snapshot_id(feature_values, date_values, ticker),
            "ticker": ticker,
        }
    )
    result = {
        "ticker": ticker,
        "schema_version": SCHEMA_VERSION,
        "snapshot_id": snapshot_metadata["snapshot_id"],
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "feature_names": list(FEATURES),
        "window_size": WINDOW_SIZE,
        "output_width": MAX_FORECAST_DAYS,
        "close_index": FEATURES.index("Close"),
        "dates": date_values,
        "features": rows,
        "historical_prices": close_values.tolist(),
        "future_dates": future_dates,
        "calendar": calendar_id,
        "data_snapshot": snapshot_metadata,
        "market_context": metadata.get("market_context", {}),
    }
    validate_training_snapshot(result)
    return result


def validate_training_snapshot(snapshot: dict[str, Any]) -> None:
    """Defensive validation for tests and future callers before serialization."""

    required = {"ticker", "snapshot_id", "feature_names", "dates", "features", "historical_prices"}
    if not required.issubset(snapshot):
        raise ValueError("Training snapshot is missing required fields.")
    if snapshot["feature_names"] != list(FEATURES):
        raise ValueError("Training snapshot feature order is incompatible.")
    features = snapshot["features"]
    if len(features) != len(snapshot["dates"]):
        raise ValueError("Training snapshot dates and rows must have equal length.")
    for row in features:
        if len(row) != len(FEATURES) or not all(math.isfinite(float(value)) for value in row):
            raise ValueError("Training snapshot contains invalid feature values.")
    prices = snapshot["historical_prices"]
    if len(prices) != len(features) or not all(
        math.isfinite(float(value)) and float(value) > 0 for value in prices
    ):
        raise ValueError("Training snapshot contains invalid close prices.")
