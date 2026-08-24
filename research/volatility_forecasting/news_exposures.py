"""Frozen, target-independent transmission hypotheses for market news shocks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .news import NewsValidationError

EXPOSURE_SCHEMA_VERSION = "news-exposure-map-v1"
ALLOWED_EXPOSURE_TOPICS = frozenset(
    {
        "commodity_disruption",
        "energy_policy",
        "fiscal_policy",
        "inflation",
        "military_conflict",
        "monetary_policy",
        "oil_supply",
        "regulation",
        "sanctions",
        "shipping_disruption",
        "terrorism",
    }
)


@dataclass(frozen=True)
class NewsExposureMap:
    schema_version: str
    methodology: str
    source_sha256: str
    exposures: dict[str, dict[str, float]]


def _weights(value: object, *, context: str) -> dict[str, float]:
    if not isinstance(value, dict):
        raise NewsValidationError(f"{context} must be a topic-to-weight object")
    normalized: dict[str, float] = {}
    for topic, weight in value.items():
        if not isinstance(topic, str) or topic.strip().lower() not in ALLOWED_EXPOSURE_TOPICS:
            raise NewsValidationError(f"{context} contains an unsupported exposure topic")
        if isinstance(weight, bool):
            raise NewsValidationError(f"{context} exposure weights must be numeric")
        try:
            numeric = float(weight)
        except (TypeError, ValueError) as error:
            raise NewsValidationError(f"{context} exposure weights must be numeric") from error
        if not np.isfinite(numeric) or not 0 <= numeric <= 1:
            raise NewsValidationError(f"{context} exposure weights must be finite and in [0, 1]")
        normalized[topic.strip().lower()] = numeric
    return dict(sorted(normalized.items()))


def load_news_exposure_map(
    path: Path,
    *,
    required_tickers: set[str] | None = None,
) -> NewsExposureMap:
    """Load profiles and overrides with exact ticker-coverage validation."""
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NewsValidationError("news exposure map is missing or invalid JSON") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != EXPOSURE_SCHEMA_VERSION:
        raise NewsValidationError("news exposure map schema version does not match")
    methodology = payload.get("methodology")
    if not isinstance(methodology, str) or not methodology.strip():
        raise NewsValidationError("news exposure map methodology is required")
    raw_profiles = payload.get("profiles")
    raw_assignments = payload.get("ticker_profiles")
    raw_overrides = payload.get("overrides", {})
    if not isinstance(raw_profiles, dict) or not raw_profiles:
        raise NewsValidationError("news exposure profiles must be a non-empty object")
    if not isinstance(raw_assignments, dict) or not raw_assignments:
        raise NewsValidationError("news exposure ticker profiles must be a non-empty object")
    if not isinstance(raw_overrides, dict):
        raise NewsValidationError("news exposure overrides must be an object")

    profiles = {
        str(name): _weights(weights, context=f"profile {name}")
        for name, weights in raw_profiles.items()
        if isinstance(name, str) and name.strip()
    }
    if len(profiles) != len(raw_profiles):
        raise NewsValidationError("news exposure profile names must be non-empty strings")

    exposures: dict[str, dict[str, float]] = {}
    for raw_ticker, raw_profile in raw_assignments.items():
        if not isinstance(raw_ticker, str) or not raw_ticker.strip():
            raise NewsValidationError("news exposure ticker names must be non-empty strings")
        if not isinstance(raw_profile, str) or raw_profile not in profiles:
            raise NewsValidationError("news exposure ticker references an unknown profile")
        ticker = raw_ticker.strip().upper()
        if ticker in exposures:
            raise NewsValidationError("news exposure map contains duplicate normalized tickers")
        exposures[ticker] = dict(profiles[raw_profile])

    for raw_ticker, raw_weights in raw_overrides.items():
        ticker = str(raw_ticker).strip().upper()
        if ticker not in exposures:
            raise NewsValidationError("news exposure override references an unassigned ticker")
        exposures[ticker].update(_weights(raw_weights, context=f"override {ticker}"))
        exposures[ticker] = dict(sorted(exposures[ticker].items()))

    if required_tickers is not None:
        expected = {str(ticker).strip().upper() for ticker in required_tickers}
        missing = sorted(expected - set(exposures))
        extra = sorted(set(exposures) - expected)
        if missing or extra:
            raise NewsValidationError(
                f"news exposure coverage mismatch: missing={missing}, extra={extra}"
            )
    return NewsExposureMap(
        schema_version=EXPOSURE_SCHEMA_VERSION,
        methodology=methodology.strip(),
        source_sha256=hashlib.sha256(raw).hexdigest(),
        exposures=dict(sorted(exposures.items())),
    )
