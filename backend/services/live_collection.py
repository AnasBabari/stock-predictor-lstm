"""Frozen operational contract for genuine forward-volatility collection."""

from __future__ import annotations

from datetime import date
from itertools import product
from types import MappingProxyType

from services.volatility_contract import (
    AUTO_MODEL_POLICY,
    SUPPORTED_VOLATILITY_HORIZONS,
)

LIVE_UNIVERSE_V1 = (
    "MSFT",
    "AAPL",
    "NVDA",
    "GOOGL",
    "AMZN",
    "META",
    "JPM",
    "XOM",
    "JNJ",
    "WMT",
    "CAT",
    "NEE",
    "PLD",
    "KO",
    "NMM",
)
LIVE_HORIZONS_V1 = SUPPORTED_VOLATILITY_HORIZONS
LIVE_UNIVERSE_VERSION = "live_universe_v1"
LIVE_START_DATE = date.fromisoformat("2026-09-02")

LIVE_MODEL_POLICY_V1 = MappingProxyType(dict(AUTO_MODEL_POLICY))
LIVE_COLLECTION_ITEMS_V1 = tuple(product(LIVE_UNIVERSE_V1, LIVE_HORIZONS_V1))
LIVE_EXPECTED_RECORD_COUNT = len(LIVE_COLLECTION_ITEMS_V1)


def validate_live_collection_item(ticker: str, horizon: int) -> tuple[str, int]:
    """Return a normalized frozen-universe item or fail closed."""
    symbol = str(ticker).strip().upper()
    normalized_horizon = int(horizon)
    if symbol not in LIVE_UNIVERSE_V1:
        raise ValueError("Ticker is not part of the frozen live universe.")
    if normalized_horizon not in LIVE_HORIZONS_V1:
        raise ValueError("Horizon is not part of the frozen live contract.")
    return symbol, normalized_horizon


__all__ = [
    "LIVE_COLLECTION_ITEMS_V1",
    "LIVE_EXPECTED_RECORD_COUNT",
    "LIVE_HORIZONS_V1",
    "LIVE_MODEL_POLICY_V1",
    "LIVE_START_DATE",
    "LIVE_UNIVERSE_V1",
    "LIVE_UNIVERSE_VERSION",
    "validate_live_collection_item",
]
