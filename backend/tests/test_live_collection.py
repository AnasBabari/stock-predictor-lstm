from __future__ import annotations

import pytest

from services.live_collection import (
    LIVE_COLLECTION_ITEMS_V1,
    LIVE_EXPECTED_RECORD_COUNT,
    LIVE_HORIZONS_V1,
    LIVE_MODEL_POLICY_V1,
    LIVE_START_DATE,
    LIVE_UNIVERSE_V1,
    validate_live_collection_item,
)


def test_frozen_live_collection_contract_is_exact() -> None:
    assert LIVE_UNIVERSE_V1 == (
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
    assert LIVE_HORIZONS_V1 == (1, 5, 10, 20)
    assert LIVE_START_DATE.isoformat() == "2026-09-02"
    assert LIVE_EXPECTED_RECORD_COUNT == 60
    assert len(LIVE_COLLECTION_ITEMS_V1) == 60
    assert len(set(LIVE_COLLECTION_ITEMS_V1)) == 60
    assert dict(LIVE_MODEL_POLICY_V1) == {
        1: "garch_11",
        5: "rolling_mean",
        10: "rolling_mean",
        20: "rolling_mean",
    }


def test_live_collection_item_validation_fails_closed() -> None:
    assert validate_live_collection_item(" msft ", 5) == ("MSFT", 5)
    with pytest.raises(ValueError, match="frozen live universe"):
        validate_live_collection_item("TSLA", 5)
    with pytest.raises(ValueError, match="frozen live contract"):
        validate_live_collection_item("MSFT", 7)
