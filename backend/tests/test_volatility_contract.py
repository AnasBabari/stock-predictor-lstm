from __future__ import annotations

import pytest

from services.volatility_contract import (
    AUTO_MODEL_POLICY,
    SUPPORTED_VOLATILITY_HORIZONS,
    VOLATILITY_FEATURE_SET_VERSION,
    VOLATILITY_MODEL_VERSION,
    validate_volatility_horizon,
)


def test_production_horizon_contract_and_auto_policy_are_frozen() -> None:
    assert SUPPORTED_VOLATILITY_HORIZONS == (1, 5, 10, 20)
    assert dict(AUTO_MODEL_POLICY) == {
        1: "garch_11",
        5: "rolling_mean",
        10: "rolling_mean",
        20: "rolling_mean",
    }
    assert VOLATILITY_MODEL_VERSION == "deployable_v5"
    assert VOLATILITY_FEATURE_SET_VERSION == "deployable_feature_columns_v5"


@pytest.mark.parametrize("horizon", [1, 5, 10, 20])
def test_supported_horizons_normalize_to_int(horizon: int) -> None:
    assert validate_volatility_horizon(str(horizon)) == horizon


@pytest.mark.parametrize("horizon", [True, 0, 2, 3, 5.5, 7, 14, 30, "not-a-number"])
def test_unsupported_horizons_fail_closed(horizon: object) -> None:
    with pytest.raises(ValueError, match="one of"):
        validate_volatility_horizon(horizon)
