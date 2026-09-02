"""Shared production contract for deployable volatility forecasts.

The empirical benchmark and the active HTTP service use the same four
session horizons.  Keeping this contract in a small dependency-free module
prevents the API, ledger, and snapshot builder from drifting independently.
"""

from __future__ import annotations

from types import MappingProxyType

SUPPORTED_VOLATILITY_HORIZONS = (1, 5, 10, 20)
VOLATILITY_MAX_HORIZON = max(SUPPORTED_VOLATILITY_HORIZONS)
VOLATILITY_MODEL_POLICY_VERSION = "empirical_volatility_benchmark_v3"
VOLATILITY_MODEL_VERSION = "deployable_v5"
VOLATILITY_FEATURE_SET_VERSION = "deployable_feature_columns_v5"

# This is the frozen validation policy used by the active service.  It is a
# routing policy, not a claim that every asset has the same best model.
AUTO_MODEL_POLICY = MappingProxyType(
    {
        1: "garch_11",
        5: "rolling_mean",
        10: "rolling_mean",
        20: "rolling_mean",
    }
)


def is_supported_volatility_horizon(horizon: int) -> bool:
    """Return whether ``horizon`` is part of the production contract."""

    try:
        validate_volatility_horizon(horizon)
        return True
    except (TypeError, ValueError):
        return False


def validate_volatility_horizon(horizon: int) -> int:
    """Normalize and validate a production volatility horizon."""

    if isinstance(horizon, bool):
        raise ValueError(f"horizon must be one of {list(SUPPORTED_VOLATILITY_HORIZONS)}")
    try:
        normalized = int(horizon)
        if isinstance(horizon, float) and not horizon.is_integer():
            raise ValueError
    except (TypeError, ValueError) as err:
        raise ValueError(f"horizon must be one of {list(SUPPORTED_VOLATILITY_HORIZONS)}") from err
    if normalized not in SUPPORTED_VOLATILITY_HORIZONS:
        raise ValueError(f"horizon must be one of {list(SUPPORTED_VOLATILITY_HORIZONS)}")
    return normalized


__all__ = [
    "AUTO_MODEL_POLICY",
    "SUPPORTED_VOLATILITY_HORIZONS",
    "VOLATILITY_MAX_HORIZON",
    "VOLATILITY_FEATURE_SET_VERSION",
    "VOLATILITY_MODEL_VERSION",
    "VOLATILITY_MODEL_POLICY_VERSION",
    "is_supported_volatility_horizon",
    "validate_volatility_horizon",
]
