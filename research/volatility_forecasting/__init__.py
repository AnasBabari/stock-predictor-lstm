"""Leakage-safe global volatility forecasting research package.

This package is deliberately offline-only. Production imports must use the
small exported inference contract, never the training implementation.
"""

from .contracts import (
    DEFAULT_HORIZONS,
    VOLATILITY_PROTOCOL_VERSION,
    VolatilityForecastProtocol,
)
from .data import VolatilityPanelExamples, build_volatility_panel_examples

__all__ = [
    "DEFAULT_HORIZONS",
    "VOLATILITY_PROTOCOL_VERSION",
    "VolatilityForecastProtocol",
    "VolatilityPanelExamples",
    "build_volatility_panel_examples",
]
