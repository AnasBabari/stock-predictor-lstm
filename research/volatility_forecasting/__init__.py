"""Leakage-safe global volatility forecasting research package.

This package is deliberately offline-only. Production imports must use the
small exported inference contract, never the training implementation.
"""

from .cache import ExampleCacheError, load_example_cache, save_example_cache
from .contracts import (
    DEFAULT_HORIZONS,
    VOLATILITY_PROTOCOL_VERSION,
    VolatilityForecastProtocol,
)
from .data import VolatilityPanelExamples, build_volatility_panel_examples
from .gdelt import GdeltEventRow, gdelt_row_to_news_event, parse_gdelt_v2_export_line
from .news import NEWS_FEATURE_NAMES_V1, NewsEvent, NewsOrigin, aggregate_news_features

__all__ = [
    "DEFAULT_HORIZONS",
    "VOLATILITY_PROTOCOL_VERSION",
    "VolatilityForecastProtocol",
    "ExampleCacheError",
    "load_example_cache",
    "save_example_cache",
    "VolatilityPanelExamples",
    "build_volatility_panel_examples",
    "GdeltEventRow",
    "gdelt_row_to_news_event",
    "parse_gdelt_v2_export_line",
    "NEWS_FEATURE_NAMES_V1",
    "NewsEvent",
    "NewsOrigin",
    "aggregate_news_features",
]
