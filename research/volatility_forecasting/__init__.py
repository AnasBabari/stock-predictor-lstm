"""Leakage-safe global volatility forecasting research package.

This package is deliberately offline-only. Production imports must use the
small exported inference contract, never the training implementation.
"""

from .cache import ExampleCacheError, load_example_cache, save_example_cache
from .contracts import (
    DEFAULT_HORIZONS,
    VOLATILITY_PROTOCOL_VERSION,
    VolatilityForecastProtocol,
    VolatilityLossWeights,
)
from .data import (
    VolatilityPanelExamples,
    build_volatility_panel_examples,
    subset_volatility_panel_examples,
)
from .gdelt import GdeltEventRow, gdelt_row_to_news_event, parse_gdelt_v2_export_line
from .news import NEWS_FEATURE_NAMES_V2, NewsEvent, NewsOrigin, aggregate_news_features

_MODEL_SYMBOLS = {
    "BaselineResidualLSTM",
    "BaselineResidualLSTMConfig",
    "BaselineResidualTCN",
    "BaselineResidualTCNConfig",
    "RobustSequenceScaler",
    "TorchTrainingConfig",
    "train_baseline_residual_tcn",
}


def __getattr__(name: str):
    if name in _MODEL_SYMBOLS:
        from . import model

        return getattr(model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DEFAULT_HORIZONS",
    "VOLATILITY_PROTOCOL_VERSION",
    "VolatilityForecastProtocol",
    "VolatilityLossWeights",
    "ExampleCacheError",
    "load_example_cache",
    "save_example_cache",
    "VolatilityPanelExamples",
    "build_volatility_panel_examples",
    "subset_volatility_panel_examples",
    "BaselineResidualLSTM",
    "BaselineResidualLSTMConfig",
    "BaselineResidualTCN",
    "BaselineResidualTCNConfig",
    "RobustSequenceScaler",
    "TorchTrainingConfig",
    "train_baseline_residual_tcn",
    "GdeltEventRow",
    "gdelt_row_to_news_event",
    "parse_gdelt_v2_export_line",
    "NEWS_FEATURE_NAMES_V2",
    "NewsEvent",
    "NewsOrigin",
    "aggregate_news_features",
]
