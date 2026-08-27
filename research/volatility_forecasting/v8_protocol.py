"""Frozen v8 protocol for historical temporal + cross-asset certification.

This module is the v8 analogue of ``prospective.py`` but for a sealed
historical 70/15/15 split. It does not modify the v7 future-prospective
cycle. All v8 certifiable artifacts derive from this single source of truth.

Protocol versions:
- ``global-volatility-distribution-v8-news-transfer`` (primary, news-enhanced)
- ``global-volatility-distribution-v8-numeric`` (numeric fallback if news absent)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .contracts import VolatilityForecastProtocol

V8_PROTOCOL_VERSION_NEWS = "global-volatility-distribution-v8-news-transfer"
V8_PROTOCOL_VERSION_NUMERIC = "global-volatility-distribution-v8-numeric"
V8_ARCHITECTURE_NEWS = "volatility-fusion-tcn-v1"
V8_ARCHITECTURE_NUMERIC = "baseline-residual-tcn-v3"

V8_MODEL_VERSION_NEWS = "global-volatility-news-fusion-v8"
V8_MODEL_VERSION_NUMERIC = "global-volatility-v8-numeric"

V8_SPLIT_VERSION = "v8-chronological-70-15-15-purged"
V8_NEWS_SCHEMA_VERSION = "news-v1"
V8_FEATURE_SCHEMA_VERSION = "deployable_v5+news-v1"
V8_TARGET_VERSION = "future-rv-total-v1"
V8_NEWS_TARGET_VERSION = "future-rv-total-v1-news-v8"


@dataclass(frozen=True)
class V8SplitManifest:
    split_version: str = V8_SPLIT_VERSION
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    chronological: bool = True
    purge: bool = True
    embargo_sessions: int = 30
    purge_horizon_sessions: int = 30  # >= max_horizon


@dataclass(frozen=True)
class V8ProtocolSettings:
    """All frozen knobs for one v8 run. Create via ``v8_protocol()``."""

    protocol_version: str = V8_PROTOCOL_VERSION_NEWS
    architecture_version: str = V8_ARCHITECTURE_NEWS
    model_version: str = V8_MODEL_VERSION_NEWS
    news_enabled: bool = True
    asset_transfer_enabled: bool = True
    certification_scope: str = "historical_temporal_test_plus_asset_transfer"
    metric_source: str = "locked_historical_temporal_test_plus_asset_transfer"
    split: V8SplitManifest = V8SplitManifest()
    horizons: tuple[int, ...] = (1, 3, 5, 7, 14, 30)
    required_horizons: tuple[int, ...] = (1, 3, 5, 7)
    forecast_target: str = "forward_realized_volatility"
    window_size: int = 60
    embargo_sessions: int = 30
    minimum_train_sessions: int = 756
    validation_sessions: int = 126  # not used for 70/15/15 but kept for auditing
    early_stopping_sessions: int = 63
    seeds: tuple[int, ...] = (41, 42, 43)
    news_windows: tuple[str, ...] = ("1h", "4h", "1d", "3d", "5d", "20d")
    news_taxonomy_version: str = "v1"
    realized_variance_proxy: str = "overnight_plus_rogers_satchell"
    baseline_family: str = "causal_log_har"
    comparison_baseline_family: str = "adaptive_calibrated_har_c2c_v1"

    def __post_init__(self) -> None:
        if self.train_fraction + self.validation_fraction + self.test_fraction != 1.0:
            raise ValueError("split fractions must sum to 1.0")
        if not self.horizons or tuple(sorted(set(self.horizons))) != self.horizons:
            raise ValueError("horizons must be increasing unique")
        if self.embargo_sessions < max(self.horizons):
            raise ValueError("embargo must be >= max horizon")

    @property
    def train_fraction(self) -> float:  # type: ignore[misc]
        return float(self.split.train_fraction)

    @property
    def validation_fraction(self) -> float:
        return float(self.split.validation_fraction)

    @property
    def test_fraction(self) -> float:
        return float(self.split.test_fraction)


V8_NEWS_TAXONOMY: tuple[str, ...] = (
    "earnings",
    "guidance",
    "merger_acquisition",
    "capital_raise",
    "dividend",
    "buyback",
    "product",
    "regulation",
    "litigation",
    "credit",
    "bankruptcy",
    "management_change",
    "labor",
    "supply_chain",
    "cybersecurity",
    "geopolitical",
    "macro",
    "central_bank",
    "commodity",
    "energy",
    "currency",
    "interest_rates",
    "analyst_action",
    "insider_activity",
    "other",
)


def v8_protocol(*, news_enabled: bool = True) -> VolatilityForecastProtocol:
    """Return the frozen v8 VolatilityForecastProtocol (window/horizons/embeddings)."""
    version = V8_PROTOCOL_VERSION_NEWS if news_enabled else V8_PROTOCOL_VERSION_NUMERIC
    architecture = V8_ARCHITECTURE_NEWS if news_enabled else V8_ARCHITECTURE_NUMERIC
    return VolatilityForecastProtocol(
        protocol_version=version,
        architecture_version=architecture,
        target_version=V8_TARGET_VERSION if not news_enabled else V8_NEWS_TARGET_VERSION,
        horizons=(1, 3, 5, 7, 14, 30),
        window_size=60,
        embargo_sessions=30,
        minimum_train_sessions=756,
        validation_sessions=126,
        early_stopping_sessions=63,
        temporal_holdout_sessions=252,  # retained for monitoring comparability
        asset_holdout_fraction=0.20,
        seeds=(41, 42, 43),
    )


def v8_settings(*, news_enabled: bool = True) -> V8ProtocolSettings:
    if news_enabled:
        return V8ProtocolSettings()
    return V8ProtocolSettings(
        protocol_version=V8_PROTOCOL_VERSION_NUMERIC,
        architecture_version=V8_ARCHITECTURE_NUMERIC,
        model_version=V8_MODEL_VERSION_NUMERIC,
        news_enabled=False,
        news_windows=(),
    )


def v8_manifest(*, news_enabled: bool = True) -> dict[str, object]:
    s = v8_settings(news_enabled=news_enabled)
    p = v8_protocol(news_enabled=news_enabled)
    return {
        "protocol_version": s.protocol_version,
        "model_version": s.model_version,
        "architecture_version": s.architecture_version,
        "target_version": p.target_version,
        "feature_schema_version": V8_FEATURE_SCHEMA_VERSION if s.news_enabled else "deployable_v5",
        "news_schema_version": V8_NEWS_SCHEMA_VERSION if s.news_enabled else None,
        "forecast_target": s.forecast_target,
        "horizons": list(s.horizons),
        "required_horizons": list(s.required_horizons),
        "window_size": s.window_size,
        "seeds": list(s.seeds),
        "split": asdict(s.split),
        "news_enabled": s.news_enabled,
        "asset_transfer_enabled": s.asset_transfer_enabled,
        "certification_scope": s.certification_scope,
        "metric_source": s.metric_source,
        "news_windows": list(s.news_windows) if s.news_windows else [],
        "news_taxonomy": list(V8_NEWS_TAXONOMY),
        "news_taxonomy_version": s.news_taxonomy_version,
    }
