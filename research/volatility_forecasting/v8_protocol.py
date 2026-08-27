"""Frozen v8 protocol for historical temporal + cross-asset certification.

This module is the v8 analogue of ``prospective.py`` but for a sealed
historical 70/15/15 split. It does not modify the v7 future-prospective
cycle. All v8 certifiable artifacts derive from this single source of truth.

Protocol versions:
- ``global-volatility-distribution-v8-news-transfer`` (primary, news-enhanced)
- ``global-volatility-distribution-v8-numeric`` (numeric fallback if news absent)

Design: ONE canonical frozen object ``V8ProtocolSettings``.  ``v8_protocol()``
derives the ``VolatilityForecastProtocol`` from it — no duplication.
"""

from __future__ import annotations

import math
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
V8_NEWS_TAXONOMY_VERSION = "v1"
V8_FEATURE_SCHEMA_VERSION = "deployable_v5+news-v1"
V8_TARGET_VERSION = "future-rv-total-v1"
V8_NEWS_TARGET_VERSION = "future-rv-total-v1-news-v8"

# v8 does NOT use the v7 prospective future reserve.  Any generic code that
# checks ``temporal_holdout_sessions == 252`` must not misinterpret v8 as v7.
# Keep the underlying protocol field at 0 and expose the v7 legacy value
# under a separate name for audit comparison.
V8_TEMPORAL_HOLDOUT_SESSIONS = 0
V8_PROSPECTIVE_TMP_HOLDOUT_FOR_COMPARISON = 252


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
    """All frozen knobs for one v8 run. Canonical source for ``v8_protocol()``."""

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
    news_taxonomy_version: str = V8_NEWS_TAXONOMY_VERSION
    realized_variance_proxy: str = "overnight_plus_rogers_satchell"
    baseline_family: str = "causal_log_har"
    comparison_baseline_family: str = "adaptive_calibrated_har_c2c_v1"

    def __post_init__(self) -> None:
        if not math.isclose(
            self.split.train_fraction + self.split.validation_fraction + self.split.test_fraction,
            1.0,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
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


def _canonical_v8_settings(*, news_enabled: bool) -> V8ProtocolSettings:
    if news_enabled:
        return V8ProtocolSettings()
    return V8ProtocolSettings(
        protocol_version=V8_PROTOCOL_VERSION_NUMERIC,
        architecture_version=V8_ARCHITECTURE_NUMERIC,
        model_version=V8_MODEL_VERSION_NUMERIC,
        news_enabled=False,
        news_windows=(),
    )


def v8_protocol(*, news_enabled: bool = True) -> VolatilityForecastProtocol:
    """Derive the frozen ``VolatilityForecastProtocol`` from canonical settings (no duplication)."""
    s = _canonical_v8_settings(news_enabled=news_enabled)
    return VolatilityForecastProtocol(
        protocol_version=s.protocol_version,
        architecture_version=s.architecture_version,
        target_version=V8_TARGET_VERSION if not news_enabled else V8_NEWS_TARGET_VERSION,
        horizons=s.horizons,
        window_size=s.window_size,
        embargo_sessions=s.embargo_sessions,
        minimum_train_sessions=s.minimum_train_sessions,
        validation_sessions=s.validation_sessions,
        early_stopping_sessions=s.early_stopping_sessions,
        temporal_holdout_sessions=V8_TEMPORAL_HOLDOUT_SESSIONS,
        asset_holdout_fraction=0.20,
        seeds=s.seeds,
        realized_variance_proxy=s.realized_variance_proxy,
        baseline_family=s.baseline_family,
        comparison_baseline_family=s.comparison_baseline_family,
    )


def v8_settings(*, news_enabled: bool = True) -> V8ProtocolSettings:
    return _canonical_v8_settings(news_enabled=news_enabled)


def v8_manifest(*, news_enabled: bool = True) -> dict[str, object]:
    """Single manifest derived from canonical settings + derived protocol (drift-checked)."""
    s = v8_settings(news_enabled=news_enabled)
    p = v8_protocol(news_enabled=news_enabled)
    # Drift guard: protocol and settings must agree on all overlapping fields
    assert p.protocol_version == s.protocol_version
    assert p.architecture_version == s.architecture_version
    assert p.horizons == s.horizons
    assert p.window_size == s.window_size
    assert p.embargo_sessions == s.embargo_sessions
    assert p.seeds == s.seeds
    # v8_temporal_holdout must not equal v7's 252 — isolation assertion
    assert p.temporal_holdout_sessions == V8_TEMPORAL_HOLDOUT_SESSIONS
    assert p.temporal_holdout_sessions != V8_PROSPECTIVE_TMP_HOLDOUT_FOR_COMPARISON
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
        # Explicitly record that v8 does not use the v7 prospective holdout
        "prospective_temporal_holdout_sessions": V8_PROSPECTIVE_TMP_HOLDOUT_FOR_COMPARISON,
        "temporal_holdout_sessions": V8_TEMPORAL_HOLDOUT_SESSIONS,
        "v8_certification_never_invokes_v7_prospective_path": True,
    }
