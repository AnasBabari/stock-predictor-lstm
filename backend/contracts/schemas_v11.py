"""Formal V11 Multimodal Feature and Target Schema Contract (53 Features, Required 1/3/5/7 Horizons)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Final

# 1. Exact 34 Numeric Feature Definitions (Stationary, Non-Leaking)
MULTIMODAL_NUMERIC_FEATURE_COLUMNS_V11: Final[tuple[str, ...]] = (
    # Multi-horizon returns (7)
    "return_1d",
    "return_2d",
    "return_3d",
    "return_5d",
    "return_10d",
    "return_20d",
    "return_60d",
    # Trend indicators (4)
    "ema5_to_ema20_ratio",
    "slope_ema5",
    "slope_ema20",
    "price_to_sma20_ratio",
    # Drawdowns (2)
    "drawdown_20d",
    "drawdown_60d",
    # Momentum & acceleration (5)
    "momentum_5d",
    "momentum_10d",
    "momentum_20d",
    "momentum_acceleration_5_20",
    "short_minus_long_momentum",
    # Volatility measures (8)
    "vol_realized_5d",
    "vol_realized_20d",
    "vol_realized_60d",
    "vol_parkinson_20d",
    "vol_garman_klass_20d",
    "har_daily_vol",
    "har_weekly_vol",
    "har_monthly_vol",
    # Volume dynamics (4)
    "volume_z_score_20d",
    "volume_5d_to_20d_ratio",
    "price_volume_interaction_5d",
    "abnormal_volume_flag",
    # Relative sector and market context (4)
    "rel_return_5d_vs_sector",
    "rel_return_20d_vs_sector",
    "rel_return_5d_vs_market",
    "rel_return_20d_vs_market",
)

# 2. Exact 19 Causal News Feature Definitions
MULTIMODAL_NEWS_FEATURE_COLUMNS_V11: Final[tuple[str, ...]] = (
    # Volume & velocity (7)
    "total_articles_20d",
    "articles_1h",
    "articles_4h",
    "articles_1d",
    "articles_5d",
    "velocity_ratio_1d",
    "acceleration_1h",
    # Source diversity & entropy (2)
    "unique_sources_5d",
    "source_entropy_5d",
    # Sentiment & dispersion (3)
    "mean_sentiment_5d",
    "sentiment_magnitude_5d",
    "sentiment_disagreement_5d",
    # Severity & novelty (3)
    "mean_severity_5d",
    "mean_uncertainty_5d",
    "max_novelty_score_5d",
    # Event taxonomy intensity (4)
    "clinical_trial_events_5d",
    "fda_regulatory_events_5d",
    "earnings_guidance_events_5d",
    "analyst_action_events_5d",
)

MULTIMODAL_TOTAL_FEATURE_COLUMNS_V11: Final[tuple[str, ...]] = (
    MULTIMODAL_NUMERIC_FEATURE_COLUMNS_V11 + MULTIMODAL_NEWS_FEATURE_COLUMNS_V11
)

# Target Horizon Contracts
REQUIRED_TARGET_HORIZONS_V11: Final[tuple[int, ...]] = (1, 3, 5, 7)
AUXILIARY_TARGET_HORIZONS_V11: Final[tuple[int, ...]] = (2, 4, 6)
ALL_TARGET_HORIZONS_V11: Final[tuple[int, ...]] = (1, 2, 3, 4, 5, 6, 7)

# Horizon Loss Weights (Controlled scale weighting to prevent long horizons from dominating)
HORIZON_RETURN_LOSS_WEIGHTS_V11: Final[dict[int, float]] = {
    1: 1.00,
    3: 0.85,
    5: 0.70,
    7: 0.60,
}

HORIZON_VARIANCE_LOSS_WEIGHTS_V11: Final[dict[int, float]] = {
    1: 1.00,
    3: 0.85,
    5: 0.70,
    7: 0.60,
}


@dataclass(frozen=True)
class SchemaV11Manifest:
    numeric_count: int
    news_count: int
    total_count: int
    required_horizons: tuple[int, ...]
    schema_sha256: str


def get_schema_v11_manifest() -> SchemaV11Manifest:
    raw_str = "|".join(MULTIMODAL_TOTAL_FEATURE_COLUMNS_V11) + f"|H:{REQUIRED_TARGET_HORIZONS_V11}"
    digest = hashlib.sha256(raw_str.encode()).hexdigest()
    return SchemaV11Manifest(
        numeric_count=len(MULTIMODAL_NUMERIC_FEATURE_COLUMNS_V11),
        news_count=len(MULTIMODAL_NEWS_FEATURE_COLUMNS_V11),
        total_count=len(MULTIMODAL_TOTAL_FEATURE_COLUMNS_V11),
        required_horizons=REQUIRED_TARGET_HORIZONS_V11,
        schema_sha256=digest,
    )
