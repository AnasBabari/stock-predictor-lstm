"""v8 feature/target schema — deployable_v5 + news-v1 (numeric fallback certified today).

This module is the frozen record of what the v8 serving runtime will expect.
No feature engineering code belongs here beyond schema declaration; the
actual builders remain in ``backend.panel.features`` and
``research/volatility_forecasting/news*.py``.
"""

from __future__ import annotations

from backend.panel.features import DEPLOYABLE_FEATURE_COLUMNS_V5

# News-v1 adds causal aggregated features on top of deployable_v5.
# For the numeric fallback (certified today) these are absent and the
# runtime must treat missing news as a zero-vector with a missing_news_indicator.
NEWS_V1_FEATURES: tuple[str, ...] = (
    "News_Article_Count_1d",
    "News_Unique_Source_Count_1d",
    "News_Sentiment_Positive_1d",
    "News_Sentiment_Negative_1d",
    "News_Sentiment_Neutral_1d",
    "News_Sentiment_Dispersion_1d",
    "News_Negative_Tail_1d",
    "News_Positive_Tail_1d",
    "News_Novelty_1d",
    "News_Event_Intensity_1d",
    "News_Topic_Distribution_Entropy_1d",
    "News_Source_Disagreement_1d",
    "Missing_News_Indicator_1d",
    # Longer windows (3d,5d,20d) are derived identically; listed for schema versioning
    "News_Article_Count_5d",
    "News_Sentiment_Negative_5d",
    "News_Novelty_5d",
    "Market_VIX_Close",
    "Market_WTI_Return_1d",
    "Market_Brent_Return_1d",
    "Market_Gold_Return_1d",
    "Market_USD_Basket_Return_1d",
    "Market_TNX_Change_1d",
)

V8_NUMERIC_FEATURE_COUNT = len(DEPLOYABLE_FEATURE_COLUMNS_V5)  # 26
V8_NEWS_FEATURE_COUNT = len(DEPLOYABLE_FEATURE_COLUMNS_V5) + len(NEWS_V1_FEATURES)

V8_FEATURE_SCHEMA_VERSION = "deployable_v5+news-v1"
V8_TARGET_VERSION = "future-rv-total-v1"
V8_NEWS_TARGET_VERSION = "future-rv-total-v1-news-v8"

# Volatily target definition (mirrors research/volatility_forecasting/data.py)
# RV(i,t,h) = sqrt(sum(r(t+k)^2 for k=1..h)), target = log(RV + epsilon)
V8_HORIZONS: tuple[int, ...] = (1, 3, 5, 7, 14, 30)
V8_REQUIRED_HORIZONS: tuple[int, ...] = (1, 3, 5, 7)
V8_WINDOW_SIZE = 60
V8_EPSILON = 1e-8

# Validation: feature names must be unique and ordered
assert len(set(DEPLOYABLE_FEATURE_COLUMNS_V5)) == len(DEPLOYABLE_FEATURE_COLUMNS_V5)
assert len(set(NEWS_V1_FEATURES)) == len(NEWS_V1_FEATURES)
assert not set(DEPLOYABLE_FEATURE_COLUMNS_V5).intersection(NEWS_V1_FEATURES)
