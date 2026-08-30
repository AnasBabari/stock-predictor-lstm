"""Unit tests for MultimodalFusionModel."""

import numpy as np

from research.volatility_forecasting.multimodal_fusion_model_v2 import (
    MultimodalFusionModel,
)


def test_multimodal_fusion_forward_and_prediction():
    model = MultimodalFusionModel(numeric_dim=32, news_dim=18, horizons=(1, 3, 5, 7))
    model.eval()

    num_feats = np.random.randn(32)
    news_feats = np.random.randn(18)
    base_p = 432.42

    # Run with news
    res_news = model.predict_distribution(base_p, num_feats, news_feats, har_vol_daily=0.0168)
    assert len(res_news) == 4
    assert res_news[0].horizon == 1
    assert res_news[3].horizon == 7
    assert res_news[0].reconstructed_median_price > 0.0
    assert (
        res_news[0].prediction_interval_80pct[0]
        < res_news[0].reconstructed_median_price
        < res_news[0].prediction_interval_80pct[1]
    )

    # Run numeric-only (news=None)
    res_num_only = model.predict_distribution(base_p, num_feats, None, har_vol_daily=0.0168)
    assert len(res_num_only) == 4
