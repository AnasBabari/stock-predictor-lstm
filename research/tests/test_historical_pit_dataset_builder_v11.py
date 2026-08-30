"""Unit tests for hardened HistoricalPITDatasetBuilderV11 with non-empty news & perturbation check."""

import numpy as np
import pandas as pd

from research.volatility_forecasting.historical_pit_dataset_builder_v11 import (
    HistoricalPITDatasetBuilderV11,
)
from research.volatility_forecasting.news_aggregator_v2 import (
    EnrichedNewsArticle,
)


def test_historical_pit_panel_construction_and_perturbation_invariance():
    n_days = 100
    dates = pd.date_range("2022-01-01", periods=n_days, freq="B")
    rng = np.random.default_rng(42)

    def create_synthetic_df():
        c = np.exp(np.cumsum(rng.normal(0.0005, 0.015, size=n_days))) * 100.0
        h = c * 1.01
        low_p = c * 0.99
        o = c
        v = rng.uniform(1e6, 5e6, size=n_days)
        return pd.DataFrame(
            {"Open": o, "High": h, "Low": low_p, "Close": c, "Volume": v}, index=dates
        )

    equities = {
        "AAPL": create_synthetic_df(),
        "MSFT": create_synthetic_df(),
        "AMGN": create_synthetic_df(),
    }
    sector_df = create_synthetic_df()
    market_df = create_synthetic_df()

    # Non-empty news articles with ticker binding
    news_articles = [
        EnrichedNewsArticle(
            article_id="NEWS_1",
            ticker="AMGN",
            headline="Amgen clinical progress reported",
            source="Reuters",
            published_at="2022-03-01T14:00:00Z",
            first_seen_at="2022-03-01T14:00:05Z",
            delivery_time="2022-03-01T14:00:10Z",
            ticker_relevance=1.0,
            event_type="clinical_trial",
            sentiment_score=0.75,
            sentiment_magnitude=0.8,
            severity_score=0.3,
            uncertainty_score=0.2,
            embedding_vector=[1.0, 0.0, 0.0, 0.0],
        ),
        EnrichedNewsArticle(
            article_id="NEWS_2",
            ticker="MARKET",
            headline="FOMC Rate Decision Announcement",
            source="Bloomberg",
            published_at="2022-03-02T18:00:00Z",
            first_seen_at="2022-03-02T18:00:05Z",
            delivery_time="2022-03-02T18:00:10Z",
            ticker_relevance=0.9,
            event_type="macro",
            sentiment_score=0.1,
            sentiment_magnitude=0.4,
            severity_score=0.4,
            uncertainty_score=0.5,
            embedding_vector=[0.0, 1.0, 0.0, 0.0],
        ),
    ]

    # Define multi-interval membership mask excluding AMGN after day 85, and excluding UNKNOWN entirely (fail closed)
    membership = {
        "AAPL": [("2022-01-01", "2022-12-31")],
        "MSFT": [("2022-01-01", "2022-12-31")],
        "AMGN": [("2022-01-01", str(dates[85])[:10])],
    }

    panel1 = HistoricalPITDatasetBuilderV11.construct_panel_from_series(
        equities_ohlcv=equities,
        sector_ohlcv=sector_df,
        market_ohlcv=market_df,
        news_articles=news_articles,
        membership_masks=membership,
        horizons=(1, 3, 5, 7),
        warmup_sessions=65,
    )

    assert panel1.numeric_features.shape[1] == 34
    assert panel1.news_features.shape[1] == 19
    assert panel1.same_origin_shuffled_news.shape == panel1.news_features.shape
    assert panel1.causal_delayed_news.shape == panel1.news_features.shape
    assert panel1.returns_targets.shape[1] == 4
    assert panel1.rv_targets.shape[1] == 4
    assert len(panel1.panel_sha256) == 64

    # Perturbation Test: Altering future prices at day 95 produces ZERO change in features at day 70
    equities_perturbed = {k: v.copy() for k, v in equities.items()}
    equities_perturbed["AAPL"].iloc[95, equities_perturbed["AAPL"].columns.get_loc("Close")] *= 1.50

    panel2 = HistoricalPITDatasetBuilderV11.construct_panel_from_series(
        equities_ohlcv=equities_perturbed,
        sector_ohlcv=sector_df,
        market_ohlcv=market_df,
        news_articles=news_articles,
        membership_masks=membership,
        horizons=(1, 3, 5, 7),
        warmup_sessions=65,
    )

    target_date = str(dates[70])[:10]
    idx1 = [i for i, d in enumerate(panel1.dates) if d == target_date]
    idx2 = [i for i, d in enumerate(panel2.dates) if d == target_date]

    np.testing.assert_array_equal(panel1.numeric_features[idx1], panel2.numeric_features[idx2])
    np.testing.assert_array_equal(panel1.news_features[idx1], panel2.news_features[idx2])
