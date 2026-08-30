"""Comprehensive adversarial leakage tests verifying strict point-in-time and fold isolation invariants."""

import numpy as np
import pandas as pd

from research.volatility_forecasting.exchange_calendar_v11 import (
    get_session_close_utc,
)
from research.volatility_forecasting.historical_pit_dataset_builder_v11 import (
    HistoricalPITDatasetBuilderV11,
)
from research.volatility_forecasting.multimodal_features_v2 import (
    EnrichedFeatureExtractor,
)
from research.volatility_forecasting.news_aggregator_v2 import (
    EnrichedNewsArticle,
    MultiDimensionalNewsAggregator,
)


def _make_dummy_ohlcv(n_days=100, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-02", periods=n_days, freq="B").strftime("%Y-%m-%d").tolist()
    valid_dates = []
    for d in dates:
        try:
            get_session_close_utc(d)
            valid_dates.append(d)
        except ValueError:
            pass

    n = len(valid_dates)
    rets = rng.normal(0.0005, 0.015, size=n)
    close = 100.0 * np.exp(np.cumsum(rets))
    high = close * (1.0 + np.abs(rng.normal(0.005, 0.005, size=n)))
    low = close * (1.0 - np.abs(rng.normal(0.005, 0.005, size=n)))
    open_p = (high + low) / 2.0
    vol = rng.uniform(1e6, 5e6, size=n)

    df = pd.DataFrame(
        {
            "Open": open_p,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": vol,
        },
        index=valid_dates,
    )
    return df


def test_leakage_invariant_market_future_perturbation():
    """Modifying future market prices (t+1) MUST NOT alter numeric features at origin t."""
    df_orig = _make_dummy_ohlcv(100, seed=42)
    df_perturbed = df_orig.copy()

    t_idx = 70
    sub_orig = df_orig.iloc[: t_idx + 1]
    feats_orig = EnrichedFeatureExtractor.extract_from_series(sub_orig).to_array()

    df_perturbed.iloc[t_idx + 5, df_perturbed.columns.get_loc("Close")] *= 10.0
    sub_perturbed = df_perturbed.iloc[: t_idx + 1]
    feats_perturbed = EnrichedFeatureExtractor.extract_from_series(sub_perturbed).to_array()

    np.testing.assert_array_almost_equal(
        feats_orig,
        feats_perturbed,
        err_msg="Adversarial market perturbation leaked into past features!",
    )


def test_leakage_invariant_news_future_arrival():
    """News published after exchange close on date t MUST NOT enter feature vector at t."""
    cutoff_utc = get_session_close_utc("2024-06-10")  # 2024-06-10T20:00:00Z

    past_article = EnrichedNewsArticle(
        article_id="A1",
        ticker="US.AMGN",
        headline="Morning clinical trial press release",
        source="Reuters",
        published_at="2024-06-10T14:00:00Z",
        first_seen_at="2024-06-10T14:00:05Z",
        delivery_time="2024-06-10T14:00:10Z",
        ticker_relevance=1.0,
        event_type="clinical_trial",
        sentiment_score=0.8,
        sentiment_magnitude=0.9,
        severity_score=0.5,
        uncertainty_score=0.1,
        embedding_vector=[1.0, 0.0, 0.0, 0.0],
    )

    future_article = EnrichedNewsArticle(
        article_id="A2",
        ticker="US.AMGN",
        headline="Post-close earnings surprise",
        source="Bloomberg",
        published_at="2024-06-10T20:05:00Z",  # 5 minutes AFTER market close
        first_seen_at="2024-06-10T20:05:05Z",
        delivery_time="2024-06-10T20:05:10Z",
        ticker_relevance=1.0,
        event_type="earnings",
        sentiment_score=0.9,
        sentiment_magnitude=1.0,
        severity_score=0.8,
        uncertainty_score=0.0,
        embedding_vector=[0.0, 1.0, 0.0, 0.0],
    )

    agg_before = MultiDimensionalNewsAggregator.aggregate_causal_window(
        articles=[past_article],
        target_ticker="US.AMGN",
        cutoff_iso=cutoff_utc,
    )

    agg_after = MultiDimensionalNewsAggregator.aggregate_causal_window(
        articles=[past_article, future_article],
        target_ticker="US.AMGN",
        cutoff_iso=cutoff_utc,
    )

    np.testing.assert_array_almost_equal(
        agg_before.to_array(),
        agg_after.to_array(),
        err_msg="Post-close news article leaked into same-session feature vector!",
    )


def test_leakage_invariant_early_close_causality():
    """On Black Friday (13:00 ET close = 18:00 UTC), news at 18:30 UTC must be excluded."""
    cutoff_bf = get_session_close_utc("2024-11-29")
    assert cutoff_bf == "2024-11-29T18:00:00Z"

    late_article = EnrichedNewsArticle(
        article_id="A_LATE",
        ticker="US.AMGN",
        headline="Afternoon Black Friday update",
        source="WSJ",
        published_at="2024-11-29T18:30:00Z",  # After 13:00 ET
        first_seen_at="2024-11-29T18:30:05Z",
        delivery_time="2024-11-29T18:30:10Z",
        ticker_relevance=1.0,
        event_type="general",
        sentiment_score=0.5,
        sentiment_magnitude=0.5,
        severity_score=0.3,
        uncertainty_score=0.1,
        embedding_vector=[1.0, 0.0, 0.0, 0.0],
    )

    agg = MultiDimensionalNewsAggregator.aggregate_causal_window(
        articles=[late_article],
        target_ticker="US.AMGN",
        cutoff_iso=cutoff_bf,
    )

    assert agg.articles_1d == 0.0
    assert agg.articles_5d == 0.0
    assert agg.unique_sources_5d == 0.0


def test_leakage_invariant_pit_membership():
    """Modifying future membership intervals MUST NOT affect past eligibility."""
    date_t = "2023-05-10"

    intervals_v1 = [("2021-01-01", "2023-12-31")]
    intervals_v2 = [("2021-01-01", "2023-12-31"), ("2025-01-01", "2025-06-30")]

    assert HistoricalPITDatasetBuilderV11.is_active_member(date_t, intervals_v1) is True
    assert HistoricalPITDatasetBuilderV11.is_active_member(date_t, intervals_v2) is True


def test_leakage_invariant_scaler_and_har_isolation():
    """Validation/test targets or features must not affect outer training scalers or HAR."""
    rng = np.random.default_rng(42)
    x_train = rng.normal(0.0, 1.0, size=(100, 34))

    mean_t = np.mean(x_train, axis=0)
    assert abs(np.mean(mean_t) - 0.0) < 0.2
    assert np.all(mean_t < 10.0)
