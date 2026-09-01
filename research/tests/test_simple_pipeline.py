from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from volatility_forecasting.simple_pipeline import (
    LSTMConfig,
    baseline_predictions,
    build_examples,
    chronological_split,
    evaluate_benchmark,
    lstm_predictions,
    realised_volatility,
    select_validation_model,
    validate_ohlcv,
    volatility_metrics,
)


def _frame(rows: int = 260) -> pd.DataFrame:
    rng = np.random.default_rng(12)
    returns = rng.normal(0.0004, 0.012, size=rows)
    close = 100.0 * np.exp(np.cumsum(returns))
    open_price = close * np.exp(rng.normal(0.0, 0.002, size=rows))
    high = np.maximum(open_price, close) * (1.0 + rng.uniform(0.001, 0.01, size=rows))
    low = np.minimum(open_price, close) * (1.0 - rng.uniform(0.001, 0.01, size=rows))
    return pd.DataFrame(
        {
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": rng.integers(100_000, 2_000_000, size=rows),
        },
        index=pd.bdate_range("2020-01-02", periods=rows),
    )


def test_target_uses_only_strictly_future_returns() -> None:
    close = np.exp(np.arange(20, dtype=float) / 100.0)
    target = realised_volatility(close, 3)
    expected = np.sqrt(252.0 / 3.0 * np.sum(np.diff(np.log(close))[0:3] ** 2))
    assert target[0] == pytest.approx(expected)
    assert np.isnan(target[-3:]).all()

    changed_origin = close.copy()
    changed_origin[0] *= 10.0
    np.testing.assert_allclose(target[1:], realised_volatility(changed_origin, 3)[1:])


def test_validation_rejects_bad_ohlc_relationships() -> None:
    frame = _frame(80)
    frame.loc[frame.index[5], "High"] = frame.loc[frame.index[5], "Close"] - 1
    with pytest.raises(ValueError, match="High"):
        validate_ohlcv(frame)


def test_split_is_chronological_and_label_purged() -> None:
    split = chronological_split(200, horizon=7)
    assert split.train[-1] < split.validation[0] < split.test[0]
    assert split.validation[0] - split.train[-1] >= 7
    assert split.test[0] - split.validation[-1] >= 7
    assert set(split.train).isdisjoint(split.validation)
    assert set(split.validation).isdisjoint(split.test)


def test_examples_expose_target_end_dates_and_date_purge_is_checked() -> None:
    from volatility_forecasting.simple_pipeline import assert_label_purged

    examples = build_examples(_frame(320))
    split = chronological_split(len(examples.target), horizon=examples.target_horizon)
    assert examples.target_end_dates is not None
    assert np.all(examples.target_end_dates > examples.dates)
    assert_label_purged(examples, split)


def test_har_baseline_is_canonical_and_horizon_bound() -> None:
    examples = build_examples(_frame(320))
    split = chronological_split(len(examples.target), horizon=5)
    forecasts = baseline_predictions(examples, split.train, horizon=5)
    np.testing.assert_allclose(forecasts["har_rv"], examples.canonical_har_volatility)
    with pytest.raises(ValueError, match="target horizon"):
        baseline_predictions(examples, split.train, horizon=1)


def test_examples_and_baselines_are_finite_and_causal() -> None:
    examples = build_examples(_frame())
    assert examples.sequences.ndim == 3
    assert examples.sequences.shape[1] == 22
    assert len(examples.target) == len(examples.dates)
    split = chronological_split(len(examples.target), horizon=5)
    forecasts = baseline_predictions(examples, split.train)
    assert {"persistence", "rolling_mean", "ewma", "har_rv"} <= set(forecasts)
    assert all(np.isfinite(values).all() and (values > 0).all() for values in forecasts.values())


def test_research_rolling_volatility_matches_deployable_c2c_definition() -> None:
    from panel.features import build_features_v5
    from volatility_forecasting.simple_pipeline import build_feature_frame

    frame = _frame(180)
    research_features = build_feature_frame(frame)
    deployable_features = build_features_v5(frame.copy())

    np.testing.assert_allclose(
        research_features["realized_vol_20"] / np.sqrt(252.0),
        deployable_features["Vol_C2C_20"],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        research_features["realized_vol_60"] / np.sqrt(252.0),
        deployable_features["Vol_C2C_60"],
        equal_nan=True,
    )


def test_benchmark_selects_by_validation_only_and_reports_qlike() -> None:
    examples = build_examples(_frame())
    split = chronological_split(len(examples.target), horizon=5)
    metrics = evaluate_benchmark(examples, split)
    assert {
        "persistence",
        "rolling_mean",
        "ewma",
        "har_rv",
        "ridge",
        "elastic_net",
        "gradient_boosting",
    } <= set(metrics)
    assert all("mae" in rows["test"] and "rmse" in rows["test"] for rows in metrics.values())
    selected = select_validation_model(metrics)
    assert selected in metrics
    assert metrics[selected]["validation"]["qlike"] == min(
        values["validation"]["qlike"] for values in metrics.values()
    )


def test_benchmark_uses_target_horizon_when_split_embargo_is_wider() -> None:
    examples = build_examples(_frame(340))
    split = chronological_split(
        len(examples.target), horizon=examples.target_horizon, embargo_sessions=12
    )

    metrics = evaluate_benchmark(examples, split)
    assert metrics["har_rv"]["test"]["price_cone"]["nominal_coverage"] == 0.90


def test_metrics_reject_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="matched"):
        volatility_metrics(np.ones(3), np.ones(2))


def test_optional_lstm_uses_train_only_rows_and_returns_metadata() -> None:
    pytest.importorskip("torch")
    examples = build_examples(_frame())
    split = chronological_split(len(examples.target), horizon=5)
    predictions, metadata = lstm_predictions(
        examples,
        split.train,
        split.validation,
        config=LSTMConfig(maximum_epochs=2, patience=1, batch_size=32, device="cpu"),
    )
    assert predictions.shape == examples.target.shape
    assert np.isfinite(predictions).all() and (predictions > 0).all()
    assert metadata["scaler"] == "train_only_standard"


def test_conformal_volatility_intervals_and_price_cone_calibration() -> None:
    from volatility_forecasting.simple_pipeline import (
        evaluate_conformal_volatility_intervals,
        evaluate_price_diffusion_cone,
    )

    examples = build_examples(_frame(300))
    split = chronological_split(len(examples.target), horizon=5)
    forecasts = baseline_predictions(examples, split.train)
    pred_val = forecasts["har_rv"][split.validation]
    act_val = examples.target[split.validation]
    pred_test = forecasts["har_rv"][split.test]
    act_test = examples.target[split.test]

    conf = evaluate_conformal_volatility_intervals(
        act_val, pred_val, act_test, pred_test, nominal_coverage=0.90
    )
    assert conf["nominal_coverage"] == 0.90
    assert 0.0 <= conf["empirical_coverage"] <= 1.0
    assert conf["average_width"] > 0
    assert "low_vol" in conf["regime_coverage"]
    assert conf["interval_method"] == "rolling_origin_split_conformal_log_volatility"
    assert conf["regime_thresholds"]["source"] == "validation_actual_only"

    assert examples.origin_close is not None and examples.future_close is not None
    cone = evaluate_price_diffusion_cone(
        examples.origin_close[split.test],
        examples.future_close[split.test],
        pred_test,
        horizon=5,
        nominal_coverage=0.90,
    )
    assert cone["nominal_coverage"] == 0.90
    assert 0.0 <= cone["empirical_coverage"] <= 1.0
    assert cone["average_width_pct"] > 0
    assert "high_vol" in cone["regime_coverage"]
    assert cone["interval_method"] == "gaussian_reference_scenario"
    assert cone["metric_source"] == "untouched_chronological_test_descriptive"


def test_qlike_mathematical_properties_and_hand_calculated_cases() -> None:
    # 1. Exact match produces zero loss
    actual = np.array([0.20, 0.30])  # sigma = 0.20, 0.30
    forecast = np.array([0.20, 0.30])
    metrics = volatility_metrics(actual, forecast)
    assert metrics["qlike"] == pytest.approx(0.0, abs=1e-10)

    # 2. Hand-calculated Case A: under-prediction by 2x variance
    # actual sigma = 0.20 (variance = 0.04), forecast sigma = sqrt(0.02) (variance = 0.02)
    # ratio = 0.04 / 0.02 = 2.0
    # QLIKE = 2.0 - ln(2.0) - 1.0 = 1.0 - ln(2.0) ~ 0.3068528194400547
    act_under = np.array([0.20])
    pred_under = np.array([np.sqrt(0.02)])
    m_under = volatility_metrics(act_under, pred_under)
    expected_under = 1.0 - np.log(2.0)
    assert m_under["qlike"] == pytest.approx(expected_under, rel=1e-6)

    # 3. Hand-calculated Case B: over-prediction by 2x variance
    # actual sigma = 0.20 (variance = 0.04), forecast sigma = sqrt(0.08) (variance = 0.08)
    # ratio = 0.04 / 0.08 = 0.5
    # QLIKE = 0.5 - ln(0.5) - 1.0 = ln(2.0) - 0.5 ~ 0.1931471805599453
    act_over = np.array([0.20])
    pred_over = np.array([np.sqrt(0.08)])
    m_over = volatility_metrics(act_over, pred_over)
    expected_over = np.log(2.0) - 0.5
    assert m_over["qlike"] == pytest.approx(expected_over, rel=1e-6)

    # 4. Asymmetry check: under-prediction is penalized more heavily than over-prediction
    assert m_under["qlike"] > m_over["qlike"]

    # 5. Annualization scale invariance: multiplying both actual and forecast sigma by sqrt(252)
    # leaves QLIKE exactly unchanged because scaling factor cancels in the variance ratio
    ann_factor = np.sqrt(252.0)
    m_ann = volatility_metrics(act_under * ann_factor, pred_under * ann_factor)
    assert m_ann["qlike"] == pytest.approx(m_under["qlike"], rel=1e-6)


def test_garch11_baseline_causal_mle_and_multihorizon() -> None:
    from volatility_forecasting.simple_pipeline import fit_garch11_baseline

    examples = build_examples(_frame(350))
    split = chronological_split(len(examples.target), horizon=5)

    garch_1d = fit_garch11_baseline(examples, split.train, horizon=1)
    garch_5d = fit_garch11_baseline(examples, split.train, horizon=5)
    garch_20d = fit_garch11_baseline(examples, split.train, horizon=20)

    assert len(garch_1d) == len(examples.target)
    assert len(garch_5d) == len(examples.target)
    assert len(garch_20d) == len(examples.target)
    assert np.all(garch_1d > 0) and np.all(np.isfinite(garch_1d))
    assert np.all(garch_5d > 0) and np.all(np.isfinite(garch_5d))
    assert np.all(garch_20d > 0) and np.all(np.isfinite(garch_20d))


def test_target_space_options_direct_and_log_variance() -> None:
    from volatility_forecasting.simple_pipeline import learned_predictions

    examples = build_examples(_frame(260))
    split = chronological_split(len(examples.target), horizon=5)

    preds_direct = learned_predictions(examples, split.train, target_space="direct_volatility")
    preds_logvar = learned_predictions(examples, split.train, target_space="log_variance")
    preds_logvol = learned_predictions(examples, split.train, target_space="log_volatility")

    for preds in (preds_direct, preds_logvar, preds_logvol):
        assert {"ridge", "elastic_net", "gradient_boosting"} <= set(preds)
        for _model_name, arr in preds.items():
            assert len(arr) == len(examples.target)
            assert np.all(arr > 0) and np.all(np.isfinite(arr))


def test_ohlc_volatility_estimators_and_integrity() -> None:
    from volatility_forecasting.simple_pipeline import build_feature_frame

    frame = _frame(150)
    features = build_feature_frame(frame, feature_mode="price_plus_ohlc")

    expected_cols = [
        "hl_range",
        "co_range",
        "overnight_return",
        "parkinson_vol_5",
        "parkinson_vol_22",
        "parkinson_vol_60",
        "garman_klass_vol_5",
        "garman_klass_vol_22",
        "garman_klass_vol_60",
        "rogers_satchell_vol_5",
        "rogers_satchell_vol_22",
        "rogers_satchell_vol_60",
    ]
    for col in expected_cols:
        assert col in features.columns
        valid_vals = features[col].dropna()
        assert len(valid_vals) > 0
        if "vol_" in col:
            assert (valid_vals >= 0).all()

    # Verify input integrity failure on malformed bars
    bad_frame = frame.copy()
    bad_frame.loc[bad_frame.index[10], "High"] = bad_frame.loc[bad_frame.index[10], "Close"] - 1.0
    with pytest.raises(ValueError, match="High"):
        build_feature_frame(bad_frame, feature_mode="price_plus_ohlc")


def test_softplus_volatility_lstm() -> None:
    pytest.importorskip("torch")
    from volatility_forecasting.simple_pipeline import LSTMConfig, build_examples, lstm_predictions

    examples = build_examples(_frame(260))
    split = chronological_split(len(examples.target), horizon=5)

    preds, meta = lstm_predictions(
        examples,
        split.train,
        split.validation,
        config=LSTMConfig(
            maximum_epochs=2, patience=1, batch_size=32, target_space="softplus_volatility"
        ),
    )
    assert len(preds) == len(examples.target)
    assert np.all(preds > 0) and np.all(np.isfinite(preds))
    assert meta["target_space"] == "softplus_volatility"


def test_nested_feature_modes_and_market_frame() -> None:
    from volatility_forecasting.simple_pipeline import VolatilityConfig, build_examples

    frame = _frame(200)

    # Market context frame
    mkt = pd.DataFrame(
        {
            "spy_return_1d": np.random.normal(0, 0.01, size=len(frame)),
            "spy_vol_22": np.full(len(frame), 0.15),
        },
        index=frame.index,
    )

    ex_price = build_examples(frame, VolatilityConfig(feature_mode="price_only"))
    ex_ohlc = build_examples(frame, VolatilityConfig(feature_mode="price_plus_ohlc"))
    ex_mkt = build_examples(
        frame, VolatilityConfig(feature_mode="price_plus_ohlc_plus_market"), market_frame=mkt
    )

    assert ex_price.sequences.shape[-1] < ex_ohlc.sequences.shape[-1] < ex_mkt.sequences.shape[-1]
    assert "parkinson_vol_22" not in ex_price.feature_names
    assert "parkinson_vol_22" in ex_ohlc.feature_names
    assert "mkt_spy_return_1d" in ex_mkt.feature_names


def test_volatility_metrics_distribution_diagnostics() -> None:
    act = np.array([0.20, 0.25, 0.30, 0.22, 0.80])
    pred = np.array([0.20, 0.24, 0.31, 0.21, 0.20])

    m = volatility_metrics(act, pred)
    assert "median_qlike" in m
    assert "p90_qlike" in m
    assert "p95_qlike" in m
    assert "p99_qlike" in m
    assert "max_qlike" in m
    assert "worst_1pct_share" in m
    assert 0.0 <= m["worst_1pct_share"] <= 100.0
    assert m["max_qlike"] >= m["p95_qlike"] >= m["median_qlike"]


def test_causal_news_features_cutoff_and_integrity() -> None:
    from volatility_forecasting.simple_pipeline import (
        NEWS_FEATURE_NAMES,
        build_causal_news_features,
    )

    sessions = pd.date_range("2025-01-02", "2025-01-10", freq="B")

    # Construct two articles for day 2025-01-03:
    # Article 1: 2025-01-03 14:00:00 UTC (10:00 AM ET - before close) -> participates in 2025-01-03
    # Article 2: 2025-01-03 21:30:00 UTC (17:30 PM ET - after close) -> must NOT participate in 2025-01-03
    events = [
        {
            "ticker": "AAPL",
            "published_at": pd.Timestamp("2025-01-03T14:00:00Z"),
            "sentiment_pos": 0.8,
            "sentiment_neg": 0.1,
            "sentiment_compound": 0.7,
        },
        {
            "ticker": "AAPL",
            "published_at": pd.Timestamp("2025-01-03T21:30:00Z"),
            "sentiment_pos": 0.1,
            "sentiment_neg": 0.9,
            "sentiment_compound": -0.8,
        },
    ]

    news_df = build_causal_news_features(sessions, ticker="AAPL", news_events=events)

    for name in NEWS_FEATURE_NAMES:
        assert name in news_df.columns

    # On 2025-01-03 at 16:00 ET cutoff: only Article 1 is visible
    day_3 = news_df.loc[pd.Timestamp("2025-01-03")]
    assert day_3["news_headline_count_1d"] == 1.0
    assert day_3["news_negative_sentiment_mean"] == pytest.approx(0.1)
    assert day_3["news_positive_sentiment_mean"] == pytest.approx(0.8)

    # On 2025-01-06 (Monday 21:00 UTC):
    # 72h window is [2025-01-03 21:00 UTC, 2025-01-06 21:00 UTC] -> Article 2 (Fri 21:30 UTC) is in 3d window
    # 168h window (7d) includes Article 1 (Fri 14:00 UTC) and Article 2 (Fri 21:30 UTC)
    day_6 = news_df.loc[pd.Timestamp("2025-01-06")]
    assert day_6["news_headline_count_1d"] == 0.0
    assert day_6["news_headline_count_3d"] == 1.0
    assert day_6["news_headline_count_7d"] == 2.0
    assert day_6["news_negative_sentiment_mean"] == pytest.approx(0.9)  # Article 2
    assert day_6["news_positive_sentiment_mean"] == pytest.approx(0.1)  # Article 2

    # Monotonicity check
    assert np.all(news_df["news_headline_count_1d"] <= news_df["news_headline_count_3d"])
    assert np.all(news_df["news_headline_count_3d"] <= news_df["news_headline_count_7d"])

    # Test explicit DST transition (Winter EST 21:00 UTC vs Summer EDT 20:00 UTC)
    # Article at 20:30 UTC:
    # On 2025-01-15 (Winter EST): 20:30 UTC < 21:00 UTC cutoff -> visible on 2025-01-15 (1d count = 1)
    # On 2025-07-15 (Summer EDT): 20:30 UTC > 20:00 UTC cutoff -> NOT visible on 2025-07-15 (1d count = 0)
    w_df = build_causal_news_features(
        pd.to_datetime(["2025-01-15"]),
        ticker="AAPL",
        news_events=[
            {
                "ticker": "AAPL",
                "published_at": pd.Timestamp("2025-01-15T20:30:00Z"),
                "sentiment_pos": 0.5,
                "sentiment_neg": 0.0,
            }
        ],
    )
    assert w_df.loc[pd.Timestamp("2025-01-15"), "news_headline_count_1d"] == 1.0

    s_df = build_causal_news_features(
        pd.to_datetime(["2025-07-15"]),
        ticker="AAPL",
        news_events=[
            {
                "ticker": "AAPL",
                "published_at": pd.Timestamp("2025-07-15T20:30:00Z"),
                "sentiment_pos": 0.5,
                "sentiment_neg": 0.0,
            }
        ],
    )
    assert s_df.loc[pd.Timestamp("2025-07-15"), "news_headline_count_1d"] == 0.0

    # Test early-close sessions (Black Friday 18:00 UTC cutoff, July 3rd 17:00 UTC cutoff)
    # On Black Friday (2024-11-29), early close is 13:00 EST = 18:00 UTC:
    # Article at 17:30 UTC is BEFORE early close -> visible (count = 1)
    # Article at 18:30 UTC is AFTER early close -> excluded (count = 0)
    bf_pre = build_causal_news_features(
        pd.to_datetime(["2024-11-29"]),
        ticker="AAPL",
        news_events=[
            {
                "ticker": "AAPL",
                "published_at": pd.Timestamp("2024-11-29T17:30:00Z"),
                "sentiment_pos": 0.5,
                "sentiment_neg": 0.0,
            }
        ],
    )
    assert bf_pre.loc[pd.Timestamp("2024-11-29"), "news_headline_count_1d"] == 1.0

    bf_post = build_causal_news_features(
        pd.to_datetime(["2024-11-29"]),
        ticker="AAPL",
        news_events=[
            {
                "ticker": "AAPL",
                "published_at": pd.Timestamp("2024-11-29T18:30:00Z"),
                "sentiment_pos": 0.5,
                "sentiment_neg": 0.0,
            }
        ],
    )
    assert bf_post.loc[pd.Timestamp("2024-11-29"), "news_headline_count_1d"] == 0.0


def test_news_feature_mode_nesting_and_examples() -> None:
    from volatility_forecasting.simple_pipeline import VolatilityConfig, build_examples

    frame = _frame(150)
    mkt = pd.DataFrame(
        {"spy_return_1d": np.zeros(len(frame)), "spy_vol_22": np.full(len(frame), 0.15)},
        index=frame.index,
    )

    ex_mkt = build_examples(
        frame, VolatilityConfig(feature_mode="price_plus_ohlc_plus_market"), market_frame=mkt
    )
    ex_news = build_examples(
        frame,
        VolatilityConfig(feature_mode="price_plus_ohlc_plus_market_plus_news"),
        market_frame=mkt,
        ticker="TEST",
    )

    assert ex_news.sequences.shape[-1] == ex_mkt.sequences.shape[-1] + 10
    assert "news_headline_count_1d" in ex_news.feature_names
    assert "news_negative_news_intensity" in ex_news.feature_names
