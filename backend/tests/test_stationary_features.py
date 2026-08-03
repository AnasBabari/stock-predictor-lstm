"""Tests for the stationary schema-v4 features and snapshot quality diagnostics."""

from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd

from config import FEATURES_V4
from data_pipeline import snapshot_quality_diagnostics
from features.pipeline import build_browser_features
from features.stationary import add_stationary_features
from features.technical import add_technical_indicators


def make_ohlcv_frame():
    """Deterministic 300-row OHLCV frame with a clean price path."""
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    closes = 100.0 * np.exp(0.001 * np.arange(300))
    opens = closes * np.exp(-0.001)
    highs = closes * np.exp(0.002)
    lows = closes * np.exp(-0.003)
    volumes = 1_000_000 + np.arange(300) * 1000.0
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes},
        index=dates,
    )


def _context_frame(df):
    """Append technical indicators and constant market returns (no network)."""
    result = add_technical_indicators(df)
    result["SPY_Return_1D"] = 0.0005
    result["QQQ_Return_1D"] = 0.0006
    result["VIX_Return_1D"] = -0.0002
    result["TNX_Return_1D"] = 0.0001
    return result


def test_stationary_features_are_price_relative():
    df = _context_frame(make_ohlcv_frame())
    result = add_stationary_features(df)
    close = df["Close"]
    prev = close.shift(1)

    np.testing.assert_allclose(
        result["Log_Open_Rel"].iloc[1:], np.log(df["Open"].iloc[1:] / prev.iloc[1:]), rtol=1e-10
    )
    np.testing.assert_allclose(
        result["Return_1D"].iloc[1:], np.log(close.iloc[1:] / prev.iloc[1:]), rtol=1e-10
    )
    # The open price matches the prior close by construction.
    np.testing.assert_allclose(result["Log_Open_Rel"].dropna(), 0.0, atol=1e-8)
    # RSI centered and MACD ratios stay in plausible ratio ranges.
    assert (result["RSI_14_Centered"].abs() <= 1.0).all()
    assert (result["Close_SMA_20"].dropna() > -0.5).all()


def test_stationary_features_are_causal():
    """Row t of a rolling feature must not change when future rows change."""
    df = _context_frame(make_ohlcv_frame())
    future = df.copy()
    future.loc[future.index[-1], "Close"] = future.loc[future.index[-1], "Close"] * 2.0

    before = add_stationary_features(df)
    after = add_stationary_features(future)
    # The final row itself may change; every earlier row must be identical.
    for column in before.columns:
        np.testing.assert_allclose(
            before[column].iloc[:-1].to_numpy(),
            after[column].iloc[:-1].to_numpy(),
            rtol=1e-10,
            equal_nan=True,
        )


def test_build_browser_features_matches_schema_and_is_deterministic():
    df = make_ohlcv_frame()
    with patch(
        "features.pipeline.add_market_context",
        return_value=(_context_frame(df), {"status": "test", "sources": {}}),
    ):
        features_a, metadata_a = build_browser_features(df, FEATURES_V4)
        features_b, metadata_b = build_browser_features(df, FEATURES_V4)

    assert list(features_a.columns) == list(FEATURES_V4)
    assert not features_a.isna().any().any()
    assert np.isfinite(features_a.values).all()
    pd.testing.assert_frame_equal(features_a, features_b)
    assert metadata_a["schema_version"] == 4
    assert metadata_a["feature_schema"] == "stationary_v4"
    assert metadata_a["date_range"].startswith("2020-")


def test_browser_features_are_identical_across_regime_shifts():
    """Feature identity must not depend on the absolute price level."""
    df = make_ohlcv_frame()
    shifted = df.copy()
    shifted[["Open", "High", "Low", "Close"]] = shifted[["Open", "High", "Low", "Close"]] * 10.0

    with patch(
        "features.pipeline.add_market_context",
        return_value=(_context_frame(shifted), {"status": "test", "sources": {}}),
    ):
        base = build_browser_features(df, FEATURES_V4)[0]
        rescaled = build_browser_features(shifted, FEATURES_V4)[0]
    for column in FEATURES_V4:
        if column in {"Month_Sin", "Month_Cos", "Day_Sin", "Day_Cos"}:
            continue
        np.testing.assert_allclose(base[column].to_numpy(), rescaled[column].to_numpy(), rtol=1e-9)


def test_quality_diagnostics_detects_large_moves_and_duplicates():
    dates = pd.date_range("2024-01-01", periods=120, freq="B")
    prices = np.linspace(100.0, 150.0, 120)
    prices[60] = prices[59] * 1.3  # a 30% single-day move
    df = pd.DataFrame({"Open": prices, "High": prices, "Low": prices, "Close": prices}, index=dates)
    duplicate = df.iloc[[10]].copy()
    df = pd.concat([df.iloc[:10], duplicate, df.iloc[10:]])
    duplicate_prices = np.insert(prices, 10, prices[10])

    diagnostics = snapshot_quality_diagnostics(df, duplicate_prices, df.index, {})
    codes = {issue["code"] for issue in diagnostics["issues"]}
    assert "large_single_day_move" in codes
    assert "duplicate_dates" in codes
    assert diagnostics["checks"]["duplicate_dates"] == 1


def test_quality_diagnostics_annotates_stale_and_benchmark_gaps():
    dates = pd.date_range(datetime.now() - timedelta(days=400), periods=100, freq="B")
    prices = np.linspace(100.0, 110.0, 100)
    df = pd.DataFrame({"Open": prices, "High": prices, "Low": prices, "Close": prices}, index=dates)
    sources = {"SPY_Return_1D": {"last_date": "2024-01-01"}}
    diagnostics = snapshot_quality_diagnostics(df, prices, dates, sources)
    codes = {issue["code"] for issue in diagnostics["issues"]}
    assert "stale_latest_observation" in codes
    assert "benchmark_series_end_earlier" in codes


def test_quality_diagnostics_clean_snapshot():
    dates = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=100)
    prices = np.linspace(100.0, 110.0, 100)
    df = pd.DataFrame({"Open": prices, "High": prices, "Low": prices, "Close": prices}, index=dates)
    diagnostics = snapshot_quality_diagnostics(df, prices, dates, {})
    assert diagnostics["status"] == "clean"
    assert diagnostics["issues"] == []
