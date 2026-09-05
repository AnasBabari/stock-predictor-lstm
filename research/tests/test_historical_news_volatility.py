import json

import numpy as np
import pandas as pd

from scripts.run_historical_news_volatility import (
    FEATURE_NAMES,
    NEWS_FEATURE_NAMES,
    canonicalize,
    fit_and_score,
)


def article(updated="2023-01-03T10:01:00Z", headline="CEO resigns!"):
    return {
        "provider": "alpaca",
        "id": "1",
        "ticker": "AAPL",
        "published_at": "2023-01-03T10:00:00Z",
        "provider_updated_at": updated,
        "headline": headline,
    }


def test_revision_missingness_and_late_revisions_excluded():
    rows, stats = canonicalize([article(), article(None), article("2023-01-04T10:00:00Z")])
    assert len(rows) == 1
    assert rows[0]["t_available"] == "2023-01-03T10:01:00Z"
    assert rows[0]["event_type"] == "management"
    assert stats["missing_updates"] == 1


def test_canonical_headline_duplicate_keeps_earliest_available():
    later = article("2023-01-03T10:05:00Z", "CEO resigns")
    later["provider"] = "other"
    rows, stats = canonicalize([later, article()])
    assert len(rows) == 1
    assert stats["duplicate_count"] == 1
    assert rows[0]["provider"] == "alpaca"


def test_matched_fit_excludes_reserve_and_neutral_news_is_identical(tmp_path):
    dates = pd.bdate_range("2018-01-01", "2025-03-31")
    prices = pd.DataFrame(
        {"Close": 100 * np.exp(np.sin(np.arange(len(dates)) / 8) * 0.02)}, index=dates
    )
    features = pd.DataFrame(
        0.0, index=dates, columns=list(FEATURE_NAMES) + list(NEWS_FEATURE_NAMES)
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    fit_and_score({"AAPL": (prices, features)}, first)
    predictions = pd.read_parquet(first / "validation_predictions.parquet")
    assert predictions.date.min() >= pd.Timestamp("2023-01-01")
    assert predictions.date.max() < pd.Timestamp("2025-01-01")
    assert np.allclose(predictions.A_ohlcv_variance, predictions.C_sentiment_variance)
    prices.loc[prices.index >= "2025-01-01", "Close"] *= 100
    features.loc[features.index >= "2025-01-01"] = 1e6
    fit_and_score({"AAPL": (prices, features)}, second)
    assert json.loads((first / "results.json").read_text()) == json.loads(
        (second / "results.json").read_text()
    )
