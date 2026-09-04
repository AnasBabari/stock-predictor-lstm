from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from research.price_forecasting.gpu_pipeline import (
    FEATURE_NAMES,
    PriceTrainingConfig,
    _build_model,
    build_global_price_dataset,
)
from research.price_forecasting.news_archive import (
    NEWS_FEATURE_NAMES,
    _normalise_article,
    build_causal_news_features,
    load_news_archive,
    merge_news_archive,
    validate_news_archive,
)


def _frame(rows: int = 700) -> pd.DataFrame:
    index = pd.bdate_range("2022-01-03", periods=rows)
    trend = 100.0 * np.exp(np.linspace(0.0, 0.35, rows))
    wave = 1.0 + 0.01 * np.sin(np.arange(rows) / 9.0)
    close = trend * wave
    open_price = close * (1.0 + 0.001 * np.cos(np.arange(rows) / 7.0))
    return pd.DataFrame(
        {
            "open": open_price,
            "high": np.maximum(open_price, close) * 1.01,
            "low": np.minimum(open_price, close) * 0.99,
            "close": close,
            "volume": 1_000_000 + np.arange(rows) * 100,
        },
        index=index,
    )


def test_global_price_dataset_has_purged_chronological_partitions() -> None:
    config = PriceTrainingConfig(lookback=60, horizon=7, maximum_epochs=2, patience=1)
    dataset = build_global_price_dataset({"MSFT": _frame(), "AAPL": _frame()}, config)
    partitions = (dataset.split_train, dataset.split_validation, dataset.split_test)
    assert all(len(partition) > 30 for partition in partitions)
    assert not set(partitions[0]) & set(partitions[1])
    assert not set(partitions[1]) & set(partitions[2])
    for ticker_index in range(2):
        train = partitions[0][dataset.ticker_indices[partitions[0]] == ticker_index]
        validation = partitions[1][dataset.ticker_indices[partitions[1]] == ticker_index]
        test = partitions[2][dataset.ticker_indices[partitions[2]] == ticker_index]
        assert (
            dataset.target_end_positions[train].max() < dataset.origin_positions[validation].min()
        )
        assert dataset.target_end_positions[validation].max() < dataset.origin_positions[test].min()
    assert dataset.sequences.shape[1:] == (60, 25)
    assert dataset.targets.shape[1] == 7
    assert dataset.feature_mode == "price_only"


def test_news_archive_merge_is_deduplicated_and_checksummed(tmp_path: Path) -> None:
    path = tmp_path / "MSFT.jsonl"
    records = [
        {"id": "b", "published_at": "2025-01-02T12:00:00Z", "headline": "two"},
        {"id": "a", "published_at": "2025-01-01T12:00:00Z", "headline": "one"},
    ]
    first = merge_news_archive(path, records)
    second = merge_news_archive(path, [records[0]])
    assert first["article_count"] == second["article_count"] == 2
    assert first["sha256"] == second["sha256"]
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [item["id"] for item in lines] == ["a", "b"]

    loaded = load_news_archive(path)
    assert len(loaded) == 2
    assert loaded[0]["id"] == "a"


def test_news_archive_yahoo_and_edgar_normalization() -> None:
    # Yahoo Finance structure normalization
    yahoo_item = {
        "id": "yh-12345",
        "content": {
            "title": "Microsoft Surpasses Revenue Expectations with Record Profits",
            "pubDate": "2026-09-04T14:30:00Z",
            "provider": {"displayName": "Reuters"},
            "canonicalUrl": {"url": "https://example.com/reuters-msft"},
        },
    }
    norm_yahoo = _normalise_article(yahoo_item, "MSFT", "2026-09-04T15:00:00Z")
    assert norm_yahoo is not None
    assert norm_yahoo["ticker"] == "MSFT"
    assert norm_yahoo["provider"] == "yahoo"
    assert "Microsoft Surpasses" in norm_yahoo["headline"]
    assert norm_yahoo["published_at"] == "2026-09-04T14:30:00Z"
    assert norm_yahoo["source"] == "Reuters"
    assert norm_yahoo["sentiment_pos"] > 0

    # SEC EDGAR structure normalization
    edgar_item = {
        "accessionNumber": "sec-0001193125-26-123456",
        "acceptanceDateTime": "2026-09-01T20:30:35.000Z",
        "primaryDocDescription": "AAPL Form 8-K: Earnings Release",
        "symbols": ["AAPL"],
    }
    norm_edgar = _normalise_article(edgar_item, "AAPL", "2026-09-04T15:00:00Z")
    assert norm_edgar is not None
    assert norm_edgar["ticker"] == "AAPL"
    assert norm_edgar["provider"] == "sec_edgar"
    assert norm_edgar["headline"] == "AAPL Form 8-K: Earnings Release"
    assert norm_edgar["published_at"] == "2026-09-01T20:30:35Z"


def test_causal_news_features_enforces_market_close_cutoff() -> None:
    sessions = pd.DatetimeIndex(["2024-07-15", "2024-07-16"])  # Summer EDT: 16:00 EDT = 20:00 UTC
    events = [
        {
            "id": "before-close",
            "ticker": "AAPL",
            "published_at": "2024-07-15T19:30:00Z",  # Before 20:00 close of 2024-07-15
            "headline": "AAPL Record Profits Announced",
            "sentiment_pos": 0.8,
            "sentiment_neg": 0.0,
            "sentiment_compound": 0.8,
        },
        {
            "id": "after-close",
            "ticker": "AAPL",
            "published_at": "2024-07-15T20:30:00Z",  # After 20:00 close of 2024-07-15
            "headline": "AAPL Evening Late News Event",
            "sentiment_pos": 0.0,
            "sentiment_neg": 0.9,
            "sentiment_compound": -0.9,
        },
    ]
    features = build_causal_news_features(sessions, "AAPL", events)
    assert len(features) == 2
    assert list(features.columns) == list(NEWS_FEATURE_NAMES)

    # Day 1: Only the event BEFORE 20:00 UTC participates in session 2024-07-15
    assert features.loc["2024-07-15", "news_headline_count_1d"] == 1.0
    assert features.loc["2024-07-15", "news_positive_sentiment_mean"] == pytest.approx(0.8)
    assert features.loc["2024-07-15", "news_negative_sentiment_mean"] == pytest.approx(0.0)

    # Day 2: The after-close event is within 24h, and both events are within 72h (3d)
    assert features.loc["2024-07-16", "news_headline_count_1d"] == 1.0
    assert features.loc["2024-07-16", "news_headline_count_3d"] == 2.0


def test_news_feature_mode_fails_closed_without_archive() -> None:
    config = PriceTrainingConfig(lookback=60, horizon=7, feature_mode="price_plus_news")
    with pytest.raises(ValueError, match="Historical news events missing"):
        build_global_price_dataset(
            {"MSFT": _frame(), "AAPL": _frame()}, config, feature_mode="price_plus_news"
        )


def test_news_feature_mode_produces_35_features() -> None:
    frame_a = _frame(700)
    frame_m = _frame(700)
    events_a = [
        {
            "id": "a-1",
            "ticker": "AAPL",
            "published_at": "2022-06-01T15:00:00Z",
            "headline": "Apple WWDC Innovations",
            "sentiment_pos": 0.6,
            "sentiment_neg": 0.0,
            "sentiment_compound": 0.6,
        }
    ]
    events_m = [
        {
            "id": "m-1",
            "ticker": "MSFT",
            "published_at": "2022-06-01T15:00:00Z",
            "headline": "Microsoft Cloud Growth",
            "sentiment_pos": 0.5,
            "sentiment_neg": 0.0,
            "sentiment_compound": 0.5,
        }
    ]
    dataset = build_global_price_dataset(
        {"AAPL": frame_a, "MSFT": frame_m},
        feature_mode="price_plus_news",
        news_archives={"AAPL": events_a, "MSFT": events_m},
    )
    assert dataset.feature_mode == "price_plus_news"
    assert len(dataset.feature_names) == 35
    assert dataset.feature_names == FEATURE_NAMES + NEWS_FEATURE_NAMES
    assert dataset.sequences.shape[1:] == (60, 35)


def test_temporal_attention_lstm_forward_pass() -> None:
    settings = PriceTrainingConfig(hidden_size=32, layers=2, horizon=7, use_attention=True)
    model = _build_model(torch, nn, feature_count=25, ticker_count=5, settings=settings)
    batch_size = 8
    dummy_x = torch.randn(batch_size, 60, 25)
    dummy_tickers = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2], dtype=torch.long)
    output = model(dummy_x, dummy_tickers)
    assert output.shape == (batch_size, 7)
    assert torch.isfinite(output).all()
    # Verify residual skip connection is zero-initialized to avoid drowning out LSTM representation
    assert torch.all(model.skip.weight == 0.0)


def test_direction_penalty_evaluates_true_return_signs() -> None:
    # Target mean is positive (drift), e.g. 0.02, std is 0.05
    target_mean = torch.tensor([0.02], dtype=torch.float32)
    target_std = torch.tensor([0.05], dtype=torch.float32)

    # Case 1: Actual return is +0.01 (positive, but below mean 0.02).
    # Standardized y = (0.01 - 0.02) / 0.05 = -0.2 (negative in standardized space!)
    # Model predicts return +0.03 (positive return). Standardized pred = (0.03 - 0.02) / 0.05 = +0.2
    y_std = torch.tensor([[-0.2]], dtype=torch.float32)
    pred_std = torch.tensor([[0.2]], dtype=torch.float32)

    unscaled_pred = pred_std * target_std + target_mean  # +0.03
    unscaled_target = y_std * target_std + target_mean  # +0.01

    penalty = torch.mean(torch.relu(-torch.sign(unscaled_target) * (unscaled_pred / target_std)))
    # Both returns are positive: no penalty should be incurred!
    assert penalty.item() == 0.0

    # Case 2: Actual return is +0.01, but model predicts negative return -0.01.
    # Opposite signs: penalty must be positive!
    pred_std_wrong = torch.tensor([[-0.6]], dtype=torch.float32)  # -0.01 unscaled
    unscaled_pred_wrong = pred_std_wrong * target_std + target_mean
    penalty_wrong = torch.mean(
        torch.relu(-torch.sign(unscaled_target) * (unscaled_pred_wrong / target_std))
    )
    assert penalty_wrong.item() > 0.0


def test_validate_news_archive_diagnostics_and_coverage() -> None:
    sessions = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    diag_empty = validate_news_archive([], sessions)
    assert not diag_empty["is_valid"]
    assert diag_empty["article_count"] == 0
    assert diag_empty["coverage_1d"] == 0.0

    # 1 article published on 2024-01-02 at 10:00 UTC
    records = [{"published_at": "2024-01-02T10:00:00Z"}]
    diag_valid = validate_news_archive(records, sessions)
    assert diag_valid["is_valid"]
    assert diag_valid["article_count"] == 1
    assert "coverage_1d" in diag_valid
    assert "coverage_3d" in diag_valid
    assert "coverage_7d" in diag_valid
    # Day 1: 10:00 UTC is within 24h of 2024-01-02 21:00 UTC close -> coverage_1d = 0.5 (1 of 2 sessions)
    assert diag_valid["coverage_1d"] == 0.5
    # Day 2: 10:00 UTC is within 72h of 2024-01-03 21:00 UTC close -> coverage_3d = 1.0 (2 of 2 sessions)
    assert diag_valid["coverage_3d"] == 1.0


def test_news_manifest_records_detected_providers(tmp_path: Path) -> None:
    path = tmp_path / "AAPL.jsonl"
    records = [
        {
            "id": "sec-1",
            "published_at": "2024-01-01T12:00:00Z",
            "headline": "filing",
            "provider": "sec_edgar",
        },
        {
            "id": "yh-1",
            "published_at": "2024-01-02T12:00:00Z",
            "headline": "news",
            "provider": "yahoo",
        },
    ]
    manifest = merge_news_archive(path, records)
    assert manifest["provider"] == "sec_edgar,yahoo"
    assert manifest["article_count"] == 2
