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


def test_examples_and_baselines_are_finite_and_causal() -> None:
    examples = build_examples(_frame())
    assert examples.sequences.ndim == 3
    assert examples.sequences.shape[1] == 22
    assert len(examples.target) == len(examples.dates)
    split = chronological_split(len(examples.target), horizon=5)
    forecasts = baseline_predictions(examples, split.train)
    assert {"persistence", "rolling_mean", "ewma", "har_rv"} <= set(forecasts)
    assert all(np.isfinite(values).all() and (values > 0).all() for values in forecasts.values())


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
