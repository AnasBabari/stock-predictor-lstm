from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from research.volatility_forecasting.data import VolatilityPanelExamples
from research.volatility_forecasting.model import (
    BaselineResidualTCNConfig,
    TorchTrainingConfig,
)
from research.volatility_forecasting.news_candidate_v8 import (
    build_v8_news_negative_controls,
    evaluate_v8_news_ablation,
)
from research.volatility_forecasting.v8_protocol import v8_protocol


def _examples() -> VolatilityPanelExamples:
    rows = 20
    return VolatilityPanelExamples(
        features=np.ones((rows, 22, 2), dtype=np.float32),
        baseline_variance=np.ones((rows, 6), dtype=np.float32),
        realized_variance=np.ones((rows, 6), dtype=np.float32),
        cumulative_returns=np.zeros((rows, 6), dtype=np.float32),
        direction_classes=np.ones((rows, 6), dtype=np.int64),
        tickers=np.full(rows, "MSFT"),
        origin_dates=np.arange(np.datetime64("2024-01-01"), np.datetime64("2024-01-21")),
        origin_closes=np.ones(rows),
        horizons=(1, 3, 5, 7, 14, 30),
        feature_names=("a", "b"),
    )


def test_paired_ablation_uses_identical_oof_rows_and_news_only_on_fused_model(monkeypatch) -> None:
    examples = _examples()
    calls = []

    def fake_evaluate(*args, **kwargs):
        calls.append(kwargs)
        indices = np.arange(10, 20)
        config = kwargs["model_config"]
        predictions = SimpleNamespace(variance=np.ones((10, 6)))
        fold = SimpleNamespace(
            metrics=tuple({"relative_qlike": 0.9} for _horizon in examples.horizons)
        )
        return SimpleNamespace(
            oof_indices=indices,
            folds=(fold,) * 5,
            predictions=predictions,
            promotion=tuple(
                SimpleNamespace(volatility_promoted=True) for _horizon in examples.horizons
            ),
            config=config,
        )

    monkeypatch.setattr(
        "research.volatility_forecasting.news_candidate_v8.evaluate_tcn_development",
        fake_evaluate,
    )
    monkeypatch.setattr(
        "research.volatility_forecasting.news_candidate_v8.assess_news_ablation",
        lambda **kwargs: tuple(
            SimpleNamespace(promoted=True, horizon=horizon) for horizon in kwargs["horizons"]
        ),
    )
    protocol = v8_protocol(news_enabled=True)
    architecture = BaselineResidualTCNConfig(
        feature_count=2,
        horizon_count=6,
        window_size=22,
    )
    result = evaluate_v8_news_ablation(
        examples=examples,
        fold_plan=SimpleNamespace(),
        protocol=protocol,
        news_features=np.ones((20, 3), dtype=np.float32),
        seeds=(41,),
        market_architecture=architecture,
        training_config=TorchTrainingConfig(maximum_epochs=1, patience=1),
        device="cpu",
    )
    assert result[0].promoted is True
    assert calls[0].get("news_features") is None
    assert calls[0]["model_config"].news_feature_count == 0
    assert calls[1]["news_features"].shape == (20, 3)
    assert calls[1]["model_config"].news_feature_count == 3
    assert len(calls) == 4
    assert result[0].negative_controls[0].rejected is True
    assert result[0].negative_controls[1].rejected is True


def test_news_ablation_rejects_misaligned_rows() -> None:
    examples = _examples()
    architecture = BaselineResidualTCNConfig(
        feature_count=2,
        horizon_count=6,
        window_size=22,
    )
    with pytest.raises(ValueError, match="aligned feature matrix"):
        evaluate_v8_news_ablation(
            examples=examples,
            fold_plan=SimpleNamespace(),
            protocol=v8_protocol(news_enabled=True),
            news_features=np.ones((19, 3), dtype=np.float32),
            seeds=(41,),
            market_architecture=architecture,
            training_config=TorchTrainingConfig(maximum_epochs=1, patience=1),
            device="cpu",
        )


def test_negative_controls_never_move_future_rows_backward() -> None:
    examples = _examples()
    examples = VolatilityPanelExamples(
        **{
            **examples.__dict__,
            "tickers": np.asarray(["A", "B"] * 10),
            "origin_dates": np.repeat(
                np.arange(np.datetime64("2024-01-01"), np.datetime64("2024-01-11")), 2
            ),
        }
    )
    news = np.arange(20, dtype=np.float32).reshape(20, 1)
    controls = build_v8_news_negative_controls(examples, news, seed=7, delay_days=2)

    shuffled = controls["cross_sectionally_shuffled_news"][:, 0]
    for rows in np.arange(20).reshape(10, 2):
        assert set(shuffled[rows]) == set(news[rows, 0])

    delayed = controls["news_delayed_seven_days"][:, 0]
    np.testing.assert_array_equal(delayed[:4], 0.0)
    np.testing.assert_array_equal(delayed[4:], news[:-4, 0])


def test_negative_controls_validate_delay_and_finite_values() -> None:
    examples = _examples()
    with pytest.raises(ValueError, match="positive delay"):
        build_v8_news_negative_controls(examples, np.ones((20, 1)), delay_days=0)
    invalid = np.ones((20, 1))
    invalid[3, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        build_v8_news_negative_controls(examples, invalid)
