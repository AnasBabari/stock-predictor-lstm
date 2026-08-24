from __future__ import annotations

import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from volatility_forecasting.model import (  # noqa: E402
    BaselineResidualTCN,
    BaselineResidualTCNConfig,
    CausalConv1d,
    RobustSequenceScaler,
    TorchTrainingConfig,
    train_baseline_residual_tcn,
    volatility_multitask_loss,
)


def _batch(rows: int = 16, window: int = 60, features: int = 6, horizons: int = 3):
    rng = np.random.default_rng(9)
    x = rng.normal(size=(rows, window, features)).astype(np.float32)
    baseline = np.exp(rng.normal(-8.0, 0.2, size=(rows, horizons))).astype(np.float32)
    target_var = baseline * np.exp(0.25 * x[:, -1, :1])
    target_var = np.repeat(target_var[:, :1], horizons, axis=1).astype(np.float32)
    returns = rng.normal(size=(rows, horizons)).astype(np.float32) * np.sqrt(target_var)
    direction = np.where(returns < -0.001, 0, np.where(returns > 0.001, 2, 1)).astype(np.int64)
    return x, baseline, target_var, returns, direction


def test_zero_initialized_heads_begin_at_matched_baselines() -> None:
    x, baseline, *_ = _batch()
    config = BaselineResidualTCNConfig(feature_count=x.shape[-1], horizon_count=baseline.shape[-1])
    model = BaselineResidualTCN(config).eval()
    with torch.no_grad():
        forecast_var, return_location, logits, residual = model(
            torch.from_numpy(x), torch.from_numpy(baseline)
        )

    np.testing.assert_allclose(forecast_var.numpy(), baseline, rtol=1e-6)
    np.testing.assert_array_equal(return_location.numpy(), np.zeros_like(return_location.numpy()))
    np.testing.assert_array_equal(residual.numpy(), np.zeros_like(residual.numpy()))
    assert logits.shape == (len(x), baseline.shape[1], 3)


def test_causal_convolution_output_before_perturbation_is_unchanged() -> None:
    torch.manual_seed(1)
    layer = CausalConv1d(2, 3, kernel_size=3, dilation=2).eval()
    values = torch.randn(1, 2, 20)
    changed = values.clone()
    changed[:, :, 12:] += 1000.0
    with torch.no_grad():
        original_output = layer(values)
        changed_output = layer(changed)
    torch.testing.assert_close(original_output[:, :, :12], changed_output[:, :, :12])
    assert not torch.allclose(original_output[:, :, 15:], changed_output[:, :, 15:])


def test_news_channel_is_explicit_and_changes_the_shared_representation() -> None:
    x, baseline, *_ = _batch()
    model = BaselineResidualTCN(
        BaselineResidualTCNConfig(
            feature_count=x.shape[-1],
            horizon_count=baseline.shape[-1],
            news_feature_count=5,
            news_channels=4,
            dropout=0.0,
        )
    ).eval()
    news = torch.zeros((len(x), 5))
    changed = news.clone()
    changed[:, 0] = 3.0
    with torch.no_grad():
        first = model(torch.from_numpy(x), torch.from_numpy(baseline), news)
        second = model(torch.from_numpy(x), torch.from_numpy(baseline), changed)
    assert not torch.allclose(first[2], second[2])
    with pytest.raises(ValueError, match="requires aligned"):
        model(torch.from_numpy(x), torch.from_numpy(baseline))


def test_market_only_model_rejects_news_features() -> None:
    x, baseline, *_ = _batch()
    model = BaselineResidualTCN(
        BaselineResidualTCNConfig(feature_count=x.shape[-1], horizon_count=baseline.shape[-1])
    )
    with pytest.raises(ValueError, match="cannot accept news"):
        model(torch.from_numpy(x), torch.from_numpy(baseline), torch.zeros((len(x), 1)))


def test_multitask_loss_is_finite_and_backpropagates() -> None:
    x, baseline, target_var, returns, direction = _batch()
    model = BaselineResidualTCN(
        BaselineResidualTCNConfig(feature_count=x.shape[-1], horizon_count=baseline.shape[-1])
    )
    loss, breakdown = volatility_multitask_loss(
        model(torch.from_numpy(x), torch.from_numpy(baseline)),
        torch.from_numpy(baseline),
        torch.from_numpy(target_var),
        torch.from_numpy(returns),
        torch.from_numpy(direction),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in breakdown.values())
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert breakdown["variance_crps"] >= 0
    assert torch.isfinite(breakdown["volatility_selection"])


def test_robust_scaler_uses_only_supplied_training_rows() -> None:
    x, *_ = _batch(rows=20)
    training = x[:12]
    validation = x[12:].copy()
    first = RobustSequenceScaler.fit(training)
    validation *= 10_000.0
    second = RobustSequenceScaler.fit(training)
    np.testing.assert_array_equal(first.median, second.median)
    np.testing.assert_array_equal(first.iqr, second.iqr)
    assert np.max(np.abs(first.transform(validation))) <= first.clip


@pytest.mark.skipif(
    sys.version_info >= (3, 14),
    reason="PyTorch wheels are not available for this interpreter",
)
def test_tiny_cpu_training_returns_bounded_model_and_train_only_scaler() -> None:
    x, baseline, target_var, returns, direction = _batch(rows=32, window=16, features=4)
    result = train_baseline_residual_tcn(
        train_features=x[:24],
        train_baseline_variance=baseline[:24],
        train_realized_variance=target_var[:24],
        train_cumulative_returns=returns[:24],
        train_direction_classes=direction[:24],
        validation_features=x[24:],
        validation_baseline_variance=baseline[24:],
        validation_realized_variance=target_var[24:],
        validation_cumulative_returns=returns[24:],
        validation_direction_classes=direction[24:],
        model_config=BaselineResidualTCNConfig(
            feature_count=4,
            horizon_count=3,
            channels=8,
            dilations=(1, 2),
            dropout=0.0,
        ),
        training_config=TorchTrainingConfig(
            maximum_epochs=2,
            patience=2,
            batch_size=8,
            use_amp=False,
        ),
        seed=5,
        device="cpu",
    )
    assert result.best_epoch in (1, 2)
    assert len(result.history) == 2
    assert result.parameter_count < 20_000
    assert result.device == "cpu"
    assert result.duration_seconds >= 0
    np.testing.assert_allclose(
        result.scaler.median,
        RobustSequenceScaler.fit(x[:24]).median,
    )


def test_news_training_uses_a_separate_train_only_scaler() -> None:
    x, baseline, target_var, returns, direction = _batch(rows=24, window=16, features=4)
    rng = np.random.default_rng(81)
    news = rng.normal(size=(24, 3)).astype(np.float32)
    result = train_baseline_residual_tcn(
        train_features=x[:18],
        train_baseline_variance=baseline[:18],
        train_realized_variance=target_var[:18],
        train_cumulative_returns=returns[:18],
        train_direction_classes=direction[:18],
        validation_features=x[18:],
        validation_baseline_variance=baseline[18:],
        validation_realized_variance=target_var[18:],
        validation_cumulative_returns=returns[18:],
        validation_direction_classes=direction[18:],
        train_news_features=news[:18],
        validation_news_features=news[18:] * 1000,
        model_config=BaselineResidualTCNConfig(
            feature_count=4,
            horizon_count=3,
            news_feature_count=3,
            channels=8,
            dilations=(1,),
            dropout=0.0,
        ),
        training_config=TorchTrainingConfig(
            maximum_epochs=1,
            patience=1,
            batch_size=8,
            use_amp=False,
        ),
        seed=6,
        device="cpu",
    )
    assert result.news_scaler is not None
    np.testing.assert_allclose(
        result.news_scaler.median,
        RobustSequenceScaler.fit(news[:18]).median,
    )
