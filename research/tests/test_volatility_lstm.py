"""Tests for BaselineResidualLSTM and recurrent encoders in volatility forecasting."""

from __future__ import annotations

import torch

from research.volatility_forecasting.model import (
    BaselineResidualLSTM,
    BaselineResidualLSTMConfig,
    BaselineResidualTCN,
    BaselineResidualTCNConfig,
    VolatilityLossWeights,
    volatility_multitask_loss,
)


def test_baseline_residual_lstm_initialization_and_forward() -> None:
    config = BaselineResidualLSTMConfig(
        feature_count=26,
        horizon_count=6,
        encoder_family="lstm",
        window_size=60,
        channels=48,
        lstm_layers=2,
        lstm_hidden=32,
    )
    model = BaselineResidualLSTM(config).eval()

    batch_size = 8
    features = torch.randn(batch_size, 60, 26)
    baseline_variance = torch.ones(batch_size, 6) * 0.0004

    var_pred, ret_loc, dir_logits, log_res = model(features, baseline_variance)

    assert var_pred.shape == (batch_size, 6)
    assert ret_loc.shape == (batch_size, 6)
    assert dir_logits.shape == (batch_size, 6, 3)
    assert log_res.shape == (batch_size, 6)

    # Initial zero heads should yield log_res == 0, var_pred == baseline_variance, ret_loc == 0
    assert torch.allclose(log_res, torch.zeros_like(log_res), atol=1e-5)
    assert torch.allclose(var_pred, baseline_variance, atol=1e-5)
    assert torch.allclose(ret_loc, torch.zeros_like(ret_loc), atol=1e-5)


def test_baseline_residual_gru_initialization_and_forward() -> None:
    config = BaselineResidualTCNConfig(
        feature_count=26,
        horizon_count=6,
        encoder_family="gru",
        window_size=60,
        channels=32,
        lstm_layers=1,
        lstm_hidden=24,
    )
    model = BaselineResidualTCN(config).eval()

    batch_size = 4
    features = torch.randn(batch_size, 60, 26)
    baseline_variance = torch.ones(batch_size, 6) * 0.0001

    var_pred, ret_loc, dir_logits, log_res = model(features, baseline_variance)

    assert var_pred.shape == (batch_size, 6)
    assert ret_loc.shape == (batch_size, 6)
    assert dir_logits.shape == (batch_size, 6, 3)
    assert log_res.shape == (batch_size, 6)


def test_multitask_loss_on_lstm() -> None:
    config = BaselineResidualLSTMConfig(
        feature_count=26,
        horizon_count=6,
        encoder_family="lstm",
        window_size=60,
    )
    model = BaselineResidualLSTM(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    weights = VolatilityLossWeights()

    batch_size = 16
    features = torch.randn(batch_size, 60, 26)
    baseline_variance = torch.full((batch_size, 6), 0.0004)
    realized_variance = torch.full((batch_size, 6), 0.0005)
    cumulative_returns = torch.zeros(batch_size, 6)
    direction_classes = torch.ones((batch_size, 6), dtype=torch.long)

    model.train()
    optimizer.zero_grad()
    var_pred, ret_loc, dir_logits, log_res = model(features, baseline_variance)

    loss, details = volatility_multitask_loss(
        prediction=(var_pred, ret_loc, dir_logits, log_res),
        baseline_variance=baseline_variance,
        realized_variance=realized_variance,
        cumulative_returns=cumulative_returns,
        direction_classes=direction_classes,
        weights=weights,
    )

    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    assert "qlike" in details
    assert "variance_crps" in details
