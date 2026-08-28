from __future__ import annotations

import numpy as np
import torch

from research.volatility_forecasting.architecture_ablation import (
    classify_regimes,
    compute_forecast_metrics,
)
from research.volatility_forecasting.data import VolatilityPanelExamples
from research.volatility_forecasting.model import (
    BaselineResidualTCN,
    BaselineResidualTCNConfig,
)


def test_classify_regimes_causality():
    """Verify regime classification uses trailing Vol_C2C_20 correctly."""
    n = 300
    features = np.zeros((n, 60, 26))
    # Set Vol_C2C_20 (col 10) to low vol (0.005 -> annual ~0.08), mid vol (0.012 -> annual ~0.19), high vol (0.030 -> annual ~0.47)
    features[:100, -1, 10] = 0.005
    features[100:200, -1, 10] = 0.012
    features[200:, -1, 10] = 0.030

    feature_names = tuple(f"f_{i}" for i in range(26))
    names_list = list(feature_names)
    names_list[10] = "Vol_C2C_20"
    feature_names = tuple(names_list)

    examples = VolatilityPanelExamples(
        features=features,
        baseline_variance=np.ones((n, 6)) * 0.04,
        realized_variance=np.ones((n, 6)) * 0.04,
        cumulative_returns=np.zeros((n, 6)),
        direction_classes=np.zeros((n, 6), dtype=int),
        origin_dates=np.array(
            [np.datetime64("2020-01-01") + np.timedelta64(i, "D") for i in range(n)]
        ),
        tickers=tuple("AAPL" for _ in range(n)),
        origin_closes=np.ones(n),
        feature_names=feature_names,
        horizons=(1, 3, 5, 7, 14, 30),
    )

    indices = np.arange(n)
    regimes = classify_regimes(examples, indices)
    assert len(regimes) == n
    assert (regimes[:100] == 0).all()  # LOW
    assert (regimes[100:200] == 1).all()  # NORMAL
    assert (regimes[200:] == 2).all()  # HIGH


def test_compute_forecast_metrics_finite_and_bounded():
    """Verify metric calculations for QLIKE, RMSE, MAE, R2, MedAE."""
    y_true = np.array([0.04, 0.09, 0.16, 0.25])
    y_pred = np.array([0.05, 0.08, 0.15, 0.26])
    y_base = np.array([0.06, 0.10, 0.18, 0.28])

    metrics = compute_forecast_metrics(y_true, y_pred, y_base)
    assert "qlike" in metrics
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics
    assert "medae" in metrics
    assert "ratio_to_baseline" in metrics
    assert metrics["qlike"] > 0
    assert metrics["rmse"] > 0
    assert metrics["ratio_to_baseline"] < 1.0  # y_pred is closer to y_true than y_base


def test_v9_lstm_vs_gru_encoder_shapes():
    """Verify BaselineResidualTCNConfig dynamically routes to LSTM or GRU."""
    config_lstm = BaselineResidualTCNConfig(
        feature_count=26,
        horizon_count=6,
        window_size=60,
        encoder_family="lstm",
        channels=32,
        lstm_layers=2,
        lstm_hidden=32,
    )
    model_lstm = BaselineResidualTCN(config_lstm)

    config_gru = BaselineResidualTCNConfig(
        feature_count=26,
        horizon_count=6,
        window_size=60,
        encoder_family="gru",
        channels=32,
        lstm_layers=2,
        lstm_hidden=32,
    )
    model_gru = BaselineResidualTCN(config_gru)

    x = torch.randn(8, 60, 26)
    b = torch.ones(8, 6) * 0.05

    out_lstm = model_lstm(x, b)
    out_gru = model_gru(x, b)

    assert out_lstm[0].shape == (8, 6)
    assert out_gru[0].shape == (8, 6)
    assert out_lstm[3].shape == (8, 6)
    assert out_gru[3].shape == (8, 6)
