"""Tests for GPU harness status, neural architectures, and real training execution."""

from __future__ import annotations

import io

import numpy as np
import torch

from research.volatility_forecasting.gpu_harness_v10 import (
    GRUVolatilityModel,
    LSTMVolatilityModel,
    PatchTSTVolatilityModel,
    TCNVolatilityModel,
    TrainingExecutionConfig,
    check_gpu_runtime,
    cleanup_gpu_memory,
    train_candidate_fold,
)


def test_gpu_runtime_check_and_cleanup_run_safely() -> None:
    status = check_gpu_runtime()
    assert isinstance(status.cuda_available, bool)
    assert isinstance(status.device_name, str)
    cleanup_gpu_memory()


def test_neural_architectures_forward_pass_and_positivity() -> None:
    # (batch, seq_len, in_features)
    x = torch.randn(8, 60, 26)

    # 1. TCN
    tcn = TCNVolatilityModel(in_features=26, num_channels=[16, 32])
    out_tcn = tcn(x)
    assert out_tcn.shape == (8, 1)
    assert (out_tcn > 0).all()

    # 2. LSTM
    lstm = LSTMVolatilityModel(in_features=26, hidden_dim=16, num_layers=2)
    out_lstm = lstm(x)
    assert out_lstm.shape == (8, 1)
    assert (out_lstm > 0).all()

    # 3. GRU
    gru = GRUVolatilityModel(in_features=26, hidden_dim=16, num_layers=2)
    out_gru = gru(x)
    assert out_gru.shape == (8, 1)
    assert (out_gru > 0).all()

    # 4. PatchTST
    patch = PatchTSTVolatilityModel(in_features=26, patch_len=8, stride=4, d_model=16)
    out_patch = patch(x)
    assert out_patch.shape == (8, 1)
    assert (out_patch > 0).all()


def test_train_candidate_fold_serializes_real_weights_for_all_families() -> None:
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(30, 20, 26))
    y_train = np.maximum(0.0004 + 0.0001 * X_train[:, -1, 0], 1e-6)
    X_val = rng.normal(size=(10, 20, 26))
    y_val = np.maximum(0.0004 + 0.0001 * X_val[:, -1, 0], 1e-6)

    for fam in ["ridge", "elasticnet", "tcn", "lstm", "gru", "patch_transformer"]:
        cfg = TrainingExecutionConfig(candidate_family=fam, horizon=1, fold_idx=0, max_epochs=3)
        res = train_candidate_fold(cfg, X_train, y_train, X_val, y_val)
        assert res["status"] == "success"
        assert res["val_qlike"] > 0.0
        assert isinstance(res["weights_bytes"], bytes)
        assert len(res["weights_bytes"]) > 10

        # For neural families, verify torch.load can deserialize state dict
        if fam in ("tcn", "lstm", "gru", "patch_transformer"):
            buf = io.BytesIO(res["weights_bytes"])
            state = torch.load(buf, weights_only=True)
            assert isinstance(state, dict)
