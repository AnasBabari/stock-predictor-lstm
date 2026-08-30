"""Tests for BoundedCUDAExecutor and hardware manifest recording."""

import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from research.volatility_forecasting.gpu_executor_v10 import BoundedCUDAExecutor


def test_bounded_cuda_executor_training_and_serialization():
    executor = BoundedCUDAExecutor()
    in_features = 4
    forecast_days = 3

    model = nn.Sequential(
        nn.Linear(20 * in_features, 16),
        nn.GELU(),
        nn.Linear(16, forecast_days),
    )

    # Flatten wrapper
    class FlatWrapper(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            return self.net(x.reshape(x.shape[0], -1))

    wrapped_model = FlatWrapper(model)

    # Synthetic data
    rng = np.random.default_rng(42)
    X_tr = torch.tensor(rng.normal(0, 1, size=(64, 20, in_features)), dtype=torch.float32)
    y_tr = torch.tensor(rng.normal(0, 0.02, size=(64, forecast_days)), dtype=torch.float32)
    X_val = torch.tensor(rng.normal(0, 1, size=(16, 20, in_features)), dtype=torch.float32)
    y_val = torch.tensor(rng.normal(0, 0.02, size=(16, forecast_days)), dtype=torch.float32)

    dataset = TensorDataset(X_tr, y_tr)
    loader = DataLoader(dataset, batch_size=16)

    with tempfile.TemporaryDirectory() as tmpdir:
        res = executor.train_and_serialize(
            model=wrapped_model,
            train_loader=loader,
            val_x=X_val,
            val_y=y_val,
            save_dir=Path(tmpdir),
            family="tcn",
            target_contract="price-return-distribution-v1",
            horizon=3,
            epochs=5,
            seed=42,
        )

        assert res.success is True
        assert res.checkpoint_path is not None
        assert Path(res.checkpoint_path).exists()
        assert res.val_loss < float("inf")
        assert res.hardware_manifest.seed == 42
        assert res.hardware_manifest.duration_seconds > 0.0

        # Reload checkpoint state dict
        loaded_state = torch.load(res.checkpoint_path, weights_only=True)
        wrapped_model.load_state_dict(loaded_state)
        assert len(loaded_state) > 0
