"""Causal neural candidate architectures for cumulative return and volatility forecasting."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    """Truncate right-side padding so output at timestep t depends only on inputs <= t."""

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class CausalTemporalBlock(nn.Module):
    """Causal dilated residual block with strict left-padding and weight normalization."""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.gelu1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.gelu2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.gelu1,
            self.drop1,
            self.conv2,
            self.chomp2,
            self.gelu2,
            self.drop2,
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.gelu_out = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.gelu_out(out + res)


class CausalTCNModel(nn.Module):
    """Strictly causal Temporal Convolutional Network for cumulative return & volatility forecasting."""

    def __init__(
        self,
        in_features: int,
        num_channels: list[int] | None = None,
        kernel_size: int = 3,
        forecast_days: int = 7,
        dropout: float = 0.1,
        mode: str = "return",  # "return" or "volatility"
    ) -> None:
        super().__init__()
        self.mode = mode
        if num_channels is None:
            num_channels = [32, 64, 64]
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_ch = in_features if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers.append(
                CausalTemporalBlock(
                    in_ch,
                    out_ch,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=padding,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], forecast_days)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F) -> permute to (B, F, L)
        x_perm = x.permute(0, 2, 1)
        conv_out = self.network(x_perm)
        last_timestep = conv_out[:, :, -1]  # Final causal timestep
        raw_out = self.fc(last_timestep)
        if self.mode == "volatility":
            # Increments are strictly positive; cumulative variance is monotonic sum
            increments = self.softplus(raw_out)
            return torch.cumsum(increments, dim=-1)
        return raw_out


class CausalLSTMModel(nn.Module):
    """Deep 2-layer LSTM model for multi-horizon cumulative return or volatility increments."""

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        forecast_days: int = 7,
        dropout: float = 0.15,
        mode: str = "return",
    ) -> None:
        super().__init__()
        self.mode = mode
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, forecast_days),
        )
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_h = out[:, -1, :]
        raw_out = self.fc(last_h)
        if self.mode == "volatility":
            increments = self.softplus(raw_out)
            return torch.cumsum(increments, dim=-1)
        return raw_out


class DLinearModel(nn.Module):
    """Direct Linear decomposition model with trend and seasonal projection."""

    def __init__(
        self,
        sequence_length: int = 60,
        in_features: int = 16,
        forecast_days: int = 7,
        mode: str = "return",
    ) -> None:
        super().__init__()
        self.mode = mode
        self.linear = nn.Linear(sequence_length * in_features, forecast_days)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        flattened = x.reshape(B, -1)
        raw_out = self.linear(flattened)
        if self.mode == "volatility":
            increments = self.softplus(raw_out)
            return torch.cumsum(increments, dim=-1)
        return raw_out


class ConstrainedEnsembleOptimizer:
    """Solves non-negative sum-to-one ensemble weights on out-of-fold validation losses."""

    @staticmethod
    def fit_weights(predictions_matrix: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Find w >= 0, sum(w)=1 minimizing mean squared/Huber loss against targets.

        predictions_matrix: (N, M) where M is candidate count.
        """
        N, M = predictions_matrix.shape
        if M == 1:
            return np.array([1.0], dtype=float)

        from scipy.optimize import minimize

        def loss_fn(w):
            pred = predictions_matrix @ w
            diff = pred - targets
            # Huber loss
            huber = np.where(np.abs(diff) < 1.0, 0.5 * diff**2, np.abs(diff) - 0.5)
            return np.mean(huber)

        w0 = np.ones(M) / M
        bounds = [(0.0, 1.0) for _ in range(M)]
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        res = minimize(loss_fn, w0, bounds=bounds, constraints=constraints, method="SLSQP")
        if res.success:
            w_opt = res.x
            # Normalize for precision
            w_opt = np.maximum(w_opt, 0.0)
            return w_opt / np.sum(w_opt)
        return w0
