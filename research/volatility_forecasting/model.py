"""Compact baseline-residual TCN for probabilistic volatility forecasting.

The module is offline-only and intentionally imports PyTorch. Production uses
an exported, signed ONNX graph and does not import this training code.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class BaselineResidualTCNConfig:
    feature_count: int
    horizon_count: int
    channels: int = 48
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16)
    kernel_size: int = 3
    dropout: float = 0.15
    maximum_log_variance_correction: float = 1.5
    maximum_mean_standard_deviations: float = 0.50

    def __post_init__(self) -> None:
        if self.feature_count < 1 or self.horizon_count < 1:
            raise ValueError("feature_count and horizon_count must be positive")
        if self.channels < 4:
            raise ValueError("channels must be at least four")
        if not self.dilations or any(value < 1 for value in self.dilations):
            raise ValueError("dilations must contain positive integers")
        if self.kernel_size < 2:
            raise ValueError("kernel_size must be at least two")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True)
class VolatilityLossWeights:
    qlike: float = 0.60
    variance_crps: float = 0.25
    return_location: float = 0.05
    direction: float = 0.05
    baseline_regularization: float = 0.05

    def __post_init__(self) -> None:
        values = (
            self.qlike,
            self.variance_crps,
            self.return_location,
            self.direction,
            self.baseline_regularization,
        )
        if any(value < 0 for value in values) or not np.isclose(sum(values), 1.0):
            raise ValueError("loss weights must be non-negative and sum to one")


@dataclass(frozen=True)
class TorchTrainingConfig:
    maximum_epochs: int = 60
    patience: int = 8
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    minimum_delta: float = 1e-4
    use_amp: bool = True
    num_workers: int = 0

    def __post_init__(self) -> None:
        if self.maximum_epochs < 1 or self.patience < 1 or self.batch_size < 1:
            raise ValueError("epoch, patience, and batch settings must be positive")


@dataclass(frozen=True)
class RobustSequenceScaler:
    median: np.ndarray
    iqr: np.ndarray
    clip: float = 10.0

    @classmethod
    def fit(cls, features: np.ndarray, *, clip: float = 10.0) -> RobustSequenceScaler:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 3 or len(values) == 0:
            raise ValueError("scaler requires non-empty [rows, window, features] input")
        flattened = values.reshape(-1, values.shape[-1])
        median = np.median(flattened, axis=0)
        q25 = np.percentile(flattened, 25, axis=0)
        q75 = np.percentile(flattened, 75, axis=0)
        iqr = np.maximum(q75 - q25, 1e-8)
        if not np.isfinite(median).all() or not np.isfinite(iqr).all():
            raise ValueError("scaler statistics are not finite")
        return cls(median=median, iqr=iqr, clip=float(clip))

    def transform(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.shape[-1] != len(self.median):
            raise ValueError("feature dimension does not match fitted scaler")
        scaled = (values - self.median) / self.iqr
        return np.clip(scaled, -self.clip, self.clip).astype(np.float32)

    def to_dict(self) -> dict[str, object]:
        return {
            "median": self.median.tolist(),
            "iqr": self.iqr.tolist(),
            "clip": self.clip,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> RobustSequenceScaler:
        return cls(
            median=np.asarray(payload["median"], dtype=np.float64),
            iqr=np.asarray(payload["iqr"], dtype=np.float64),
            clip=float(payload.get("clip", 10.0)),
        )


class CausalConv1d(nn.Conv1d):
    """Left-padded convolution whose output at t cannot read rows after t."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, padding=0, **kwargs)
        self.left_padding = (self.kernel_size[0] - 1) * self.dilation[0]

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(values, (self.left_padding, 0)))


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.conv2 = CausalConv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = self.dropout(F.silu(self.norm1(self.conv1(values))))
        values = self.dropout(F.silu(self.norm2(self.conv2(values))))
        return F.silu(values + residual)


class BaselineResidualTCN(nn.Module):
    """Shared causal encoder with variance, return, and direction heads."""

    def __init__(self, config: BaselineResidualTCNConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Conv1d(config.feature_count, config.channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                ResidualTemporalBlock(
                    config.channels,
                    config.kernel_size,
                    dilation,
                    config.dropout,
                )
                for dilation in config.dilations
            ]
        )
        self.final_norm = nn.LayerNorm(config.channels)
        self.log_variance_residual_head = nn.Linear(config.channels, config.horizon_count)
        self.return_location_head = nn.Linear(config.channels, config.horizon_count)
        self.direction_head = nn.Linear(config.channels, config.horizon_count * 3)
        self._initialize_baseline_heads()

    def _initialize_baseline_heads(self) -> None:
        # Zero variance/mean heads make the initial model exactly match the
        # HAR variance and zero-return baselines, avoiding destructive random
        # endpoint moves before evidence is learned.
        nn.init.zeros_(self.log_variance_residual_head.weight)
        nn.init.zeros_(self.log_variance_residual_head.bias)
        nn.init.zeros_(self.return_location_head.weight)
        nn.init.zeros_(self.return_location_head.bias)
        nn.init.zeros_(self.direction_head.bias)

    def forward(
        self,
        features: torch.Tensor,
        baseline_variance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, window, features]")
        if baseline_variance.ndim != 2:
            raise ValueError("baseline_variance must have shape [batch, horizons]")
        values = self.input_projection(features.transpose(1, 2))
        for block in self.blocks:
            values = block(values)
        encoded = self.final_norm(values[:, :, -1])

        log_residual = self.config.maximum_log_variance_correction * torch.tanh(
            self.log_variance_residual_head(encoded)
        )
        safe_baseline = torch.clamp(baseline_variance, min=1e-12, max=1e2)
        forecast_variance = safe_baseline * torch.exp(log_residual)
        forecast_variance = torch.clamp(forecast_variance, min=1e-12, max=1e2)

        standardized_mean = self.config.maximum_mean_standard_deviations * torch.tanh(
            self.return_location_head(encoded)
        )
        return_location = standardized_mean * torch.sqrt(forecast_variance)
        direction_logits = self.direction_head(encoded).reshape(
            -1,
            self.config.horizon_count,
            3,
        )
        return forecast_variance, return_location, direction_logits, log_residual


def volatility_multitask_loss(
    prediction: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    baseline_variance: torch.Tensor,
    realized_variance: torch.Tensor,
    cumulative_returns: torch.Tensor,
    direction_classes: torch.Tensor,
    *,
    weights: VolatilityLossWeights | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """QLIKE-first objective with baseline shrinkage and auxiliary tasks."""
    loss_weights = weights or VolatilityLossWeights()
    forecast_variance, return_location, direction_logits, log_residual = prediction
    safe_target = torch.clamp(realized_variance, min=1e-12, max=1e2)
    safe_forecast = torch.clamp(forecast_variance, min=1e-12, max=1e2)
    ratio = safe_target / safe_forecast
    qlike = torch.mean(ratio - torch.log(ratio) - 1.0)

    safe_baseline = torch.clamp(baseline_variance, min=1e-12, max=1e2)
    sigma = torch.sqrt(safe_forecast)
    standardized_return = cumulative_returns / sigma
    density = torch.exp(-0.5 * torch.square(standardized_return)) / math.sqrt(2.0 * math.pi)
    distribution = 0.5 * (1.0 + torch.erf(standardized_return / math.sqrt(2.0)))
    crps = sigma * (
        standardized_return * (2.0 * distribution - 1.0) + 2.0 * density - 1.0 / math.sqrt(math.pi)
    )
    # Normalize by the matched baseline scale so long horizons and high-vol
    # assets cannot dominate the proper-score objective merely by magnitude.
    variance_crps = torch.mean(crps / torch.sqrt(safe_baseline))

    detached_scale = torch.sqrt(safe_forecast.detach())
    return_location_loss = F.smooth_l1_loss(
        return_location / detached_scale,
        cumulative_returns / detached_scale,
    )
    direction = F.cross_entropy(
        direction_logits.reshape(-1, 3),
        direction_classes.reshape(-1),
    )
    variance_regularization = torch.mean(torch.square(log_residual))
    baseline_regularization = variance_regularization + torch.mean(
        torch.square(return_location) / safe_forecast
    )
    volatility_selection = (
        loss_weights.qlike * qlike
        + loss_weights.variance_crps * variance_crps
        + loss_weights.baseline_regularization * variance_regularization
    )
    total = (
        loss_weights.qlike * qlike
        + loss_weights.variance_crps * variance_crps
        + loss_weights.return_location * return_location_loss
        + loss_weights.direction * direction
        + loss_weights.baseline_regularization * baseline_regularization
    )
    return total, {
        "total": total.detach(),
        "volatility_selection": volatility_selection.detach(),
        "qlike": qlike.detach(),
        "variance_crps": variance_crps.detach(),
        "return_location": return_location_loss.detach(),
        "direction_cross_entropy": direction.detach(),
        "baseline_regularization": baseline_regularization.detach(),
    }


@dataclass(frozen=True)
class TrainingResult:
    model: BaselineResidualTCN
    scaler: RobustSequenceScaler
    best_epoch: int
    history: tuple[dict[str, float], ...]
    device: str
    duration_seconds: float
    parameter_count: int


def _tensor_dataset(
    features: np.ndarray,
    baseline_variance: np.ndarray,
    realized_variance: np.ndarray,
    cumulative_returns: np.ndarray,
    direction_classes: np.ndarray,
) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(np.asarray(features, dtype=np.float32)),
        torch.from_numpy(np.asarray(baseline_variance, dtype=np.float32)),
        torch.from_numpy(np.asarray(realized_variance, dtype=np.float32)),
        torch.from_numpy(np.asarray(cumulative_returns, dtype=np.float32)),
        torch.from_numpy(np.asarray(direction_classes, dtype=np.int64)),
    )


def _evaluate_loader(
    model: BaselineResidualTCN,
    loader: DataLoader,
    device: torch.device,
    weights: VolatilityLossWeights,
) -> dict[str, float]:
    totals: dict[str, float] = defaultdict_float()
    rows = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x, baseline, target_var, target_return, direction = [
                value.to(device) for value in batch
            ]
            _, breakdown = volatility_multitask_loss(
                model(x, baseline),
                baseline,
                target_var,
                target_return,
                direction,
                weights=weights,
            )
            batch_rows = len(x)
            rows += batch_rows
            for name, value in breakdown.items():
                totals[name] += float(value.cpu()) * batch_rows
    return {name: value / max(rows, 1) for name, value in totals.items()}


def defaultdict_float() -> dict[str, float]:
    return {
        "total": 0.0,
        "volatility_selection": 0.0,
        "qlike": 0.0,
        "variance_crps": 0.0,
        "return_location": 0.0,
        "direction_cross_entropy": 0.0,
        "baseline_regularization": 0.0,
    }


def train_baseline_residual_tcn(
    *,
    train_features: np.ndarray,
    train_baseline_variance: np.ndarray,
    train_realized_variance: np.ndarray,
    train_cumulative_returns: np.ndarray,
    train_direction_classes: np.ndarray,
    validation_features: np.ndarray,
    validation_baseline_variance: np.ndarray,
    validation_realized_variance: np.ndarray,
    validation_cumulative_returns: np.ndarray,
    validation_direction_classes: np.ndarray,
    model_config: BaselineResidualTCNConfig,
    training_config: TorchTrainingConfig | None = None,
    loss_weights: VolatilityLossWeights | None = None,
    seed: int = 42,
    device: str | None = None,
) -> TrainingResult:
    """Fit one fold with train-only scaling, AMP, clipping, and early stop."""
    settings = training_config or TorchTrainingConfig()
    weights = loss_weights or VolatilityLossWeights()
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = bool(settings.use_amp and selected_device.type == "cuda")

    scaler = RobustSequenceScaler.fit(train_features)
    train_x = scaler.transform(train_features)
    validation_x = scaler.transform(validation_features)
    train_dataset = _tensor_dataset(
        train_x,
        train_baseline_variance,
        train_realized_variance,
        train_cumulative_returns,
        train_direction_classes,
    )
    validation_dataset = _tensor_dataset(
        validation_x,
        validation_baseline_variance,
        validation_realized_variance,
        validation_cumulative_returns,
        validation_direction_classes,
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=settings.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=settings.num_workers,
        pin_memory=selected_device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
        pin_memory=selected_device.type == "cuda",
    )

    model = BaselineResidualTCN(model_config).to(selected_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    amp_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    patience_used = 0
    history: list[dict[str, float]] = []
    started = time.perf_counter()

    for epoch in range(1, settings.maximum_epochs + 1):
        model.train()
        for batch in train_loader:
            x, baseline, target_var, target_return, direction = [
                value.to(selected_device, non_blocking=True) for value in batch
            ]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=selected_device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                loss, _ = volatility_multitask_loss(
                    model(x, baseline),
                    baseline,
                    target_var,
                    target_return,
                    direction,
                    weights=weights,
                )
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip_norm)
            amp_scaler.step(optimizer)
            amp_scaler.update()

        validation_metrics = _evaluate_loader(model, validation_loader, selected_device, weights)
        history.append({"epoch": float(epoch), **validation_metrics})
        validation_loss = validation_metrics["volatility_selection"]
        if validation_loss < best_loss - settings.minimum_delta:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_used = 0
        else:
            patience_used += 1
            if patience_used >= settings.patience:
                break

    model.load_state_dict(best_state)
    duration = time.perf_counter() - started
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return TrainingResult(
        model=model,
        scaler=scaler,
        best_epoch=best_epoch,
        history=tuple(history),
        device=str(selected_device),
        duration_seconds=float(duration),
        parameter_count=int(parameter_count),
    )
