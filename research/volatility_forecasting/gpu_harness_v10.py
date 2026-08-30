"""RTX GPU / CPU candidate training execution harness for StockLSTM V10.

Provides deterministic multi-family candidate training (HAR, Ridge, ElasticNet,
TCN, LSTM, GRU, PatchTST), 3D rolling temporal sequence generation, train-only
robust scaling, QLIKE optimization, and matched baseline evaluations.
"""

from __future__ import annotations

import gc
import io
import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from research.volatility_forecasting.baselines_v10 import (
    ElasticNetVolatilityBaseline,
    HARRVBaseline,
    RidgeVolatilityBaseline,
)

logger = logging.getLogger("gpu_harness_v10")


class TrainOnlyRobustScaler:
    """RobustScaler fitted exclusively on training information sets to prevent data leakage."""

    def __init__(self, clip_min: float = -10.0, clip_max: float = 10.0) -> None:
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.feature_count_: int = 0

    def fit(self, X: np.ndarray) -> TrainOnlyRobustScaler:
        """Compute median and IQR strictly along feature axes."""
        if X.ndim == 3:
            X_flat = X.reshape(-1, X.shape[-1])
        elif X.ndim == 2:
            X_flat = X
        else:
            raise ValueError(f"Expected 2D or 3D array, got {X.ndim}D")

        self.feature_count_ = X_flat.shape[-1]
        self.center_ = np.median(X_flat, axis=0)
        q25 = np.percentile(X_flat, 25, axis=0)
        q75 = np.percentile(X_flat, 75, axis=0)
        iqr = q75 - q25

        # Fallback to std or 1.0 for zero IQR features
        std = np.std(X_flat, axis=0)
        scale = np.where(iqr > 1e-6, iqr, np.where(std > 1e-6, std, 1.0))
        self.scale_ = scale
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply train-derived center and scale; clip outliers."""
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("Scaler must be fitted before transform.")
        X_norm = (X - self.center_) / self.scale_
        return np.clip(X_norm, self.clip_min, self.clip_max)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def to_dict(self) -> dict[str, Any]:
        return {
            "center": self.center_.tolist() if self.center_ is not None else [],
            "scale": self.scale_.tolist() if self.scale_ is not None else [],
            "feature_count": self.feature_count_,
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainOnlyRobustScaler:
        scaler = cls(clip_min=data.get("clip_min", -10.0), clip_max=data.get("clip_max", 10.0))
        scaler.center_ = np.array(data["center"], dtype=float) if data.get("center") else None
        scaler.scale_ = np.array(data["scale"], dtype=float) if data.get("scale") else None
        scaler.feature_count_ = int(data.get("feature_count", 0))
        return scaler


def build_temporal_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    sequence_length: int = 60,
    min_history: int = 20,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Construct 3D rolling windows (N, L, F) independently per security without future lookahead."""
    seq_list = []
    target_list = []
    meta_rows = []

    # Sort strictly by (SecurityID, Date)
    df_sorted = df.sort_values(by=["SecurityID", "Date"]).copy()

    for sec_id, group in df_sorted.groupby("SecurityID", sort=False):
        n_rows = len(group)
        if n_rows < sequence_length:
            continue

        feat_mat = group[feature_cols].to_numpy(dtype=float)
        targ_vec = group[target_col].to_numpy(dtype=float)
        date_vec = group["Date"].to_numpy(dtype=str)

        for t in range(sequence_length - 1, n_rows):
            window = feat_mat[t - sequence_length + 1 : t + 1, :]
            target_val = targ_vec[t]
            date_val = str(date_vec[t])

            seq_list.append(window)
            target_list.append(target_val)
            meta_rows.append({"SecurityID": sec_id, "Date": date_val, "OriginIndex": t})

    if not seq_list:
        X_seq = np.empty((0, sequence_length, len(feature_cols)), dtype=float)
        y_seq = np.empty((0,), dtype=float)
        meta_df = pd.DataFrame(columns=["SecurityID", "Date", "OriginIndex"])
    else:
        X_seq = np.asarray(seq_list, dtype=float)
        y_seq = np.asarray(target_list, dtype=float)
        meta_df = pd.DataFrame(meta_rows)

    return X_seq, y_seq, meta_df


class Chomp1d(nn.Module):
    """Causal padding truncation to ensure output at t uses only inputs <= t."""

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Causal dilated residual convolutional block."""

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
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNVolatilityModel(nn.Module):
    """Temporal Convolutional Network for multi-horizon volatility forecasting."""

    def __init__(
        self,
        in_features: int,
        num_channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_channels is None:
            num_channels = [32, 64]
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_ch = in_features if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers.append(
                TemporalBlock(
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
        self.head = nn.Linear(num_channels[-1], 1)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.as_tensor(x, dtype=torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x_perm = x.permute(0, 2, 1)
        conv_out = self.network(x_perm)
        last_timestep = conv_out[:, :, -1]
        out = self.softplus(self.head(last_timestep))
        return out.squeeze(-1) if out.ndim > 1 else out


class LSTMVolatilityModel(nn.Module):
    """Multi-layer LSTM model with strictly positive Softplus output."""

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, 1)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.as_tensor(x, dtype=torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        return self.softplus(self.head(last_out)).squeeze(-1)


class GRUVolatilityModel(nn.Module):
    """Multi-layer GRU model with strictly positive Softplus output."""

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, 1)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.as_tensor(x, dtype=torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        out, _ = self.gru(x)
        last_out = out[:, -1, :]
        return self.softplus(self.head(last_out)).squeeze(-1)


class PatchTSTVolatilityModel(nn.Module):
    """Patch Time Series Transformer for volatility forecasting."""

    def __init__(
        self,
        in_features: int,
        patch_len: int = 8,
        stride: int = 4,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.patch_proj = nn.Linear(patch_len * in_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.as_tensor(x, dtype=torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        batch_size, seq_len, in_feat = x.shape
        if seq_len < self.patch_len:
            pad = torch.zeros((batch_size, self.patch_len - seq_len, in_feat), device=x.device)
            x = torch.cat([pad, x], dim=1)
            seq_len = self.patch_len

        patches = []
        for i in range(0, seq_len - self.patch_len + 1, self.stride):
            p = x[:, i : i + self.patch_len, :].reshape(batch_size, -1)
            patches.append(self.patch_proj(p))

        if not patches:
            p = x[:, -self.patch_len :, :].reshape(batch_size, -1)
            patches.append(self.patch_proj(p))

        patch_seq = torch.stack(patches, dim=1)
        encoded = self.transformer(patch_seq)
        last_patch = encoded[:, -1, :]
        return self.softplus(self.head(last_patch)).squeeze(-1)


def cleanup_gpu_memory() -> None:
    """Force garbage collection and CUDA cache clearing."""
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


@dataclass(frozen=True)
class TrainingExecutionConfig:
    candidate_family: str
    horizon: int
    fold_idx: int
    seed: int = 42
    max_epochs: int = 20
    early_stopping_patience: int = 5
    learning_rate: float = 1e-3
    gradient_clip_norm: float = 1.0
    use_amp: bool = False
    device: str = "cpu"
    max_vram_gb: float = 4.5
    timeout_seconds: int = 300

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def qlike_vector(pred_var: np.ndarray, actual_var: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.maximum(pred_var, eps)
    a = np.maximum(actual_var, eps)
    ratio = a / p
    return ratio - np.log(ratio) - 1.0


def qlike_loss(pred_var: np.ndarray, actual_var: np.ndarray, eps: float = 1e-12) -> float:
    vec = qlike_vector(pred_var, actual_var, eps=eps)
    return float(np.mean(vec))


def torch_qlike_loss(
    pred_var: torch.Tensor, actual_var: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    p = torch.clamp(pred_var, min=eps)
    a = torch.clamp(actual_var, min=eps)
    ratio = a / p
    return torch.mean(ratio - torch.log(ratio) - 1.0)


def train_candidate_fold(
    config: TrainingExecutionConfig,
    X_train_seq: np.ndarray,
    y_train: np.ndarray,
    X_val_seq: np.ndarray,
    y_val: np.ndarray,
) -> dict[str, Any]:
    """Train candidate fold on 3D sequence inputs with train-only scaling and matched baselines."""
    t0 = time.perf_counter()
    family = config.candidate_family.lower()

    # 1. Fit train-only robust scaler
    scaler = TrainOnlyRobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_seq)
    X_val_scaled = scaler.transform(X_val_seq)

    # 2. Compute matched HAR baseline predictions on validation set
    har = HARRVBaseline()
    har.fit(y_train)
    pred_base_val = (
        np.array([har.predict(y_train[-30:], config.horizon)] * len(y_val))
        if len(y_val) > 0
        else np.array([])
    )
    base_loss_vec = qlike_vector(pred_base_val, y_val) if len(y_val) > 0 else np.array([])
    base_qlike = float(np.mean(base_loss_vec)) if len(base_loss_vec) > 0 else 1.0

    X_train_flat = X_train_scaled[:, -1, :] if X_train_scaled.ndim == 3 else X_train_scaled
    X_val_flat = X_val_scaled[:, -1, :] if X_val_scaled.ndim == 3 else X_val_scaled

    baseline_params = {
        "har_beta_0": har.beta_0,
        "har_beta_d": har.beta_d,
        "har_beta_w": har.beta_w,
        "har_beta_m": har.beta_m,
    }

    # 3. Model Training
    if family == "har":
        pred_cand_val = pred_base_val
        status = "success"
        weights_bytes = json.dumps(baseline_params).encode("utf-8")

    elif family == "ridge":
        model = RidgeVolatilityBaseline(alpha=1.0)
        model.fit(X_train_flat, y_train)
        pred_cand_val = model.predict(X_val_flat)
        status = "success"
        weights_bytes = json.dumps(
            {
                "coef": model.model.coef_.tolist(),
                "intercept": float(model.model.intercept_),
            }
        ).encode("utf-8")

    elif family == "elasticnet":
        model = ElasticNetVolatilityBaseline(alpha=0.01, l1_ratio=0.5)
        model.fit(X_train_flat, y_train)
        pred_cand_val = model.predict(X_val_flat)
        status = "success"
        weights_bytes = json.dumps(
            {
                "coef": model.model.coef_.tolist(),
                "intercept": float(model.model.intercept_),
            }
        ).encode("utf-8")

    else:
        # PyTorch Neural Models
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        target_device = torch.device(
            "cuda:0" if torch.cuda.is_available() and config.device.startswith("cuda") else "cpu"
        )

        in_features = X_train_scaled.shape[-1]
        if family == "tcn":
            net = TCNVolatilityModel(in_features=in_features, num_channels=[32, 64])
        elif family == "lstm":
            net = LSTMVolatilityModel(in_features=in_features, hidden_dim=32, num_layers=2)
        elif family == "gru":
            net = GRUVolatilityModel(in_features=in_features, hidden_dim=32, num_layers=2)
        elif family in ("patch_transformer", "patchtst"):
            net = PatchTSTVolatilityModel(
                in_features=in_features, patch_len=8, stride=4, d_model=32
            )
        else:
            raise ValueError(f"Unknown candidate family: {config.candidate_family}")

        net.to(target_device)
        optimizer = optim.Adam(net.parameters(), lr=config.learning_rate)

        x_t = torch.tensor(X_train_scaled, dtype=torch.float32, device=target_device)
        y_t = torch.tensor(y_train, dtype=torch.float32, device=target_device)
        x_v = torch.tensor(X_val_scaled, dtype=torch.float32, device=target_device)
        y_v = torch.tensor(y_val, dtype=torch.float32, device=target_device)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for _epoch in range(config.max_epochs):
            net.train()
            optimizer.zero_grad()
            out = net(x_t)
            loss = torch_qlike_loss(out, y_t)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), config.gradient_clip_norm)
            optimizer.step()

            net.eval()
            with torch.no_grad():
                val_out = net(x_v)
                v_loss = torch_qlike_loss(val_out, y_v).item()

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.early_stopping_patience:
                    break

        if best_state is not None:
            net.load_state_dict(best_state)

        net.eval()
        with torch.no_grad():
            pred_cand_val = net(x_v).cpu().numpy()

        status = "success"
        buf = io.BytesIO()
        torch.save(best_state or net.state_dict(), buf)
        weights_bytes = buf.getvalue()

    elapsed = time.perf_counter() - t0
    cleanup_gpu_memory()

    cand_loss_vec = qlike_vector(pred_cand_val, y_val) if len(y_val) > 0 else np.array([])
    val_qlike = float(np.mean(cand_loss_vec)) if len(cand_loss_vec) > 0 else 1.0
    rel_qlike = val_qlike / max(base_qlike, 1e-12)

    return {
        "family": family,
        "horizon": config.horizon,
        "fold_idx": config.fold_idx,
        "seed": config.seed,
        "status": status,
        "val_qlike": float(val_qlike),
        "base_qlike": float(base_qlike),
        "relative_qlike": float(rel_qlike),
        "candidate_loss_vector": cand_loss_vec.tolist(),
        "baseline_loss_vector": base_loss_vec.tolist(),
        "candidate_predictions": pred_cand_val.tolist(),
        "baseline_predictions": pred_base_val.tolist(),
        "scaler_parameters": scaler.to_dict(),
        "baseline_parameters": baseline_params,
        "weights_bytes": weights_bytes,
        "wall_time_seconds": float(elapsed),
    }


@dataclass(frozen=True)
class HardwareRuntimeStatus:
    cuda_available: bool
    device_name: str
    vram_total_gb: float
    vram_allocated_gb: float
    vram_reserved_gb: float


def check_gpu_runtime() -> HardwareRuntimeStatus:
    """Query runtime GPU state safely."""
    try:
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            total_gb = props.total_memory / (1024**3)
            alloc_gb = torch.cuda.memory_allocated(0) / (1024**3)
            res_gb = torch.cuda.memory_reserved(0) / (1024**3)
            return HardwareRuntimeStatus(
                cuda_available=True,
                device_name=device_name,
                vram_total_gb=float(total_gb),
                vram_allocated_gb=float(alloc_gb),
                vram_reserved_gb=float(res_gb),
            )
    except Exception as exc:
        logger.debug("GPU query fallback: %s", exc)

    return HardwareRuntimeStatus(
        cuda_available=False,
        device_name="cpu",
        vram_total_gb=0.0,
        vram_allocated_gb=0.0,
        vram_reserved_gb=0.0,
    )
