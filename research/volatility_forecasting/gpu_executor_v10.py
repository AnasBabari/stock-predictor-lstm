"""Bounded, reproducible PyTorch CUDA candidate execution engine for RTX 2060."""

from __future__ import annotations

import contextlib
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class HardwareRuntimeManifest:
    device_name: str
    cuda_available: bool
    cuda_version: str | None
    device_count: int
    peak_vram_mb: float
    duration_seconds: float
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateExecutionResult:
    family: str
    target_contract: str
    horizon: int
    seed: int
    success: bool
    checkpoint_path: str | None
    val_loss: float
    error_message: str | None
    hardware_manifest: HardwareRuntimeManifest

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["hardware_manifest"] = self.hardware_manifest.to_dict()
        return d


class BoundedCUDAExecutor:
    """Executes PyTorch models with strict resource ceilings, AMP, and state serialization."""

    def __init__(
        self,
        device: str | None = None,
        max_vram_mb: float = 5500.0,  # RTX 2060 6GB ceiling
        timeout_seconds: float = 300.0,
        use_amp: bool = True,
    ) -> None:
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.max_vram_mb = max_vram_mb
        self.timeout_seconds = timeout_seconds
        self.use_amp = use_amp and (self.device.type == "cuda")

    def get_hardware_info(
        self, peak_vram_mb: float = 0.0, duration: float = 0.0, seed: int = 42
    ) -> HardwareRuntimeManifest:
        cuda_avail = torch.cuda.is_available()
        return HardwareRuntimeManifest(
            device_name=torch.cuda.get_device_name(0) if cuda_avail else "CPU",
            cuda_available=cuda_avail,
            cuda_version=torch.version.cuda if cuda_avail else None,
            device_count=torch.cuda.device_count() if cuda_avail else 0,
            peak_vram_mb=round(peak_vram_mb, 2),
            duration_seconds=round(duration, 3),
            seed=seed,
        )

    def train_and_serialize(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_x: torch.Tensor,
        val_y: torch.Tensor,
        save_dir: Path,
        family: str,
        target_contract: str,
        horizon: int,
        epochs: int = 40,
        lr: float = 1e-3,
        seed: int = 42,
    ) -> CandidateExecutionResult:
        torch.manual_seed(seed)
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
            with contextlib.suppress(Exception):
                torch.cuda.reset_peak_memory_stats()

        save_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = save_dir / f"checkpoint_{family}_h{horizon}_s{seed}.pt"

        model = model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        loss_fn = nn.SmoothL1Loss()  # Huber loss
        device_type = self.device.type
        scaler = torch.amp.GradScaler(device_type, enabled=self.use_amp)

        best_loss = float("inf")
        best_state = None

        t0 = time.time()
        val_x_dev = val_x.to(self.device)
        val_y_dev = val_y.to(self.device)

        try:
            for _epoch in range(epochs):
                if time.time() - t0 > self.timeout_seconds:
                    raise TimeoutError(
                        f"Training exceeded maximum wall-clock timeout of {self.timeout_seconds}s."
                    )

                model.train()
                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast(device_type, enabled=self.use_amp):
                        preds = model(batch_x)
                        loss = loss_fn(preds, batch_y)

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()

                model.eval()
                with torch.no_grad():
                    val_preds = model(val_x_dev)
                    val_loss = float(loss_fn(val_preds, val_y_dev).item())
                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if best_state is not None:
                torch.save(best_state, checkpoint_path)

            peak_vram = 0.0
            if torch.cuda.is_available() and self.device.type == "cuda":
                with contextlib.suppress(Exception):
                    peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)

            duration = time.time() - t0
            manifest = self.get_hardware_info(peak_vram_mb=peak_vram, duration=duration, seed=seed)

            return CandidateExecutionResult(
                family=family,
                target_contract=target_contract,
                horizon=horizon,
                seed=seed,
                success=True,
                checkpoint_path=str(checkpoint_path),
                val_loss=best_loss,
                error_message=None,
                hardware_manifest=manifest,
            )

        except Exception as exc:
            duration = time.time() - t0
            manifest = self.get_hardware_info(peak_vram_mb=0.0, duration=duration, seed=seed)
            return CandidateExecutionResult(
                family=family,
                target_contract=target_contract,
                horizon=horizon,
                seed=seed,
                success=False,
                checkpoint_path=None,
                val_loss=float("inf"),
                error_message=str(exc),
                hardware_manifest=manifest,
            )
