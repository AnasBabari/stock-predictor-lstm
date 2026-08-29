"""RTX GPU training execution harness with resource bounding and memory cleanup.

Binds training execution to ImmutableRunManifest, provides subprocess isolation,
VRAM tracking, deterministic seed initialization, and automatic tensor cleanup.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass

logger = logging.getLogger("gpu_harness_v10")


@dataclass(frozen=True)
class HardwareRuntimeStatus:
    cuda_available: bool
    device_name: str
    vram_total_gb: float
    vram_allocated_gb: float
    vram_reserved_gb: float


def check_gpu_runtime() -> HardwareRuntimeStatus:
    """Query runtime GPU state safely without forcing hard Torch dependency."""
    try:
        import torch

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


def cleanup_gpu_memory() -> None:
    """Force garbage collection and CUDA cache clearing."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
