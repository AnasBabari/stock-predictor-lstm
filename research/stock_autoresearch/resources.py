"""Runtime resource checks for local experiments."""

from __future__ import annotations

import sys
from dataclasses import dataclass

from .config import RuntimeBudget


@dataclass(frozen=True)
class ResourceSample:
    peak_vram_mb: int
    warning: bool
    exceeded: bool


def sample_cuda_memory(budget: RuntimeBudget) -> ResourceSample:
    # Current PyTorch CUDA wheels are not safe to import on Python 3.14.
    if sys.version_info >= (3, 14):
        return ResourceSample(0, False, False)
    try:
        import torch
    except (ImportError, OSError, RuntimeError):
        return ResourceSample(0, False, False)
    if not torch.cuda.is_available():
        return ResourceSample(0, False, False)
    peak = int(torch.cuda.max_memory_allocated() / (1024 * 1024))
    return ResourceSample(peak, peak >= budget.vram_warning_mb, peak >= budget.vram_kill_mb)


def reset_cuda_memory() -> None:
    if sys.version_info >= (3, 14):
        return
    try:
        import torch
    except (ImportError, OSError, RuntimeError):
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
