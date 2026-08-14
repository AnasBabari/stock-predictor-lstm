"""Runtime resource checks for local experiments."""

from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import dataclass

try:
    import psutil
except ImportError:
    psutil = None

from .config import RuntimeBudget


@dataclass(frozen=True)
class ResourceSample:
    rss_mb: int
    peak_vram_mb: int
    warning: bool
    exceeded: bool


def sample_process_tree_memory(pid: int | None, budget: RuntimeBudget) -> ResourceSample:
    """Sample RSS memory across the process tree and peak VRAM allocated."""
    total_rss_bytes = 0
    if pid is not None and psutil is not None:
        with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
            parent = psutil.Process(pid)
            total_rss_bytes += parent.memory_info().rss
            for child in parent.children(recursive=True):
                with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                    total_rss_bytes += child.memory_info().rss
    elif psutil is not None:
        with contextlib.suppress(Exception):
            proc = psutil.Process(os.getpid())
            total_rss_bytes = proc.memory_info().rss


    rss_mb = int(total_rss_bytes / (1024 * 1024))
    cuda_sample = sample_cuda_memory(budget)
    peak_vram_mb = cuda_sample.peak_vram_mb

    rss_warning = (rss_mb >= budget.rss_warning_mb) if budget.rss_warning_mb > 0 else False
    rss_exceeded = (rss_mb >= budget.rss_kill_mb) if budget.rss_kill_mb > 0 else False
    vram_warning = (peak_vram_mb >= budget.vram_warning_mb) if peak_vram_mb > 0 else False
    vram_exceeded = (peak_vram_mb >= budget.vram_kill_mb) if peak_vram_mb > 0 else False

    warning = rss_warning or vram_warning
    exceeded = rss_exceeded or vram_exceeded

    return ResourceSample(
        rss_mb=rss_mb,
        peak_vram_mb=peak_vram_mb,
        warning=warning,
        exceeded=exceeded,
    )


def sample_cuda_memory(budget: RuntimeBudget) -> ResourceSample:
    # Current PyTorch CUDA wheels are not safe to import on Python 3.14.
    if sys.version_info >= (3, 14):
        return ResourceSample(0, 0, False, False)
    try:
        import torch
    except (ImportError, OSError, RuntimeError):
        return ResourceSample(0, 0, False, False)
    if not torch.cuda.is_available():
        return ResourceSample(0, 0, False, False)
    try:
        peak = int(torch.cuda.max_memory_allocated() / (1024 * 1024))
    except Exception:
        peak = 0
    return ResourceSample(0, peak, peak >= budget.vram_warning_mb, peak >= budget.vram_kill_mb)


def reset_cuda_memory() -> None:
    if sys.version_info >= (3, 14):
        return
    try:
        import torch
    except (ImportError, OSError, RuntimeError):
        return
    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        except Exception:
            pass
