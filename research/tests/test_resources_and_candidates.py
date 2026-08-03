from __future__ import annotations

import sys

import numpy as np
import pytest

from stock_autoresearch.resources import sample_cuda_memory


def test_resource_sample_is_safe_without_cuda() -> None:
    budget = type("Budget", (), {"vram_warning_mb": 5200, "vram_kill_mb": 5500})()
    sample = sample_cuda_memory(budget)
    assert sample.peak_vram_mb >= 0
    assert not sample.exceeded


@pytest.mark.skipif(sys.version_info >= (3, 14), reason="PyTorch CUDA wheel is not compatible with Python 3.14")
def test_torch_lstm_candidate_has_finite_output() -> None:
    pytest.importorskip("torch")
    from stock_autoresearch.torch_candidates import TorchLSTMCandidate

    rng = np.random.default_rng(3)
    x = rng.normal(size=(12, 8, 2)).astype(np.float32)
    y = rng.normal(size=12).astype(np.float32)
    candidate = TorchLSTMCandidate(hidden_size=4, epochs=1, device="cpu").fit(x, y)
    prediction = candidate.predict(x[:3])
    assert prediction.shape == (3,)
    assert np.isfinite(prediction).all()
