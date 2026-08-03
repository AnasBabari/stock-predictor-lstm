"""Cross-framework PyTorch and TensorFlow.js parity verification utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ParityResult:
    passed: bool
    shape_matches: bool
    finite: bool
    max_abs_diff: float
    tolerance: float
    details: str


def make_parity_fixture(
    samples: int = 10, window: int = 60, features: int = 28, seed: int = 42
) -> np.ndarray:
    """Generate a deterministic synthetic input tensor fixture (N, window, features)."""
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=0.02, size=(samples, window, features)).astype(np.float32)


def verify_prediction_parity(
    pytorch_preds: np.ndarray,
    tfjs_preds: np.ndarray,
    *,
    tolerance: float = 1e-3,
) -> ParityResult:
    """Verify PyTorch and TFJS predictions match within a numerical tolerance threshold."""
    if pytorch_preds.shape != tfjs_preds.shape:
        return ParityResult(
            passed=False,
            shape_matches=False,
            finite=False,
            max_abs_diff=float("inf"),
            tolerance=tolerance,
            details=f"Shape mismatch: PyTorch {pytorch_preds.shape} vs TFJS {tfjs_preds.shape}",
        )

    py_finite = np.isfinite(pytorch_preds).all()
    tf_finite = np.isfinite(tfjs_preds).all()
    if not (py_finite and tf_finite):
        return ParityResult(
            passed=False,
            shape_matches=True,
            finite=False,
            max_abs_diff=float("inf"),
            tolerance=tolerance,
            details="One or both predictions contain non-finite values (NaN/Inf).",
        )

    abs_diff = np.abs(pytorch_preds - tfjs_preds)
    max_diff = float(np.max(abs_diff))
    passed = max_diff <= tolerance

    return ParityResult(
        passed=passed,
        shape_matches=True,
        finite=True,
        max_abs_diff=max_diff,
        tolerance=tolerance,
        details="Predictions match within tolerance"
        if passed
        else f"Max difference {max_diff:.6f} exceeded tolerance {tolerance}",
    )
