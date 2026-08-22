from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

REQUIRED_FOLDS = 5
MIN_FOLD_WINS = 4
MAX_FOLD_RELATIVE_RMSE = 1.15
SEED_DISPERSION_MAX = 0.05
REQUIRED_NEURAL_SEEDS = 3


def compute_bootstrap_ratio_upper_bound(
    candidate_squared_errors: np.ndarray,
    baseline_squared_errors: np.ndarray,
    *,
    block_length: int = 10,
    confidence: float = 0.95,
    resamples: int = 1000,
    seed: int = 42,
) -> float:
    """Moving-block bootstrap directly on the paired ratio statistic:
    sqrt(mean(candidate_sq)) / sqrt(mean(baseline_sq)).
    """
    c = np.asarray(candidate_squared_errors, dtype=float)
    b = np.asarray(baseline_squared_errors, dtype=float)
    if len(c) != len(b):
        raise ValueError(f"Length mismatch: candidate={len(c)}, baseline={len(b)}")
    n = len(c)
    if n < 10:
        return float("inf")

    rng = np.random.default_rng(seed)
    bl = max(1, min(block_length, n // 2))
    n_blocks = int(np.ceil(n / bl))
    num_starts = n - bl + 1
    ratios = np.empty(resamples, dtype=float)

    for i in range(resamples):
        starts = rng.integers(0, num_starts, size=n_blocks)
        indices = np.concatenate([np.arange(s, s + bl) for s in starts])[:n]
        mean_c = float(np.mean(c[indices]))
        mean_b = float(np.mean(b[indices]))
        if mean_b <= 0 or not np.isfinite(mean_b) or not np.isfinite(mean_c):
            ratios[i] = float("inf")
        else:
            ratios[i] = float(np.sqrt(mean_c) / np.sqrt(mean_b))

    return float(np.percentile(ratios, confidence * 100))


def diebold_mariano_hac(
    candidate_losses: np.ndarray,
    baseline_losses: np.ndarray,
    *,
    max_lag: int | None = None,
) -> tuple[float, float]:
    """DM statistic + two-sided p-value with Newey-West HAC variance."""
    c = np.asarray(candidate_losses, dtype=float)
    b = np.asarray(baseline_losses, dtype=float)
    if len(c) != len(b):
        raise ValueError("Length mismatch between candidate and baseline losses")
    d = c - b
    n = len(d)
    if n < 5:
        return 0.0, 1.0
    lag_max = max_lag if max_lag is not None else max(1, int(n ** (1 / 3)))
    d_mean = float(np.mean(d))
    var = float(np.var(d, ddof=0))
    d_demean = d - d_mean
    for lag in range(1, min(lag_max, n - 1) + 1):
        cov = float(np.mean(d_demean[:-lag] * d_demean[lag:]))
        var += 2.0 * (1.0 - lag / (lag_max + 1.0)) * cov
    se = float(np.sqrt(max(var, 0.0) / n))
    if se <= 1e-12:
        return (0.0, 1.0) if d_mean >= 0 else (-100.0, 0.0)
    stat = float(d_mean / se)
    from math import erf, sqrt

    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(stat) / sqrt(2.0))))
    return stat, float(max(0.0, min(1.0, p)))


def holm_correction(p_values: list[float], *, alpha: float = 0.05) -> list[bool]:
    """Holm step-down: returns reject decisions preserving family-wise error."""
    if not p_values:
        return []
    order = np.argsort(p_values)
    m = len(p_values)
    rejected = [False] * m
    for rank, idx in enumerate(order):
        threshold = alpha / (m - rank)
        if p_values[idx] <= threshold:
            rejected[idx] = True
        else:
            break
    return rejected


@dataclass
class HorizonEvidence:
    horizon: int
    candidate_name: str
    rel_mae: float
    rel_rmse: float
    loss_diff_upper_95: float  # bootstrap CI upper on relative skill
    dm_p_value: float  # vs persistence, HAC-corrected
    fold_relative_rmses: list[float]
    seed_relative_rmses: list[float] = field(default_factory=list)
    is_neural: bool = False
    calibration_ok: bool = True
    calibration_diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class SelectionDecision:
    horizon: int
    candidate_name: str | None
    status: str  # promoted | blended_with_baseline | experimental_no_demonstrated_edge
    alpha: float  # blend weight toward learned forecast
    reasons: list[str] = field(default_factory=list)

    def to_manifest(self) -> dict:
        return {
            "horizon": self.horizon,
            "candidate": self.candidate_name,
            "status": self.status,
            "alpha": round(self.alpha, 6),
            "reasons": list(self.reasons),
        }


def select_champion(
    evidence: HorizonEvidence,
    *,
    validation_learned_loss: np.ndarray,
    validation_baseline_loss: np.ndarray,
    family_p_values: list[float] | None = None,
    p_value_index: int | None = None,
    required_folds: int = REQUIRED_FOLDS,
    min_fold_wins: int = MIN_FOLD_WINS,
    max_fold_relative_rmse: float = MAX_FOLD_RELATIVE_RMSE,
    seed_dispersion_max: float = SEED_DISPERSION_MAX,
    required_neural_seeds: int = REQUIRED_NEURAL_SEEDS,
) -> SelectionDecision:
    """Apply §6.5 gates; estimate §6.6 shrinkage alpha when admissible."""
    reasons: list[str] = []
    horizon = evidence.horizon

    learned = np.asarray(validation_learned_loss, dtype=float)
    baseline = np.asarray(validation_baseline_loss, dtype=float)
    if len(learned) != len(baseline):
        raise ValueError(
            f"Misaligned loss arrays: learned={len(learned)}, baseline={len(baseline)}"
        )

    # 1. Point metrics gate
    if not (evidence.rel_mae < 1.0 and evidence.rel_rmse < 1.0):
        reasons.append("relative MAE/RMSE did not both beat persistence")
    if evidence.loss_diff_upper_95 >= 1.0:
        reasons.append("bootstrap 95% upper bound did not beat persistence")

    # 2. Strict fold requirements (never bypass if < required_folds)
    if len(evidence.fold_relative_rmses) < required_folds:
        reasons.append(
            f"insufficient folds: got {len(evidence.fold_relative_rmses)}, required {required_folds}"
        )
    elif not all(np.isfinite(r) for r in evidence.fold_relative_rmses):
        reasons.append("non-finite fold relative RMSE detected")
    else:
        fold_wins = sum(1 for r in evidence.fold_relative_rmses if r < 1.0)
        if fold_wins < min_fold_wins:
            reasons.append(
                f"won only {fold_wins}/{len(evidence.fold_relative_rmses)} folds (required {min_fold_wins})"
            )
        worst_fold = max(evidence.fold_relative_rmses, default=0.0)
        if worst_fold > max_fold_relative_rmse:
            reasons.append(f"worst fold {worst_fold:.3f} exceeded ceiling {max_fold_relative_rmse}")

    # 3. Seed stability for neural candidates
    if evidence.is_neural and len(evidence.seed_relative_rmses) < required_neural_seeds:
        reasons.append(
            f"neural candidate requires at least {required_neural_seeds} seeds, "
            f"got {len(evidence.seed_relative_rmses)}"
        )
    if len(evidence.seed_relative_rmses) >= 2:
        if not all(np.isfinite(s) for s in evidence.seed_relative_rmses):
            reasons.append("non-finite seed relative RMSE detected")
        else:
            seed_arr = np.asarray(evidence.seed_relative_rmses, dtype=float)
            spread = float((seed_arr.max() - seed_arr.min()) / max(1e-12, float(np.mean(seed_arr))))
            if spread > seed_dispersion_max:
                reasons.append(f"seed dispersion {spread:.3f} exceeded {seed_dispersion_max}")

    # 4. Calibration gate
    if not evidence.calibration_ok:
        reasons.append("failed task calibration gate")

    # 5. Diebold-Mariano test (requires lower loss, not just difference)
    dm_stat, dm_p = diebold_mariano_hac(learned, baseline)
    if dm_stat >= 0:
        reasons.append(f"candidate loss not lower than baseline (DM stat={dm_stat:.3f})")
    elif family_p_values is not None:
        idx = p_value_index if p_value_index is not None else -1
        rejected_family = holm_correction(family_p_values)
        if idx >= len(rejected_family) or not rejected_family[idx]:
            reasons.append("DM significance did not survive Holm correction")
    elif dm_p >= 0.05:
        reasons.append(f"DM test not significant at 5% (p={dm_p:.4f})")

    promoted = not reasons
    if promoted:
        # Shrinkage: alpha proportional to skill on development validation loss
        raw_alpha = min(1.0, max(0.0, (1.0 - evidence.rel_rmse) * 5.0))
        alpha = max(raw_alpha, 0.25) if evidence.rel_rmse < 0.98 else raw_alpha * 0.5
        status = "promoted" if alpha >= 0.999 else "blended_with_baseline"
        return SelectionDecision(horizon, evidence.candidate_name, status, alpha, [])

    return SelectionDecision(horizon, None, "experimental_no_demonstrated_edge", 0.0, reasons)
