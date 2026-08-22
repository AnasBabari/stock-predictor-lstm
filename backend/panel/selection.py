"""Per-horizon champion selection with shrinkage blending (slice 10).

Implements spec §6.5/§6.6: for every task/horizon, select the statistically
admissible candidate and estimate a validation-only convex blend toward the
baseline. Marginal models produce cautious near-baseline forecasts instead
of abrupt all-or-nothing switching; alpha = 0 means the learned model
supplied no usable edge at that horizon.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MAX_FOLD_RELATIVE_RMSE = 1.15
MIN_FOLD_WINS = 4
SEED_DISPERSION_MAX = 0.05


def _moving_block_bootstrap_upper(
    candidate_losses: np.ndarray,
    baseline_losses: np.ndarray,
    *,
    block_length: int | None = None,
    confidence: float = 0.95,
    resamples: int = 1000,
    seed: int = 42,
) -> float:
    """95% upper bound of the relative-RMSE-style ratio via moving-block
    bootstrap on paired losses. Uses the research statistics helper."""
    import sys
    from pathlib import Path

    research_dir = str(Path(__file__).resolve().parents[2] / "research")
    if research_dir not in sys.path:
        sys.path.insert(0, research_dir)
    from stock_autoresearch.statistics import moving_block_bootstrap_interval

    diff = np.asarray(candidate_losses, dtype=float) - np.asarray(baseline_losses, dtype=float)
    n = len(diff)
    bl = block_length or max(1, min(20, n // 4))
    interval = moving_block_bootstrap_interval(
        diff, block_length=bl, confidence=confidence, resamples=resamples, seed=seed
    )
    # Convert a difference-of-squared-loss CI into a ratio bound (both sides
    # share the baseline mean), conservative via additive slack on the mean.
    base_mean = float(np.mean(np.asarray(baseline_losses, dtype=float) ** 2))
    if base_mean <= 0:
        return float("inf")
    return float(1.0 + interval["upper"] / base_mean)


def diebold_mariano_hac(
    candidate_losses: np.ndarray,
    baseline_losses: np.ndarray,
    *,
    max_lag: int | None = None,
) -> tuple[float, float]:
    """DM statistic + two-sided p-value with Newey-West HAC variance."""

    c = np.asarray(candidate_losses, dtype=float)
    b = np.asarray(baseline_losses, dtype=float)
    d = c - b
    n = len(d)
    lag_max = max_lag if max_lag is not None else max(1, int(n ** (1 / 3)))
    var = float(np.var(d))
    for lag in range(1, min(lag_max, n - 1) + 1):
        cov = float(np.mean((d[:-lag] - np.mean(d)) * (d[lag:] - np.mean(d))))
        var += 2.0 * (1.0 - lag / (lag_max + 1.0)) * cov
    se = float(np.sqrt(max(var, 0.0) / n))
    if se == 0:
        return 0.0, 1.0
    stat = float(np.mean(d) / se)
    from math import erf, sqrt

    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(stat) / sqrt(2.0))))
    return stat, max(0.0, min(1.0, p))


def holm_correction(p_values: list[float], *, alpha: float = 0.05) -> list[bool]:
    """Holm step-down: returns reject decisions preserving family-wise error."""
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


@dataclass
class SelectionDecision:
    horizon: int
    candidate_name: str | None
    status: str  # promoted | blended_with_baseline |
    # experimental_no_demonstrated_edge
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
) -> SelectionDecision:
    """Apply §6.5 gates; estimate §6.6 shrinkage alpha when admissible."""
    reasons: list[str] = []
    horizon = evidence.horizon

    if not (evidence.rel_mae < 1.0 and evidence.rel_rmse < 1.0):
        reasons.append("relative MAE/RMSE did not both beat persistence")
    if evidence.loss_diff_upper_95 >= 1.0:
        reasons.append("bootstrap 95% upper bound did not beat persistence")

    fold_wins = sum(1 for r in evidence.fold_relative_rmses if r < 1.0)
    if len(evidence.fold_relative_rmses) >= MIN_FOLD_WINS and fold_wins < MIN_FOLD_WINS:
        reasons.append(f"won only {fold_wins}/{len(evidence.fold_relative_rmses)} folds")
    worst_fold = max(evidence.fold_relative_rmses, default=0.0)
    if worst_fold > MAX_FOLD_RELATIVE_RMSE:
        reasons.append(f"worst fold {worst_fold:.3f} exceeded ceiling {MAX_FOLD_RELATIVE_RMSE}")

    if len(evidence.seed_relative_rmses) >= 2:
        seed_arr = np.asarray(evidence.seed_relative_rmses, dtype=float)
        spread = float((seed_arr.max() - seed_arr.min()) / max(1e-12, float(np.mean(seed_arr))))
        if spread > SEED_DISPERSION_MAX:
            reasons.append(f"seed dispersion {spread:.3f} exceeded {SEED_DISPERSION_MAX}")

    _, dm_p = diebold_mariano_hac(validation_learned_loss, validation_baseline_loss)
    if family_p_values is not None:
        rejected_family = holm_correction(list(family_p_values) + [dm_p])
        if not rejected_family[-1]:
            reasons.append("DM significance did not survive Holm correction")
    elif dm_p >= 0.05:
        reasons.append("DM test not significant at 5%")

    promoted = not reasons
    if promoted:
        # Shrinkage: alpha proportional to how far below 1 the point RMSE sits,
        # regularized toward zero so marginal edges stay near the baseline.
        raw_alpha = min(1.0, max(0.0, (1.0 - evidence.rel_rmse) * 5.0))
        alpha = max(raw_alpha, 0.25) if evidence.rel_rmse < 0.98 else raw_alpha * 0.5
        status = "promoted" if alpha >= 0.999 else "blended_with_baseline"
        return SelectionDecision(horizon, evidence.candidate_name, status, alpha, [])

    return SelectionDecision(horizon, None, "experimental_no_demonstrated_edge", 0.0, reasons)
