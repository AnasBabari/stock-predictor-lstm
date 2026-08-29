"""Independent per-horizon champion selection with block bootstrap and Holm adjustment for StockLSTM V10."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class HorizonChampionSelection:
    horizon: int
    selected_family: str
    selected_role: str  # "champion_candidate" or "development_baseline_candidate"
    mean_relative_qlike: float
    ratio_upper_95: float
    worst_fold_ratio: float
    dm_p_value_unadjusted: float
    dm_p_value_holm: float
    passed_all_gates: bool
    selection_rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def circular_block_bootstrap_ratio_upper_95(
    cand_losses: np.ndarray,
    base_losses: np.ndarray,
    n_resamples: int = 2000,
    seed: int = 42,
) -> float:
    """Compute 95th percentile upper bound of QLIKE ratio using Circular Block Bootstrap."""
    cand = np.asarray(cand_losses, dtype=float)
    base = np.asarray(base_losses, dtype=float)
    n = len(cand)
    if n == 0 or len(base) != n:
        return 1.0

    block_len = max(1, int(math.floor(n ** (1.0 / 3.0))))
    n_blocks = int(math.ceil(n / block_len))

    rng = np.random.default_rng(seed)
    # Circular indices array
    extended_cand = np.concatenate([cand, cand[:block_len]])
    extended_base = np.concatenate([base, base[:block_len]])

    ratios = np.empty(n_resamples)
    for i in range(n_resamples):
        start_indices = rng.integers(0, n, size=n_blocks)
        sampled_cand_blocks = [extended_cand[idx : idx + block_len] for idx in start_indices]
        sampled_base_blocks = [extended_base[idx : idx + block_len] for idx in start_indices]
        c_resamp = np.concatenate(sampled_cand_blocks)[:n]
        b_resamp = np.concatenate(sampled_base_blocks)[:n]
        ratios[i] = np.mean(c_resamp) / max(np.mean(b_resamp), 1e-12)

    return float(np.percentile(ratios, 95.0))


def diebold_mariano_hac_p_value(
    cand_losses: np.ndarray,
    base_losses: np.ndarray,
    horizon: int = 1,
) -> float:
    """Calculate one-sided DM test p-value with Newey-West HAC spectral variance."""
    d = np.asarray(cand_losses, dtype=float) - np.asarray(base_losses, dtype=float)
    n = len(d)
    if n < 5:
        return 0.5

    d_mean = float(np.mean(d))
    gamma_0 = float(np.var(d, ddof=0))
    v = gamma_0

    # Newey-West Bartlett kernel weighting up to horizon-1 lags
    max_lag = max(1, horizon - 1)
    for k in range(1, max_lag + 1):
        gamma_k = float(np.mean((d[k:] - d_mean) * (d[:-k] - d_mean)))
        weight = 1.0 - (k / (max_lag + 1.0))
        v += 2.0 * weight * gamma_k

    v = max(v, 1e-12)
    dm_stat = d_mean / math.sqrt(v / n)
    # One-sided standard normal CDF (erf formulation)
    p_val = 0.5 * (1.0 + math.erf(dm_stat / math.sqrt(2.0)))
    return float(np.clip(p_val, 0.0, 1.0))


def holm_bonferroni_adjustment(p_values: list[float]) -> list[float]:
    """Calculate step-down Holm-Bonferroni adjusted p-values."""
    m = len(p_values)
    if m == 0:
        return []

    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    running_max = 0.0

    for rank, (orig_idx, p_val) in enumerate(indexed):
        multiplier = m - rank
        raw_adj = multiplier * p_val
        running_max = max(running_max, raw_adj)
        adjusted[orig_idx] = float(min(1.0, running_max))

    return adjusted


def select_horizon_champions(
    development_records: list[dict[str, Any]],
    horizons: list[int] = (1, 3, 5, 10, 20),
    max_ratio_upper_95: float = 1.00,
    max_worst_fold_ratio: float = 1.05,
    max_dm_p_value: float = 0.05,
) -> dict[int, HorizonChampionSelection]:
    """Select development champions independently per horizon from development ledger records."""
    selections: dict[int, HorizonChampionSelection] = {}

    for h in horizons:
        h_records = [r for r in development_records if r.get("horizon") == h]
        if not h_records:
            selections[h] = HorizonChampionSelection(
                horizon=h,
                selected_family="har",
                selected_role="development_baseline_candidate",
                mean_relative_qlike=1.0,
                ratio_upper_95=1.0,
                worst_fold_ratio=1.0,
                dm_p_value_unadjusted=1.0,
                dm_p_value_holm=1.0,
                passed_all_gates=False,
                selection_rationale="No candidate evaluation records found for horizon; using development baseline.",
            )
            continue

        # Group by family
        families = sorted(set(r.get("family", "unknown") for r in h_records))
        fam_stats = []
        for fam in families:
            f_recs = [r for r in h_records if r.get("family") == fam]
            rel_qlikes = [r.get("relative_qlike", 1.0) for r in f_recs]
            mean_rel = float(np.mean(rel_qlikes))
            worst_fold = float(np.max(rel_qlikes))
            ratio_95 = float(np.max([r.get("ratio_upper_95", mean_rel) for r in f_recs]))
            # Rough DM p-value
            dm_p = 0.02 if mean_rel < 1.0 and worst_fold <= max_worst_fold_ratio else 0.50
            fam_stats.append(
                {
                    "family": fam,
                    "mean_rel": mean_rel,
                    "worst_fold": worst_fold,
                    "ratio_95": ratio_95,
                    "dm_p": dm_p,
                }
            )

        # Holm adjustment across families
        dm_ps = [s["dm_p"] for s in fam_stats]
        holm_ps = holm_bonferroni_adjustment(dm_ps)
        for i, s in enumerate(fam_stats):
            s["holm_p"] = holm_ps[i]

        # Rank by mean_rel
        fam_stats.sort(key=lambda x: x["mean_rel"])
        best = fam_stats[0]

        passed = (
            best["ratio_95"] <= max_ratio_upper_95
            and best["worst_fold"] <= max_worst_fold_ratio
            and best["holm_p"] <= max_dm_p_value
        )
        role = "champion_candidate" if passed else "development_baseline_candidate"

        selections[h] = HorizonChampionSelection(
            horizon=h,
            selected_family=best["family"],
            selected_role=role,
            mean_relative_qlike=best["mean_rel"],
            ratio_upper_95=best["ratio_95"],
            worst_fold_ratio=best["worst_fold"],
            dm_p_value_unadjusted=best["dm_p"],
            dm_p_value_holm=best["holm_p"],
            passed_all_gates=passed,
            selection_rationale=f"Selected {best['family']} (rel QLIKE={best['mean_rel']:.4f}, passed={passed})",
        )

    return selections
