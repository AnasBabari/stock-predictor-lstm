"""Independent per-horizon candidate selection for StockLSTM V10.

Evaluates candidates independently at each horizon h in {1, 3, 5, 7, 14, 30}.
If a neural candidate passes on short horizons (e.g. h=1) but degrades on longer
horizons, the short horizon candidate can be promoted while long horizons fall
back to development baseline candidates (e.g. HAR).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("horizon_selection_v10")


def select_champions_by_horizon(
    ledger_records: list[dict[str, Any]],
    horizons: list[int] | tuple[int, ...] = (1, 3, 5, 7, 14, 30),
    *,
    baseline_family: str = "har",
    max_relative_qlike: float = 1.0,
    max_ratio_upper_95: float = 1.0,
    expected_folds: int = 1,
) -> dict[int, dict[str, Any]]:
    """Select the best performing candidate for each horizon independently."""
    champions: dict[int, dict[str, Any]] = {}

    for h in horizons:
        h_records = [r for r in ledger_records if int(r.get("horizon", 0)) == h]
        if not h_records:
            champions[h] = {
                "horizon": h,
                "champion_family": baseline_family,
                "role": "development_baseline_candidate",
                "relative_qlike": 1.0,
                "reason": "no_records_for_horizon",
            }
            continue

        # Group by candidate family
        families: dict[str, list[dict[str, Any]]] = {}
        for r in h_records:
            fam = str(r.get("family", "unknown"))
            families.setdefault(fam, []).append(r)

        eligible_candidates = []
        for fam, records in families.items():
            if fam == baseline_family:
                continue
            if len(records) < expected_folds:
                continue

            rel_qlikes = [float(r.get("relative_qlike", 1.0)) for r in records]
            mean_rel_qlike = float(np.mean(rel_qlikes))
            max_upper_95 = float(np.max([float(r.get("ratio_upper_95", 1.0)) for r in records]))
            worst_fold_rel_qlike = float(np.max(rel_qlikes))

            # Check promotion gates
            if (
                mean_rel_qlike < max_relative_qlike
                and max_upper_95 < max_ratio_upper_95
                and worst_fold_rel_qlike <= 1.05
            ):
                eligible_candidates.append(
                    {
                        "family": fam,
                        "mean_relative_qlike": mean_rel_qlike,
                        "max_upper_95": max_upper_95,
                        "worst_fold": worst_fold_rel_qlike,
                        "record_count": len(records),
                    }
                )

        if eligible_candidates:
            # Sort by mean relative QLIKE (lowest is best)
            eligible_candidates.sort(key=lambda c: c["mean_relative_qlike"])
            winner = eligible_candidates[0]
            champions[h] = {
                "horizon": h,
                "champion_family": winner["family"],
                "role": "learned_candidate",
                "relative_qlike": winner["mean_relative_qlike"],
                "ratio_upper_95": winner["max_upper_95"],
                "eligible_count": len(eligible_candidates),
            }
        else:
            champions[h] = {
                "horizon": h,
                "champion_family": baseline_family,
                "role": "development_baseline_candidate",
                "relative_qlike": 1.0,
                "reason": "no_learned_candidate_cleared_gates",
            }

    return champions
