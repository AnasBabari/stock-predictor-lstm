"""Slice-10 tests: champion selection gates, Holm correction, shrinkage."""

from __future__ import annotations

import numpy as np

from panel.selection import (
    HorizonEvidence,
    diebold_mariano_hac,
    holm_correction,
    select_champion,
)


def evidence(
    rel_mae=0.9,
    rel_rmse=0.9,
    upper=0.95,
    dm_p=0.01,
    folds=(0.85, 0.88, 0.9, 0.92, 0.87),
    seeds=(),
) -> HorizonEvidence:
    return HorizonEvidence(
        horizon=7,
        candidate_name="ridge_global",
        rel_mae=rel_mae,
        rel_rmse=rel_rmse,
        loss_diff_upper_95=upper,
        dm_p_value=dm_p,
        fold_relative_rmses=list(folds),
        seed_relative_rmses=list(seeds),
    )


def _clean_losses(n: int = 200, seed: int = 1):
    rng = np.random.default_rng(seed)
    base = np.abs(rng.normal(1.0, 0.2, n))
    cand = np.maximum(base * 0.85 + rng.normal(0, 0.02, n), 1e-6)
    return cand, base


def test_strong_evidence_is_promoted() -> None:
    cand, base = _clean_losses()
    decision = select_champion(
        evidence(), validation_learned_loss=cand, validation_baseline_loss=base
    )
    assert decision.status in ("promoted", "blended_with_baseline")
    assert decision.alpha > 0
    assert decision.reasons == []


def test_marginal_edge_blends_instead_of_full_promotion() -> None:
    cand, base = _clean_losses()
    dec = select_champion(
        evidence(rel_rmse=0.985),
        validation_learned_loss=cand,
        validation_baseline_loss=base,
    )
    assert dec.status == "blended_with_baseline"
    assert 0 < dec.alpha < 1


def test_no_edge_gives_alpha_zero_and_baseline_decision() -> None:
    cand, base = _clean_losses()
    dec = select_champion(
        evidence(rel_mae=1.05, rel_rmse=1.08, upper=1.12, dm_p=0.4, folds=(1.02, 1.05, 1.2)),
        validation_learned_loss=cand,
        validation_baseline_loss=base,
    )
    assert dec.status == "experimental_no_demonstrated_edge"
    assert dec.candidate_name is None
    assert dec.alpha == 0.0
    assert len(dec.reasons) >= 3


def test_worst_fold_ceiling_rejects() -> None:
    cand, base = _clean_losses()
    dec = select_champion(
        evidence(folds=(0.85, 0.9, 0.88, 0.91, 1.30)),
        validation_learned_loss=cand,
        validation_baseline_loss=base,
    )
    assert any("ceiling" in r for r in dec.reasons)


def test_seed_dispersion_rejects() -> None:
    cand, base = _clean_losses()
    dec = select_champion(
        evidence(seeds=[0.8, 1.05]),
        validation_learned_loss=cand,
        validation_baseline_loss=base,
    )
    assert any("dispersion" in r for r in dec.reasons)


def test_holm_correction_step_down_semantics() -> None:
    # Standard Holm: reject while p_(k) <= alpha/(m−k+1), stop at first failure.
    p = [0.001, 0.02, 0.04, 0.2]
    decisions = holm_correction(p, alpha=0.05)
    # sorted: .001 <= .0125 ✓; .02 > .0167 ✗ → stop; rest never rejected.
    assert decisions == [True, False, False, False]
    p2 = [0.5, 0.001, 0.03]
    d2 = holm_correction(p2, alpha=0.05)
    # sorted: .001 (✓ at .05/3); .03 (> .05/2 ✗ stop) → .5 never tested
    assert d2 == [False, True, False]


def test_dm_detects_systematic_improvement() -> None:
    rng = np.random.default_rng(2)
    base = np.abs(rng.normal(1.0, 0.1, 300))
    cand = base * 0.8 + rng.normal(0, 0.005, 300)
    stat, p = diebold_mariano_hac(cand, base, max_lag=2)
    assert stat < 0
    assert p < 0.05
