"""Guards proving the v9 evaluator cannot manufacture a winner.

Each test targets one way a selection pipeline can produce a confident-looking
but invalid champion. If any of these regress, a model could be "selected" on
the strength of a bug rather than on predictive skill.

The invariant under test:

    Given identical development evidence, the evaluator cannot produce a
    winner through metric orientation, missing folds, fake seeds, horizon
    averaging, silent baseline substitution, or mislabelled implementations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.panel.features import DEPLOYABLE_FEATURE_COLUMNS_V5
from research.volatility_forecasting.architecture_ablation import (
    DETERMINISTIC_SEED,
    EVALUATED_FAMILIES,
    REQUIRED_HORIZONS,
    UNIMPLEMENTED_FAMILIES,
    assert_family_is_implemented,
    compute_forecast_metrics,
    select_numeric_champion,
)
from research.volatility_forecasting.data import (
    VolatilityPanelExamples,
    subset_volatility_panel_examples,
)
from research.volatility_forecasting.metrics import qlike_losses

# ---------------------------------------------------------------------------
# QLIKE orientation
# ---------------------------------------------------------------------------


def test_qlike_is_asymmetric_so_orientation_is_detectable() -> None:
    """Reversing forecast/realized must change the result, not silently match."""
    forecast = np.array([0.04, 0.09, 0.16, 0.25])
    realized = np.array([0.05, 0.08, 0.15, 0.26])
    correct = float(np.mean(qlike_losses(forecast, realized)))
    reversed_ = float(np.mean(qlike_losses(realized, forecast)))
    assert not np.isclose(correct, reversed_)
    assert np.isfinite(correct) and np.isfinite(reversed_)


def test_compute_forecast_metrics_uses_forecast_first_orientation() -> None:
    """The helper must agree with a direct correctly-oriented QLIKE call."""
    realized = np.array([0.04, 0.09, 0.16, 0.25])
    forecast = np.array([0.05, 0.08, 0.15, 0.26])
    baseline = np.array([0.06, 0.10, 0.18, 0.28])
    metrics = compute_forecast_metrics(
        realized_variance=realized,
        forecast_variance=forecast,
        baseline_variance=baseline,
    )
    assert np.isclose(
        metrics["qlike"],
        float(np.mean(qlike_losses(forecast, realized))),
    )


def test_reversed_orientation_flips_the_reported_winner() -> None:
    """A silent argument swap would invert which forecast is preferred.

    QLIKE is asymmetric: it penalises under-prediction more heavily than
    over-prediction. With realized = 1, an over-predictor of 2 and an
    under-predictor of 0.5 are equidistant in log space, but correctly
    oriented QLIKE prefers the over-predictor. Reversed, it prefers the
    under-predictor. A swap is therefore never a harmless no-op.
    """
    realized = np.array([1.0])
    over = np.array([2.0])
    under = np.array([0.5])

    over_correct = float(np.mean(qlike_losses(over, realized)))
    under_correct = float(np.mean(qlike_losses(under, realized)))
    over_reversed = float(np.mean(qlike_losses(realized, over)))
    under_reversed = float(np.mean(qlike_losses(realized, under)))

    # Correct orientation prefers the over-predictor.
    assert over_correct < under_correct
    # Reversed orientation prefers the under-predictor: the ranking inverts.
    assert under_reversed < over_reversed
    # And each call is simply the other's argument-swapped value.
    assert np.isclose(over_reversed, under_correct)
    assert np.isclose(under_reversed, over_correct)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _frame(
    families: tuple[str, ...],
    *,
    folds: tuple[int, ...] = (1, 2, 3, 4, 5),
    horizons: tuple[int, ...] = REQUIRED_HORIZONS,
    seeds: dict[str, tuple[int, ...]] | None = None,
    ratio: float = 0.90,
    p95: float = 0.95,
) -> pd.DataFrame:
    """Build a results frame with one row per family/fold/seed/horizon."""
    neural = {"gru", "lstm", "tcn", "patch_transformer"}
    rows: list[dict[str, object]] = []
    for family in families:
        family_seeds = (seeds or {}).get(
            family, (41, 42, 43) if family in neural else (DETERMINISTIC_SEED,)
        )
        for fold in folds:
            for seed in family_seeds:
                for horizon in horizons:
                    rows.append(
                        {
                            "family": family,
                            "fold": fold,
                            "seed": seed,
                            "horizon": horizon,
                            "mean_qlike": 1.0,
                            "har_qlike": 1.1,
                            "relative_qlike_ratio": ratio,
                            "bootstrap_p05": 0.85,
                            "bootstrap_p95": p95,
                            "duration_seconds": 0.0,
                        }
                    )
    return pd.DataFrame(rows)


def _qualified() -> pd.DataFrame:
    """A neural family that legitimately clears every gate."""
    return _frame(("lstm",), ratio=0.90, p95=0.95)


# ---------------------------------------------------------------------------
# Selection gates
# ---------------------------------------------------------------------------


def test_fully_qualified_neural_family_is_selected() -> None:
    decision = select_numeric_champion(_qualified())
    assert decision.selected_family == "lstm"
    assert decision.selection_state == "learned_candidate_selected_on_development_only"


def test_har_is_retained_when_no_learner_qualifies() -> None:
    decision = select_numeric_champion(_frame(("lstm",), ratio=1.20, p95=1.30))
    assert decision.selected_family == "har"
    assert decision.selection_state == "baseline_retained_no_learned_candidate_qualified"
    assert decision.eligible_families == ()


def test_har_fallback_never_masquerades_as_another_family() -> None:
    """The retained fallback must be labelled HAR, not the family that lost."""
    decision = select_numeric_champion(_frame(("lstm", "tcn"), ratio=1.20, p95=1.30))
    assert decision.selected_family == "har"
    assert decision.selected_family not in {"lstm", "tcn"}
    assert "lstm" in decision.reasons_by_family
    assert "tcn" in decision.reasons_by_family


def test_a_single_losing_required_horizon_rejects_the_candidate() -> None:
    """Horizon averaging must not be able to hide a loss at a required horizon."""
    frame = _qualified()
    losing = frame[(frame["family"] == "lstm") & (frame["horizon"] == 7)].index
    frame.loc[losing, "relative_qlike_ratio"] = 1.30
    # It still wins comfortably on the other three horizons.
    others = frame[(frame["family"] == "lstm") & (frame["horizon"] != 7)]
    assert (others["relative_qlike_ratio"] < 1.0).all()
    decision = select_numeric_champion(frame)
    assert decision.selected_family == "har"
    assert any("did not beat HAR" in r for r in decision.reasons_by_family["lstm"])


def test_missing_horizon_evidence_rejects_the_candidate() -> None:
    frame = _qualified()
    frame = frame[~((frame["family"] == "lstm") & (frame["horizon"] == 5))]
    decision = select_numeric_champion(frame)
    assert decision.selected_family == "har"
    assert "required horizon evidence is incomplete" in decision.reasons_by_family["lstm"]


def test_incomplete_folds_reject_the_candidate() -> None:
    frame = _frame(("lstm",), folds=(1, 2, 3))
    decision = select_numeric_champion(frame)
    assert decision.selected_family == "har"
    assert "expanding-fold evidence is incomplete" in decision.reasons_by_family["lstm"]


def test_incomplete_neural_seeds_reject_the_candidate() -> None:
    frame = _frame(("lstm",), seeds={"lstm": (41, 42)})
    decision = select_numeric_champion(frame)
    assert decision.selected_family == "har"
    assert "seed evidence is incomplete or mislabeled" in decision.reasons_by_family["lstm"]


def test_deterministic_candidates_do_not_gain_fake_seed_replication() -> None:
    """Repeating a deterministic fit across seeds is rejected as mislabeled."""
    frame = _frame(("ridge",), seeds={"ridge": (41, 42, 43)})
    decision = select_numeric_champion(frame)
    assert decision.selected_family == "har"
    assert "seed evidence is incomplete or mislabeled" in decision.reasons_by_family["ridge"]


def test_deterministic_candidate_with_seed_zero_is_accepted() -> None:
    frame = _frame(("ridge",), seeds={"ridge": (DETERMINISTIC_SEED,)})
    decision = select_numeric_champion(frame)
    assert decision.selected_family == "ridge"


def test_non_finite_scores_reject_the_candidate() -> None:
    frame = _qualified()
    frame.loc[frame.index[0], "relative_qlike_ratio"] = float("nan")
    decision = select_numeric_champion(frame)
    assert decision.selected_family == "har"
    assert any("non-finite" in r for r in decision.reasons_by_family["lstm"])


def test_bootstrap_upper_bound_above_threshold_rejects_the_candidate() -> None:
    """A point estimate better than HAR is not enough; the interval must clear."""
    frame = _frame(("lstm",), ratio=0.90, p95=1.05)
    decision = select_numeric_champion(frame)
    assert decision.selected_family == "har"
    assert any("bootstrap upper bound" in r for r in decision.reasons_by_family["lstm"])


def test_missing_required_column_fails_closed() -> None:
    frame = _qualified().drop(columns=["bootstrap_p95"])
    with pytest.raises(ValueError, match="missing columns"):
        select_numeric_champion(frame)


def test_best_family_wins_when_several_qualify() -> None:
    frame = pd.concat(
        [
            _frame(("lstm",), ratio=0.90, p95=0.93),
            _frame(("tcn",), ratio=0.96, p95=0.99),
        ],
        ignore_index=True,
    )
    decision = select_numeric_champion(frame)
    assert decision.selected_family == "lstm"
    assert decision.eligible_families == ("lstm", "tcn")


# ---------------------------------------------------------------------------
# Candidate identity
# ---------------------------------------------------------------------------


def test_unimplemented_families_are_rejected_not_substituted() -> None:
    """A name with no implementation must never borrow another model's maths."""
    for family in sorted(UNIMPLEMENTED_FAMILIES):
        with pytest.raises(NotImplementedError):
            assert_family_is_implemented(family)


def test_unknown_family_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown candidate family"):
        assert_family_is_implemented("mystery_net")


def test_implemented_families_are_exactly_the_evaluated_set() -> None:
    assert set(EVALUATED_FAMILIES) == {
        "har",
        "ewma",
        "garch",
        "gjr",
        "ridge",
        "elasticnet",
        "gru",
        "lstm",
        "tcn",
        "patch_transformer",
    }
    # No family may be both implemented and unimplemented.
    assert not (set(EVALUATED_FAMILIES) & set(UNIMPLEMENTED_FAMILIES))
    for family in EVALUATED_FAMILIES:
        assert assert_family_is_implemented(family) == family


# ---------------------------------------------------------------------------
# Silent baseline substitution
# ---------------------------------------------------------------------------


def test_garch_fallback_is_recorded_not_silent() -> None:
    """A ticker without enough history must be recorded, never quietly scored."""
    from research.volatility_forecasting.architecture_ablation import (
        GARCH_MINIMUM_TRAIN_ROWS,
        evaluate_classical_model,
        garch_coverage_diagnostics,
        reset_garch_diagnostics,
    )

    reset_garch_diagnostics()
    examples = _panel(n=40)
    train = np.arange(0, 20)
    assert len(train) < GARCH_MINIMUM_TRAIN_ROWS
    val = np.arange(20, 40)
    try:
        evaluate_classical_model("garch", examples, train, val, horizon_idx=0)
    finally:
        recorded = garch_coverage_diagnostics()
    assert recorded, "a GARCH-to-HAR substitution must be recorded, not silent"
    assert all(item["substituted_with"] == "har" for item in recorded)
    assert all(item["family"] == "garch" for item in recorded)


def test_garch_diagnostics_reset_isolates_runs() -> None:
    from research.volatility_forecasting.architecture_ablation import (
        garch_coverage_diagnostics,
        reset_garch_diagnostics,
    )

    reset_garch_diagnostics()
    assert garch_coverage_diagnostics() == ()


def test_evaluate_classical_model_rejects_unimplemented_family() -> None:
    from research.volatility_forecasting.architecture_ablation import evaluate_classical_model

    examples = _panel(n=20)
    with pytest.raises(NotImplementedError):
        evaluate_classical_model("dlinear", examples, np.arange(10), np.arange(10, 20), 0)


# ---------------------------------------------------------------------------
# Panel subsetting identity
# ---------------------------------------------------------------------------


def _panel(n: int = 40) -> VolatilityPanelExamples:
    """Panel using the real deployable_v5 schema so column lookups are faithful."""
    rng = np.random.default_rng(7)
    names = DEPLOYABLE_FEATURE_COLUMNS_V5
    return VolatilityPanelExamples(
        features=rng.normal(size=(n, 60, len(names))).astype(np.float32),
        baseline_variance=np.abs(rng.normal(size=(n, 6)).astype(np.float32)) + 0.01,
        realized_variance=np.abs(rng.normal(size=(n, 6)).astype(np.float32)) + 0.01,
        cumulative_returns=rng.normal(size=(n, 6)).astype(np.float32) * 0.01,
        direction_classes=rng.integers(0, 3, size=(n, 6)),
        tickers=np.array(["AAPL", "MSFT", "NVDA", "CSCO"] * (n // 4), dtype=str),
        origin_dates=np.array(
            [np.datetime64("2022-01-03") + np.timedelta64(i, "D") for i in range(n)],
            dtype="datetime64[D]",
        ),
        origin_closes=np.linspace(100.0, 140.0, n),
        horizons=(1, 3, 5, 7, 14, 30),
        feature_names=names,
    )


def test_subset_preserves_row_target_and_identity_alignment() -> None:
    examples = _panel()
    rows = np.array([0, 5, 13, 27, 39])
    subset = subset_volatility_panel_examples(examples, rows)
    for row, source in enumerate(rows):
        assert np.array_equal(subset.features[row], examples.features[source])
        assert np.array_equal(subset.realized_variance[row], examples.realized_variance[source])
        assert np.array_equal(subset.baseline_variance[row], examples.baseline_variance[source])
        assert np.array_equal(subset.cumulative_returns[row], examples.cumulative_returns[source])
        assert subset.tickers[row] == examples.tickers[source]
        assert subset.origin_dates[row] == examples.origin_dates[source]
        assert subset.origin_closes[row] == examples.origin_closes[source]
    assert subset.horizons == examples.horizons
    assert subset.feature_names == examples.feature_names


def test_subset_accepts_boolean_mask_equivalently() -> None:
    examples = _panel()
    rows = np.array([1, 4, 9])
    mask = np.zeros(len(examples.features), dtype=bool)
    mask[rows] = True
    assert np.array_equal(
        subset_volatility_panel_examples(examples, mask).features,
        subset_volatility_panel_examples(examples, rows).features,
    )


def test_subset_rejects_empty_duplicate_and_out_of_bounds() -> None:
    examples = _panel()
    with pytest.raises(ValueError):
        subset_volatility_panel_examples(examples, np.array([], dtype=np.int64))
    with pytest.raises(ValueError):
        subset_volatility_panel_examples(examples, np.array([1, 1, 2]))
    with pytest.raises(ValueError):
        subset_volatility_panel_examples(examples, np.array([len(examples.features)]))
