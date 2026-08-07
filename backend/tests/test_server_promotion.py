"""Unit tests for the harmonized server promotion gates."""

import numpy as np

from server_models.promotion import assess_server_promotion

SELECTED_HORIZON = 30
CLOSES = np.full(300, 100.0) + np.random.default_rng(7).uniform(-2, 2, 300)


def _report(
    *,
    pooled_rmse=0.8,
    pooled_mae=0.8,
    day_rmse=0.85,
    day_mae=0.85,
    selected_rmse=0.85,
    selected_mae=0.85,
    rows=200,
    promoted=True,
    horizon_keys=("1", str(SELECTED_HORIZON)),
):
    per_horizon = {}
    for key in horizon_keys:
        rmse = day_rmse if key == "1" else selected_rmse
        mae = day_mae if key == "1" else selected_mae
        per_horizon[key] = {
            "relative_rmse": rmse,
            "relative_mae": mae,
            "sample_count": rows,
        }
    return {
        "aggregate": {
            "pooled": {
                "relative_rmse": pooled_rmse,
                "relative_mae": pooled_mae,
                "sample_count": rows * 30,
            },
            "per_horizon": per_horizon,
        },
        "promotion": {"promoted": promoted, "reasons": [] if promoted else ["fold gate failed"]},
    }


def test_promotes_a_candidate_that_beats_persistence_everywhere():
    passed, reasons = assess_server_promotion(
        _report(),
        selected_horizon=SELECTED_HORIZON,
        close_values=CLOSES,
        predicted_cumulative_return=0.03,
    )
    assert passed is True
    assert reasons == []


def test_rejects_when_the_walk_forward_policy_failed():
    passed, reasons = assess_server_promotion(
        _report(promoted=False), selected_horizon=SELECTED_HORIZON
    )
    assert passed is False
    assert "fold gate failed" in reasons


def test_rejects_when_pooled_relative_metrics_do_not_beat_persistence():
    passed, reasons = assess_server_promotion(
        _report(pooled_rmse=1.0, pooled_mae=0.8), selected_horizon=SELECTED_HORIZON
    )
    assert passed is False
    assert "Relative RMSE did not beat persistence." in reasons


def test_rejects_when_the_one_day_horizon_fails_the_cap():
    passed, reasons = assess_server_promotion(
        _report(day_rmse=1.02, day_mae=1.01), selected_horizon=SELECTED_HORIZON
    )
    assert passed is False
    assert any("did not beat persistence at the one-day horizon" in r for r in reasons)


def test_rejects_when_the_selected_horizon_fails_the_cap():
    passed, reasons = assess_server_promotion(
        _report(selected_rmse=1.05, selected_mae=0.9), selected_horizon=SELECTED_HORIZON
    )
    assert passed is False
    assert any("did not beat persistence at the selected horizon" in r for r in reasons)


def test_rejects_when_a_required_horizon_is_missing():
    passed, reasons = assess_server_promotion(
        _report(horizon_keys=("5", "10")), selected_horizon=SELECTED_HORIZON
    )
    assert passed is False
    assert any("one-day horizon is missing" in r for r in reasons)
    assert any("selected horizon is missing" in r for r in reasons)


def test_rejects_when_the_selected_horizon_has_too_few_observations():
    passed, reasons = assess_server_promotion(_report(rows=40), selected_horizon=SELECTED_HORIZON)
    assert passed is False
    assert any("too few evaluated observations" in r for r in reasons)


def test_rejects_a_forecast_outside_the_observed_volatility_range():
    passed, reasons = assess_server_promotion(
        _report(),
        selected_horizon=SELECTED_HORIZON,
        close_values=CLOSES,
        predicted_cumulative_return=0.9,
    )
    assert passed is False
    assert any("volatility range" in r for r in reasons)


def test_non_finite_forecast_fails_closed():
    passed, _ = assess_server_promotion(
        _report(),
        selected_horizon=SELECTED_HORIZON,
        close_values=CLOSES,
        predicted_cumulative_return=float("nan"),
    )
    assert passed is False


def test_volatility_gate_is_skipped_when_no_forecast_is_supplied():
    passed, _ = assess_server_promotion(_report(), selected_horizon=SELECTED_HORIZON)
    assert passed is True
