"""Unit tests for PlausibilityAbstentionGate."""

from backend.evaluation.abstention_gate import PlausibilityAbstentionGate


def test_gate_promotes_well_behaved_model():
    gate = PlausibilityAbstentionGate()
    res = gate.evaluate(
        predicted_day1_log_return=-0.005,
        predicted_day1_volatility=0.015,
        candidate_day1_returns=[-0.006, -0.004, -0.005],
        relative_loss_vs_baseline=0.88,
        coverage_80pct=0.82,
    )
    assert res.is_promoted is True
    assert res.decision == "promoted_model"


def test_gate_rejects_bp_disagreement_anomaly():
    gate = PlausibilityAbstentionGate()
    # Flawed BP scratch predictions: TCN -6.2%, Deep LSTM -12.9%, Attn-LSTM -21.6%
    bp_returns = [-0.062, -0.129, -0.216]
    res = gate.evaluate(
        predicted_day1_log_return=-0.1358,
        predicted_day1_volatility=0.018,
        candidate_day1_returns=bp_returns,
        relative_loss_vs_baseline=0.95,
        coverage_80pct=0.75,
    )
    assert res.is_promoted is False
    assert res.decision == "abstain_model_disagreement"
    assert res.model_disagreement_pct > 15.0


def test_gate_rejects_extreme_jump_without_catalyst():
    gate = PlausibilityAbstentionGate(max_jump_score=3.5)
    # 6-sigma jump without catalyst
    res = gate.evaluate(
        predicted_day1_log_return=0.09,
        predicted_day1_volatility=0.015,
        candidate_day1_returns=[0.09, 0.088],
        relative_loss_vs_baseline=0.85,
        coverage_80pct=0.80,
        has_confirmed_catalyst=False,
    )
    assert res.is_promoted is False
    assert res.decision == "abstain_extreme_unconfirmed"
    assert res.jump_score == 6.0


def test_gate_rejects_failed_baseline():
    gate = PlausibilityAbstentionGate()
    res = gate.evaluate(
        predicted_day1_log_return=0.002,
        predicted_day1_volatility=0.012,
        relative_loss_vs_baseline=1.05,
    )
    assert res.is_promoted is False
    assert res.decision == "abstain_failed_baseline_gate"
