"""Causality tests for CausalTCN and neural candidate models."""

import numpy as np
import torch

from research.volatility_forecasting.causal_models_v1 import (
    CausalTCNModel,
    ConstrainedEnsembleOptimizer,
)


def test_tcn_causality_by_perturbation():
    """Invariant test: modifying future inputs in a sequence MUST NOT alter intermediate causal outputs."""
    torch.manual_seed(42)
    in_features = 8
    seq_len = 50
    model = CausalTCNModel(in_features=in_features, forecast_days=7, mode="return")
    model.eval()

    # Original sequence
    x_orig = torch.randn(1, seq_len, in_features)

    # Let's inspect causal conv representation up to step t=30
    x_perm = x_orig.permute(0, 2, 1)
    with torch.no_grad():
        conv_full = model.network(x_perm)  # (1, C, seq_len)
        step_30_full = conv_full[:, :, 30].clone()

    # Perturb inputs strictly after t=30 (e.g. t=31..49)
    x_perturbed = x_orig.clone()
    x_perturbed[:, 31:, :] = x_perturbed[:, 31:, :] * 100.0 + 500.0
    x_perm_pert = x_perturbed.permute(0, 2, 1)

    with torch.no_grad():
        conv_pert = model.network(x_perm_pert)
        step_30_pert = conv_pert[:, :, 30]

    # Invariant: step 30 representation is 100% IDENTICAL despite future massive perturbation
    diff = torch.max(torch.abs(step_30_full - step_30_pert)).item()
    assert diff < 1e-6, (
        f"TCN is non-causal! Future perturbation leaked into t=30 with max diff {diff}"
    )


def test_volatility_head_monotonicity():
    """Verify that volatility mode outputs non-decreasing cumulative variance by construction."""
    model = CausalTCNModel(in_features=6, forecast_days=7, mode="volatility")
    x = torch.randn(5, 40, 6)
    out = model(x)
    assert out.shape == (5, 7)
    # Check non-decreasing across horizon dimensions
    diffs = out[:, 1:] - out[:, :-1]
    assert (diffs >= 0).all(), "Cumulative variance is not monotonically increasing!"


def test_constrained_ensemble_optimizer():
    """Verify ensemble weights sum to 1.0 and are non-negative."""
    rng = np.random.default_rng(42)
    targets = rng.normal(0.01, 0.05, size=100)
    # Model 1 is accurate, Model 2 is noisy
    m1 = targets + rng.normal(0, 0.01, size=100)
    m2 = targets + rng.normal(0, 0.08, size=100)
    preds = np.column_stack([m1, m2])

    weights = ConstrainedEnsembleOptimizer.fit_weights(preds, targets)
    assert len(weights) == 2
    assert (weights >= 0.0).all()
    assert np.isclose(np.sum(weights), 1.0)
    # Accurate model gets larger weight
    assert weights[0] > weights[1]
