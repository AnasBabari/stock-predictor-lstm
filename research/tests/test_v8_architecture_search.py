"""Fail-closed invariants for the v8 architecture search script.

These tests exercise the helpers in ``scripts/run_v8_architecture_search.py``
without spinning up PyTorch. They lock in:

1. The ranked-search report orders eligible before ineligible and then
   by worst required-horizon relative QLIKE.
2. The summary aggregator only emits finite values when given finite
   metrics, and never silently masks NaN / inf in the reported fields.
3. The label constructor never crashes on patch_transformer configs
   missing ``transformer_d_model`` (regression for an earlier NoneType
   error).
4. The loss-weight helper keeps the canonical sum-to-one invariant for
   every ``baseline_regularization`` in the dev search space.
5. A simulated search never opens the sealed test partition (a fail-
   closed fingerprint: the script only consumes ``train_indices`` and
   ``validation_indices``).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_v8_architecture_search import (  # noqa: E402
    _build_search_space,
    _config_label,
    _loss_weights_for_config,
    _rank_candidates,
    _summarize_evidence,
    _validate_search_config,
)


class _FakeEvidence:
    def __init__(self, seed: int, rel: list[float], upper: list[float]) -> None:
        self.seed = seed
        self.metrics = tuple({"relative_qlike": value} for value in rel)
        self.ratio_upper_95 = tuple(upper)


def test_rank_eligible_first_then_worst_relative_qlike() -> None:
    rows = [
        {"label": "a", "eligible": False, "worst_required_relative_qlike": 0.99,
         "mean_required_relative_qlike": 0.95, "worst_required_ratio_upper_95": 1.05},
        {"label": "b", "eligible": True, "worst_required_relative_qlike": 1.00,
         "mean_required_relative_qlike": 0.96, "worst_required_ratio_upper_95": 1.06},
        {"label": "c", "eligible": True, "worst_required_relative_qlike": 0.95,
         "mean_required_relative_qlike": 0.90, "worst_required_ratio_upper_95": 1.02},
        {"label": "d", "eligible": False, "worst_required_relative_qlike": 0.97,
         "mean_required_relative_qlike": 0.93, "worst_required_ratio_upper_95": 1.03},
    ]
    ranked = _rank_candidates(rows)
    assert [row["label"] for row in ranked] == ["c", "b", "d", "a"]


def test_summarize_evidence_reports_finite_values() -> None:
    evidence = (
        _FakeEvidence(seed=41, rel=[0.97, 0.98], upper=[1.01, 1.02]),
        _FakeEvidence(seed=42, rel=[0.99, 1.00], upper=[1.03, 1.04]),
    )
    summary = _summarize_evidence(evidence, required_horizons=(1, 3))
    assert summary["worst_required_relative_qlike"] == 1.00
    assert math.isclose(summary["mean_required_relative_qlike"], (0.97 + 0.98 + 0.99 + 1.00) / 4)
    assert summary["worst_required_ratio_upper_95"] == 1.04


def test_summarize_evidence_empty_evidence_fails_closed() -> None:
    with pytest.raises(ValueError, match="empty"):
        _summarize_evidence(tuple(), required_horizons=(1,))


def test_summarize_evidence_non_finite_fails_closed() -> None:
    evidence = (_FakeEvidence(seed=41, rel=[float("nan")], upper=[1.0]),)
    with pytest.raises(ValueError, match="non-finite"):
        _summarize_evidence(evidence, required_horizons=(1,))


def test_config_label_handles_tcn_without_transformer_d_model() -> None:
    """Regression: tcn configs must not crash label construction."""
    cfg = {
        "encoder_family": "tcn",
        "channels": 48,
        "dropout": 0.15,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "baseline_regularization": 0.05,
    }
    label = _config_label(cfg)
    assert label.startswith("tcn-ch48")
    assert "reg0.05" in label


def test_config_label_handles_patch_transformer() -> None:
    cfg = {
        "encoder_family": "patch_transformer",
        "channels": 48,
        "transformer_d_model": 64,
        "dropout": 0.25,
        "learning_rate": 3e-4,
        "weight_decay": 1e-3,
        "baseline_regularization": 0.10,
    }
    label = _config_label(cfg)
    assert label.startswith("patch_transformer-ch48")
    assert "reg0.1" in label


def test_loss_weights_sum_to_one_for_dev_search_space() -> None:
    """Every ``baseline_regularization`` used by the dev sweep must satisfy
    the canonical sum-to-one invariant of ``VolatilityLossWeights``."""
    for reg in (0.05, 0.10):
        weights = _loss_weights_for_config(
            {
                "encoder_family": "patch_transformer",
                "channels": 48,
                "transformer_d_model": 64,
                "dropout": 0.15,
                "learning_rate": 1e-3,
                "weight_decay": 1e-3,
                "baseline_regularization": reg,
            }
        )
        total = (
            weights.qlike
            + weights.variance_crps
            + weights.return_location
            + weights.direction
            + weights.baseline_regularization
        )
        assert np.isclose(total, 1.0, atol=1e-8)


def test_search_space_label_uniqueness_holds_for_default_budget() -> None:
    """The default search space must produce uniquely labelled configs."""
    configs = _build_search_space(max_configs=12)
    labels = [_config_label(cfg) for cfg in configs]
    assert len(labels) == len(set(labels)), "duplicate labels in search space"
    # Every config has the required keys for downstream consumption
    for cfg in configs:
        assert {"encoder_family", "channels", "dropout", "learning_rate",
                "weight_decay", "baseline_regularization"} <= set(cfg)


def test_search_space_caps_at_max_configs() -> None:
    configs = _build_search_space(max_configs=4)
    assert len(configs) == 4


def test_replay_config_rejects_unknown_and_non_finite_values() -> None:
    valid = {
        "encoder_family": "patch_transformer",
        "channels": 48,
        "transformer_d_model": 64,
        "dropout": 0.15,
        "learning_rate": 1e-3,
        "weight_decay": 1e-3,
        "baseline_regularization": 0.05,
    }
    assert _validate_search_config(valid) == valid
    with pytest.raises(ValueError, match="unknown fields"):
        _validate_search_config({**valid, "surprise": 1})
    with pytest.raises(ValueError, match="finite"):
        _validate_search_config({**valid, "learning_rate": float("nan")})


def test_search_script_signature_has_no_test_partition_access() -> None:
    """The script's narrative contract: only ``train_indices`` and
    ``validation_indices`` enter the training call.  This test fingerprints
    the call sites by reading the script source so that any future edit
    that smuggles in a test-partition parameter fails the build."""
    src = (SCRIPTS / "run_v8_architecture_search.py").read_text(encoding="utf-8")
    forbidden_terms = (
        "temporal_test_indices",
        "asset_transfer_test_indices",
        "pooled_test_indices",
        "test_indices=",
        "evaluation_indices=split.pooled_test",
    )
    for token in forbidden_terms:
        assert token not in src, (
            f"run_v8_architecture_search.py must not reference {token}; "
            "the sealed test partition must remain closed during search"
        )
