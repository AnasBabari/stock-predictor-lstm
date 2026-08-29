"""Tests for GPU harness status and training execution."""

from __future__ import annotations

import numpy as np

from research.volatility_forecasting.gpu_harness_v10 import (
    TrainingExecutionConfig,
    check_gpu_runtime,
    cleanup_gpu_memory,
    train_candidate_fold,
)


def test_gpu_runtime_check_and_cleanup_run_safely() -> None:
    status = check_gpu_runtime()
    assert isinstance(status.cuda_available, bool)
    assert isinstance(status.device_name, str)
    cleanup_gpu_memory()


def test_train_candidate_fold_runs_classical_and_neural_candidates() -> None:
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(60, 26))
    y_train = np.maximum(0.0004 + 0.0001 * X_train[:, 0], 1e-6)
    X_val = rng.normal(size=(20, 26))
    y_val = np.maximum(0.0004 + 0.0001 * X_val[:, 0], 1e-6)

    # 1. Ridge
    cfg_ridge = TrainingExecutionConfig(candidate_family="ridge", horizon=1, fold_idx=0)
    res_ridge = train_candidate_fold(cfg_ridge, X_train, y_train, X_val, y_val)
    assert res_ridge["status"] == "success"
    assert res_ridge["val_qlike"] > 0.0

    # 2. ElasticNet
    cfg_enet = TrainingExecutionConfig(candidate_family="elasticnet", horizon=3, fold_idx=0)
    res_enet = train_candidate_fold(cfg_enet, X_train, y_train, X_val, y_val)
    assert res_enet["status"] == "success"
    assert res_enet["val_qlike"] > 0.0

    # 3. Neural (TCN/LSTM)
    cfg_tcn = TrainingExecutionConfig(candidate_family="tcn", horizon=1, fold_idx=0, max_epochs=5)
    res_tcn = train_candidate_fold(cfg_tcn, X_train, y_train, X_val, y_val)
    assert res_tcn["status"] in ("success", "fallback_success")
    assert res_tcn["val_qlike"] > 0.0


def test_select_champions_by_horizon_role_naming() -> None:
    from research.volatility_forecasting.horizon_selection_v10 import (
        select_champions_by_horizon,
    )

    ledger = [
        {"horizon": 1, "family": "tcn", "relative_qlike": 0.90, "ratio_upper_95": 0.98},
        {"horizon": 3, "family": "tcn", "relative_qlike": 1.15, "ratio_upper_95": 1.25},
    ]
    champions = select_champions_by_horizon(ledger, horizons=[1, 3], baseline_family="har")
    assert champions[1]["champion_family"] == "tcn"
    assert champions[1]["role"] == "learned_candidate"

    # Degraded horizon falls back to development_baseline_candidate
    assert champions[3]["champion_family"] == "har"
    assert champions[3]["role"] == "development_baseline_candidate"
