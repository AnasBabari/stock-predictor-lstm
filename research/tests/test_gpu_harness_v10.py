"""Tests for GPU harness status and independent per-horizon selection."""

from __future__ import annotations

from research.volatility_forecasting.gpu_harness_v10 import (
    check_gpu_runtime,
    cleanup_gpu_memory,
)
from research.volatility_forecasting.horizon_selection_v10 import (
    select_champions_by_horizon,
)


def test_gpu_runtime_check_and_cleanup_run_safely() -> None:
    status = check_gpu_runtime()
    assert isinstance(status.cuda_available, bool)
    assert isinstance(status.device_name, str)
    cleanup_gpu_memory()


def test_select_champions_by_horizon_selects_learned_when_eligible_and_baseline_otherwise() -> None:
    # Synthetic ledger where TCN wins on h=1, but fails on h=3 and h=5
    ledger = [
        # h=1 records
        {"horizon": 1, "family": "tcn", "relative_qlike": 0.90, "ratio_upper_95": 0.98},
        {"horizon": 1, "family": "gru", "relative_qlike": 0.95, "ratio_upper_95": 0.99},
        {"horizon": 1, "family": "har", "relative_qlike": 1.00, "ratio_upper_95": 1.00},
        # h=3 records: TCN degrades, ElasticNet is close, none beat baseline upper 95
        {"horizon": 3, "family": "tcn", "relative_qlike": 1.05, "ratio_upper_95": 1.15},
        {"horizon": 3, "family": "har", "relative_qlike": 1.00, "ratio_upper_95": 1.00},
        # h=5 records: all neural degrade
        {"horizon": 5, "family": "tcn", "relative_qlike": 1.18, "ratio_upper_95": 1.30},
        {"horizon": 5, "family": "har", "relative_qlike": 1.00, "ratio_upper_95": 1.00},
    ]

    champions = select_champions_by_horizon(ledger, horizons=[1, 3, 5], baseline_family="har")

    assert champions[1]["champion_family"] == "tcn"
    assert champions[1]["role"] == "learned_candidate"

    assert champions[3]["champion_family"] == "har"
    assert champions[3]["role"] == "certified_baseline_fallback"

    assert champions[5]["champion_family"] == "har"
    assert champions[5]["role"] == "certified_baseline_fallback"
