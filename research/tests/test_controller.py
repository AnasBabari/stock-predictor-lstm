"""Tests for ExperimentController, multi-fidelity levels, subprocess bounds, and parity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stock_autoresearch.candidates import (
    CompactMLPCandidate,
    DLinearCandidate,
    ElasticNetCandidate,
    PersistenceCandidate,
    RidgeCandidate,
    SmallTCNCandidate,
)
from stock_autoresearch.config import EVALUATION_POLICY, RUNTIME_BUDGET
from stock_autoresearch.controller import ExperimentController, check_harness_integrity
from stock_autoresearch.data import Snapshot
from stock_autoresearch.parity import make_parity_fixture, verify_prediction_parity


@pytest.fixture
def sample_snapshot_csv(tmp_path: Path) -> Path:
    rows = 500
    dates = pd.date_range("2022-01-01", periods=rows, freq="B")
    close = 100.0 * np.exp(np.cumsum(np.random.default_rng(42).normal(0.001, 0.01, size=rows)))
    feat_1 = np.random.default_rng(42).normal(size=rows)
    feat_2 = np.random.default_rng(43).normal(size=rows)

    df = pd.DataFrame({"Close": close, "feat_1": feat_1, "feat_2": feat_2}, index=dates)
    csv_path = tmp_path / "snapshot.csv"
    df.to_csv(csv_path)
    return csv_path


def test_harness_integrity_check(tmp_path: Path) -> None:
    assert check_harness_integrity(tmp_path) is True


def test_parity_verification() -> None:
    fixture = make_parity_fixture(samples=5, window=60, features=28)
    assert fixture.shape == (5, 60, 28)

    py_preds = np.array([0.01, 0.02, -0.01], dtype=np.float32)
    tf_preds = np.array([0.0101, 0.0199, -0.0102], dtype=np.float32)

    res = verify_prediction_parity(py_preds, tf_preds, tolerance=1e-3)
    assert res.passed is True
    assert res.max_abs_diff < 1e-3

    bad_preds = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    bad_res = verify_prediction_parity(py_preds, bad_preds, tolerance=1e-3)
    assert bad_res.passed is False


def test_all_candidates_implement_interface() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 60, 4)).astype(np.float32)
    y = rng.normal(size=20).astype(np.float32)

    candidates = [
        PersistenceCandidate(),
        RidgeCandidate(alpha=5.0),
        ElasticNetCandidate(alpha=1.0, l1_ratio=0.5),
        CompactMLPCandidate(max_iter=10),
        DLinearCandidate(kernel_size=3),
        SmallTCNCandidate(channels=8),
    ]

    for model in candidates:
        fitted = model.fit(x, y)
        preds = fitted.predict(x[:5])
        assert preds.shape == (5,)
        assert np.isfinite(preds).all()
        desc = model.describe()
        assert "family" in desc
        assert model.parameter_count() >= 0


def test_controller_trial_execution(sample_snapshot_csv: Path, tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    controller = ExperimentController(
        snapshot_path=sample_snapshot_csv,
        ledger_path=ledger_path,
        run_tag="test_run",
    )

    entry = controller.execute_trial("persistence", level=0)
    assert entry["candidate_family"] == "persistence"
    assert entry["run_tag"] == "test_run"
    assert ledger_path.exists()

    # Check ledger contains JSON line
    lines = ledger_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["candidate_family"] == "persistence"
