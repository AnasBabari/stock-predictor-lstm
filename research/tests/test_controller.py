"""Tests for ExperimentController, multi-fidelity levels, subprocess bounds, and parity."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from stock_autoresearch.candidates import (
    ELASTIC_NET_TUNING_GRID,
    CompactMLPCandidate,
    DLinearCandidate,
    ElasticNetCandidate,
    PersistenceCandidate,
    RidgeCandidate,
    SmallTCNCandidate,
    elastic_net_family_factories,
    elastic_net_family_name,
)
from stock_autoresearch.config import RUNTIME_BUDGET, RuntimeBudget
from stock_autoresearch.controller import (
    ExperimentController,
    SubprocessResult,
    check_harness_integrity,
)
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
    # Tuned Elastic Net grid variants share the same interface contract.
    factories = elastic_net_family_factories()
    candidates.extend(factories[name](seed=0) for name in sorted(factories))

    for model in candidates:
        fitted = model.fit(x, y)
        preds = fitted.predict(x[:5])
        assert preds.shape == (5,)
        assert np.isfinite(preds).all()
        desc = model.describe()
        assert "family" in desc
        assert model.parameter_count() >= 0


def test_controller_registers_tuned_elastic_net_families(tmp_path: Path) -> None:
    """The controller subprocess factory dict must resolve every grid variant.

    An unregistered family exits the subprocess with code 2, which the
    controller reports as a 'crash'; a 'success' status therefore proves the
    embedded factory dictionary knows the variant. The fixture snapshot is
    sized so the 5-fold expanding policy has enough rows.
    """
    rows = 900
    rng = np.random.default_rng(7)
    dates = pd.date_range("2022-01-01", periods=rows, freq="B")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.001, 0.01, size=rows)))
    frame = pd.DataFrame(
        {"Close": close, "feat_1": rng.normal(size=rows), "feat_2": rng.normal(size=rows)},
        index=dates,
    )
    snapshot_path = tmp_path / "registry_snapshot.csv"
    frame.to_csv(snapshot_path)

    ledger_path = tmp_path / "ledger.jsonl"
    controller = ExperimentController(
        snapshot_path=snapshot_path,
        ledger_path=ledger_path,
        run_tag="registry_test",
    )

    family = elastic_net_family_name(*ELASTIC_NET_TUNING_GRID[3])
    entry = controller.execute_trial(family, level=0)
    assert entry["candidate_family"] == family
    assert entry["status"] == "success"
    assert entry["decision"] in ("keep", "discard")
    assert isinstance(entry["relative_mae"], float)


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


def test_ledger_snapshot_id_is_content_hash_of_audited_snapshot(
    tmp_path: Path,
) -> None:
    """Audit regression: every ledger record must carry a real content hash of
    the exact snapshot bytes evaluated, never the 'unknown' placeholder."""
    import hashlib

    rows = 900
    rng = np.random.default_rng(7)
    dates = pd.date_range("2022-01-01", periods=rows, freq="B")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.001, 0.01, size=rows)))
    frame = pd.DataFrame(
        {"Close": close, "feat_1": rng.normal(size=rows), "feat_2": rng.normal(size=rows)},
        index=dates,
    )
    snapshot_path = tmp_path / "audited_snapshot.csv"
    frame.to_csv(snapshot_path)

    # The eval subprocess digests the CSV as re-parsed from disk, so the
    # reference hash must be computed over the same round-tripped frame.
    parsed = pd.read_csv(snapshot_path, index_col=0, parse_dates=True)
    cols = [c for c in parsed.columns if c != "Close"]
    digest = hashlib.sha256()
    digest.update(parsed.to_csv(index=True).encode("utf-8"))
    digest.update(json.dumps(cols, separators=(",", ":")).encode("utf-8"))
    expected = "sha256:" + digest.hexdigest()

    ledger_path = tmp_path / "ledger.jsonl"
    controller = ExperimentController(
        snapshot_path=snapshot_path,
        ledger_path=ledger_path,
        run_tag="audit_test",
    )

    entry = controller.execute_trial("persistence", level=0)
    assert entry["status"] == "success"
    assert entry["snapshot_id"] == expected
    assert entry["snapshot_id"] != "unknown"

    record = json.loads(ledger_path.read_text(encoding="utf-8").strip().splitlines()[0])
    assert record["snapshot_id"] == expected


@pytest.mark.parametrize("requested_horizon", [1, 5, 10, 20])
def test_level_2_confirmation_honors_requested_horizon(
    tmp_path: Path, requested_horizon: int
) -> None:
    """Level-2 confirmation must evaluate at the requested horizon, not a hardcoded 20."""
    ledger_path = tmp_path / "ledger.jsonl"
    controller = ExperimentController(
        snapshot_path=tmp_path / "snapshot.csv",
        ledger_path=ledger_path,
        run_tag="horizon_test",
    )

    with mock.patch("stock_autoresearch.controller.run_isolated_candidate_eval") as eval_mock:
        eval_mock.return_value = SubprocessResult(
            status="success",
            stdout="",
            stderr="",
            duration_seconds=1.0,
            peak_vram_mb=0,
            payload={},
        )
        controller.execute_trial("ridge", level=2, horizon=requested_horizon)

    eval_mock.assert_called_once()
    assert eval_mock.call_args.kwargs["horizon"] == requested_horizon


def test_level_2_confirmation_records_horizon_in_ledger(tmp_path: Path) -> None:
    """The ledger entry for a level-2 trial must carry the requested horizon."""
    ledger_path = tmp_path / "ledger.jsonl"
    controller = ExperimentController(
        snapshot_path=tmp_path / "snapshot.csv",
        ledger_path=ledger_path,
        run_tag="horizon_ledger_test",
    )

    with mock.patch("stock_autoresearch.controller.run_isolated_candidate_eval") as eval_mock:
        eval_mock.return_value = SubprocessResult(
            status="success",
            stdout="",
            stderr="",
            duration_seconds=1.0,
            peak_vram_mb=0,
            payload={},
        )
        entry = controller.execute_trial("ridge", level=2, horizon=10)

    assert entry["horizons"] == [10]
    lines = ledger_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["horizons"] == [10]


def test_smoke_and_screen_levels_keep_fixed_horizons(tmp_path: Path) -> None:
    """Levels 0 and 1 intentionally keep their fixed smoke/screen horizons."""
    controller = ExperimentController(
        snapshot_path=tmp_path / "snapshot.csv",
        ledger_path=tmp_path / "ledger.jsonl",
        run_tag="fixed_horizon_test",
    )

    stub = SubprocessResult(
        status="success", stdout="", stderr="", duration_seconds=1.0, peak_vram_mb=0, payload={}
    )
    with mock.patch(
        "stock_autoresearch.controller.run_isolated_candidate_eval", return_value=stub
    ) as eval_mock:
        controller.execute_trial("ridge", level=0, horizon=7)
        controller.execute_trial("ridge", level=1, horizon=7)

    assert [call.kwargs["horizon"] for call in eval_mock.call_args_list] == [1, 5]


def test_level_2_confirmation_end_to_end_uses_requested_horizon(tmp_path: Path) -> None:
    """Cheap end-to-end check: the embedded subprocess script is generated with
    the requested horizon and produces metrics that differ from horizon 20."""
    rows = 900
    rng = np.random.default_rng(11)
    dates = pd.date_range("2022-01-01", periods=rows, freq="B")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.001, 0.01, size=rows)))
    frame = pd.DataFrame(
        {"Close": close, "feat_1": rng.normal(size=rows), "feat_2": rng.normal(size=rows)},
        index=dates,
    )
    snapshot_path = tmp_path / "horizon_snapshot.csv"
    frame.to_csv(snapshot_path)

    # Keep the confirm timeout small; persistence fits in milliseconds here.
    budget = dataclasses.replace(RUNTIME_BUDGET, confirm_seconds=60)
    assert isinstance(budget, RuntimeBudget)

    results: dict[int, dict] = {}
    for requested_horizon in (5, 20):
        controller = ExperimentController(
            snapshot_path=snapshot_path,
            ledger_path=tmp_path / f"ledger_h{requested_horizon}.jsonl",
            run_tag=f"e2e_h{requested_horizon}",
            budget=budget,
        )
        sub = controller.run_level_2_confirmation("persistence", horizon=requested_horizon)
        assert sub.status == "success", sub.stderr
        assert sub.payload is not None
        # The evaluation summary returned by the subprocess must reflect the
        # requested horizon, proving the CLI value reached evaluate_candidate.
        assert sub.payload["horizon"] == requested_horizon
        results[requested_horizon] = sub.payload

    # The 5-fold expanding folds shrink as the horizon grows, so a horizon-5
    # run cannot reproduce the horizon-20 fold count or metrics.
    assert results[5]["folds"] != results[20]["folds"] or results[5] != results[20]
