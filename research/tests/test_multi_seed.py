from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from stock_autoresearch import multi_seed
from stock_autoresearch.candidates import PersistenceCandidate, RidgeCandidate
from stock_autoresearch.config import EvaluationPolicy
from stock_autoresearch.data import Snapshot
from stock_autoresearch.multi_seed import evaluate_multi_seed


def make_snapshot(rows: int = 700) -> Snapshot:
    index = pd.date_range("2020-01-01", periods=rows, freq="B")
    close = 100.0 * np.exp(np.cumsum(np.full(rows, 0.001)))
    frame = pd.DataFrame(
        {"Close": close, "feature_a": np.linspace(0.0, 1.0, rows)},
        index=index,
    )
    return Snapshot(frame=frame, snapshot_id="test-snapshot", feature_names=("feature_a",))


def test_multi_seed_defaults_to_policy_seed_count() -> None:
    policy = EvaluationPolicy()
    record = evaluate_multi_seed(
        make_snapshot(),
        lambda seed: PersistenceCandidate(),
        horizon=5,
        policy=policy,
    )
    assert record["seeds"] == list(range(policy.seed_count))
    assert record["failure_count"] == 0
    assert record["status"] == "success"
    assert len(record["per_seed"]) == policy.seed_count

    for metric in (
        "median_relative_mae",
        "median_relative_rmse",
        "worst_fold_relative_rmse",
        "folds_beating_persistence",
    ):
        aggregate = record["seed_aggregate"][metric]
        assert set(aggregate) == {"mean", "median", "std", "best", "worst"}
        for value in aggregate.values():
            assert value is not None
            assert np.isfinite(value)
        assert aggregate["best"] <= aggregate["median"] <= aggregate["worst"]

    # Persistence predicts exactly the origin, so relative MAE is one per seed.
    assert record["seed_aggregate"]["median_relative_mae"]["median"] == pytest.approx(1.0)
    assert record["promotable_seed_count"] == 0
    assert record["promotable"] is False
    # Ledger-compatible scalar fields are present at the top level.
    assert (
        record["median_relative_mae"] == record["seed_aggregate"]["median_relative_mae"]["median"]
    )


def test_multi_seed_counts_failing_seed() -> None:
    def flaky_factory(seed: int):
        if seed == 1:
            raise RuntimeError("seed 1 always fails")
        return PersistenceCandidate()

    record = evaluate_multi_seed(
        make_snapshot(),
        flaky_factory,
        horizon=5,
        policy=EvaluationPolicy(),
        seeds=(0, 1, 2),
    )
    assert record["seeds"] == [0, 1, 2]
    assert record["failure_count"] == 1
    assert record["status"] == "partial"
    statuses = [entry["status"] for entry in record["per_seed"]]
    assert statuses == ["success", "failed", "success"]
    assert record["seed_aggregate"]["median_relative_rmse"]["mean"] == pytest.approx(1.0)


def test_multi_seed_all_seeds_failed() -> None:
    def broken_factory(seed: int):
        raise TimeoutError("simulated timeout")

    record = evaluate_multi_seed(
        make_snapshot(),
        broken_factory,
        horizon=5,
        policy=EvaluationPolicy(),
        seeds=(0, 1),
    )
    assert record["failure_count"] == 2
    assert record["status"] == "crash"
    assert record["promotable"] is False
    for metric in ("median_relative_mae", "median_relative_rmse"):
        assert record["seed_aggregate"][metric] == {
            "mean": None,
            "median": None,
            "std": None,
            "best": None,
            "worst": None,
        }


def test_multi_seed_delegates_to_locked_evaluate_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    locked = multi_seed.evaluate_candidate

    def counting_evaluate(snapshot, candidate_factory, *, horizon, policy, seed=0):
        calls.append(seed)
        return locked(snapshot, candidate_factory, horizon=horizon, policy=policy, seed=seed)

    monkeypatch.setattr(multi_seed, "evaluate_candidate", counting_evaluate)
    record = evaluate_multi_seed(
        make_snapshot(),
        lambda seed: PersistenceCandidate(),
        horizon=5,
        policy=EvaluationPolicy(),
        seeds=(3, 4),
    )
    # The locked evaluator is called once per seed and never modified.
    assert calls == [3, 4]
    assert record["failure_count"] == 0


def test_multi_seed_aggregate_respects_metric_direction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeResult:
        def __init__(self, beating: float):
            self._beating = beating

        def summary(self, policy):
            return {
                "median_relative_mae": 1.0 + 0.1 * self._beating,
                "median_relative_rmse": 1.0,
                "worst_fold_relative_rmse": 1.5,
                "folds_beating_persistence": self._beating,
                "promotable": False,
            }

    def fake_evaluate(snapshot, candidate_factory, *, horizon, policy, seed=0):
        return _FakeResult(float(seed))

    monkeypatch.setattr(multi_seed, "evaluate_candidate", fake_evaluate)
    record = evaluate_multi_seed(
        make_snapshot(),
        lambda seed: PersistenceCandidate(),
        horizon=5,
        policy=EvaluationPolicy(),
        seeds=(0, 1, 2),
    )
    # folds_beating_persistence is higher-is-better: best is the max seed
    # value and worst the min across heterogeneous per-seed summaries.
    beating = record["seed_aggregate"]["folds_beating_persistence"]
    assert beating["best"] == 2.0
    assert beating["worst"] == 0.0
    # Error ratios stay lower-is-better.
    mae = record["seed_aggregate"]["median_relative_mae"]
    assert mae["best"] == pytest.approx(1.0)
    assert mae["worst"] == pytest.approx(1.2)


def test_multi_seed_real_ridge_end_to_end() -> None:
    """Regression: exercise the locked evaluate_candidate through the wrapper.

    Reproduces the user-reported crash where the in-process multi-seed path
    failed on snapshots whose requested feature columns were absent. No mocks
    on evaluate_candidate; uses a real RidgeCandidate on a noisy trend.
    """
    rows = 700
    rng = np.random.default_rng(7)
    index = pd.date_range("2016-08-03", periods=rows, freq="B")
    log_close = np.log(100.0) + np.cumsum(rng.normal(0.0005, 0.01, rows))
    close = np.exp(log_close)
    ret_1 = np.concatenate([[0.0], np.diff(log_close)])
    vol_20 = pd.Series(ret_1).rolling(20, min_periods=2).std().fillna(0.0).to_numpy()
    frame = pd.DataFrame(
        {"Close": close, "ret_1": ret_1, "vol_20": vol_20},
        index=index,
    )
    snapshot = Snapshot(
        frame=frame,
        snapshot_id="regression-snapshot",
        feature_names=("ret_1", "vol_20"),
    )

    record = evaluate_multi_seed(
        snapshot,
        lambda seed: RidgeCandidate(),
        horizon=5,
        policy=EvaluationPolicy(),
        seeds=(0, 1),
    )

    assert record["seeds"] == [0, 1]
    assert record["failure_count"] == 0
    assert record["status"] == "success"
    assert record["failure_reason"] == ""
    for entry in record["per_seed"]:
        assert entry["status"] == "success"
        summary = entry["summary"]
        assert np.isfinite(float(summary["median_relative_mae"]))
        assert np.isfinite(float(summary["median_relative_rmse"]))

    # Ledger-compatible top-level aggregates are populated, not None.
    for metric in (
        "median_relative_mae",
        "median_relative_rmse",
        "worst_fold_relative_rmse",
        "folds_beating_persistence",
    ):
        assert record[metric] is not None
        assert np.isfinite(record[metric])
        assert record[metric] == record["seed_aggregate"][metric]["median"]
