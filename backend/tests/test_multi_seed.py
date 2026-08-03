"""Tests for multi-seed evaluation in the offline runner and seed aggregation."""

import numpy as np
import pytest

from evaluation.seeds import aggregate_seed_runs
from experiments.runner import ExperimentConfig, run_baseline_experiment

BASELINE_MODELS = {
    "persistence",
    "drift",
    "ridge",
    "elastic_net",
    "hist_gradient_boosting",
}


def _dataset():
    close = np.arange(1.0, 181.0)
    features = np.column_stack([close, np.ones_like(close)])
    return features, close


def _config(**overrides) -> ExperimentConfig:
    defaults = dict(
        lookback=5,
        horizons=(1, 3),
        folds=2,
        min_train_size=50,
        validation_size=20,
        gap=3,
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def test_single_seed_report_has_no_seed_summary():
    features, close = _dataset()
    report = run_baseline_experiment(
        features,
        close,
        feature_names=["Close", "Constant"],
        config=_config(),
    )
    assert set(report["models"]) == BASELINE_MODELS
    for model_report in report["models"].values():
        assert "seed_summary" not in model_report
        assert set(model_report) >= {"folds", "aggregate", "promotion"}


def test_multi_seed_adds_seed_summary_for_stochastic_models_only():
    features, close = _dataset()
    report = run_baseline_experiment(
        features,
        close,
        feature_names=["Close", "Constant"],
        config=_config(seeds=(1, 2, 3)),
    )
    assert set(report["models"]) == BASELINE_MODELS

    summary = report["models"]["hist_gradient_boosting"]["seed_summary"]
    assert summary["failure_count"] == 0
    for metric in ("relative_mae", "relative_rmse"):
        for key in ("mean", "median", "std", "best", "worst"):
            value = summary[metric][key]
            assert isinstance(value, float)
            assert np.isfinite(value)
        assert summary[metric]["best"] <= summary[metric]["median"] <= summary[metric]["worst"]

    for deterministic in ("persistence", "drift", "ridge", "elastic_net"):
        assert "seed_summary" not in report["models"][deterministic]
        assert len(report["models"][deterministic]["folds"]) == 2


def test_multi_seed_reports_first_seed_folds_and_aggregates():
    features, close = _dataset()
    multi = run_baseline_experiment(
        features,
        close,
        feature_names=["Close", "Constant"],
        config=_config(seeds=(1, 2, 3)),
    )
    first_seed_only = run_baseline_experiment(
        features,
        close,
        feature_names=["Close", "Constant"],
        config=_config(seed=1, seeds=(1,)),
    )
    # Reported folds/aggregate/evidence come from the FIRST seed.
    for model_name in BASELINE_MODELS:
        assert (
            multi["models"][model_name]["folds"] == first_seed_only["models"][model_name]["folds"]
        )
        assert (
            multi["models"][model_name]["aggregate"]
            == first_seed_only["models"][model_name]["aggregate"]
        )


class _FlakyCandidate:
    name = "flaky_neural"

    def fit(self, features, targets, *, validation_data=None):
        assert validation_data is not None
        self.width = np.asarray(targets).shape[1]
        return self

    def predict(self, features):
        return np.zeros((len(features), self.width))


def _flaky_factory(seed):
    if seed == 2:
        raise RuntimeError("seed 2 always fails")
    return _FlakyCandidate()


def test_multi_seed_counts_failing_candidate_seed():
    features, close = _dataset()
    report = run_baseline_experiment(
        features,
        close,
        feature_names=["Close", "Constant"],
        config=_config(seeds=(1, 2, 3)),
        candidate_factories=(_flaky_factory,),
    )
    flaky = report["models"]["flaky_neural"]
    assert flaky["seed_summary"]["failure_count"] == 1
    assert len(flaky["folds"]) == 2
    assert "evidence" in flaky
    # Two surviving seeds still produce finite aggregates.
    for metric in ("relative_mae", "relative_rmse"):
        assert np.isfinite(flaky["seed_summary"][metric]["median"])


class _FoldBoundFlakyCandidate:
    name = "fold_bound_flaky"

    def __init__(self, fail_on_fit: bool):
        self.fail_on_fit = fail_on_fit
        self.width = 2

    def fit(self, features, targets, *, validation_data=None):
        if self.fail_on_fit:
            raise RuntimeError("seed fails on this fold")
        assert validation_data is not None
        self.width = np.asarray(targets).shape[1]
        return self

    def predict(self, features):
        return np.zeros((len(features), self.width))


def test_multi_seed_survives_seed_failing_only_on_early_fold():
    # Regression: a seed that fails on fold 1 gets a None failure marker; if
    # it succeeds on a later fold the run must complete instead of aborting
    # on that late success.
    features, close = _dataset()
    fold_counts: dict[int, int] = {}

    def fold_flaky_factory(seed):
        fold_counts[seed] = fold_counts.get(seed, 0) + 1
        # Seed 2 fails on fold 1 only and succeeds on fold 2.
        return _FoldBoundFlakyCandidate(fail_on_fit=(seed == 2 and fold_counts[seed] == 1))

    report = run_baseline_experiment(
        features,
        close,
        feature_names=["Close", "Constant"],
        config=_config(seeds=(1, 2)),
        candidate_factories=(fold_flaky_factory,),
    )
    flaky = report["models"]["fold_bound_flaky"]
    assert len(flaky["folds"]) == 2
    summary = flaky["seed_summary"]
    assert summary["failure_count"] == 1
    for metric in ("relative_mae", "relative_rmse"):
        for key in ("mean", "median", "std", "best", "worst"):
            value = summary[metric][key]
            assert isinstance(value, float)
            assert np.isfinite(value)


def test_aggregate_seed_runs_computes_statistics():
    result = aggregate_seed_runs(
        [
            {"relative_mae": 1.0, "relative_rmse": 2.0},
            {"relative_mae": 3.0, "relative_rmse": 4.0},
        ]
    )
    assert result["failure_count"] == 0
    assert result["relative_mae"] == {
        "mean": 2.0,
        "median": 2.0,
        "std": 1.0,
        "best": 1.0,
        "worst": 3.0,
    }
    assert result["relative_rmse"]["mean"] == 3.0
    assert result["relative_rmse"]["best"] == 2.0
    assert result["relative_rmse"]["worst"] == 4.0


def test_aggregate_seed_runs_excludes_failed_entries():
    result = aggregate_seed_runs(
        [
            {"relative_mae": 1.0, "relative_rmse": 1.5},
            None,
            {"relative_mae": float("nan"), "relative_rmse": 2.0},
            {"relative_mae": 3.0, "relative_rmse": 2.5},
            {"relative_mae": 5.0},  # missing metric counts as a failure
        ]
    )
    assert result["failure_count"] == 3
    assert result["relative_mae"]["mean"] == 2.0
    assert result["relative_rmse"]["mean"] == 2.0


def test_aggregate_seed_runs_all_failed_returns_none_aggregates():
    result = aggregate_seed_runs([None, {"relative_mae": float("inf"), "relative_rmse": 1.0}])
    assert result["failure_count"] == 2
    for metric in ("relative_mae", "relative_rmse"):
        assert result[metric] == {
            "mean": None,
            "median": None,
            "std": None,
            "best": None,
            "worst": None,
        }


def test_aggregate_seed_runs_rejects_empty_input():
    with pytest.raises(ValueError):
        aggregate_seed_runs([])
