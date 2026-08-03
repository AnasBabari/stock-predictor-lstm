import numpy as np

from experiments.runner import ExperimentConfig, run_baseline_experiment


def test_walk_forward_runner_scores_all_models_on_identical_folds():
    close = np.arange(1.0, 181.0)
    features = np.column_stack([close, np.ones_like(close)])
    report = run_baseline_experiment(
        features,
        close,
        feature_names=["Close", "Constant"],
        config=ExperimentConfig(
            lookback=5,
            horizons=(1, 3),
            folds=2,
            min_train_size=50,
            validation_size=20,
            gap=3,
        ),
    )

    assert set(report["models"]) == {
        "persistence",
        "drift",
        "ridge",
        "elastic_net",
        "hist_gradient_boosting",
    }
    for model_report in report["models"].values():
        assert len(model_report["folds"]) == 2
        assert model_report["aggregate"]["horizons"] == [1, 3]
        assert model_report["aggregate"]["pooled"]["sample_count"] == 80

    assert report["models"]["drift"]["aggregate"]["pooled"]["rmse"] == 0
    assert report["models"]["drift"]["promotion"]["promoted"]
    assert not report["models"]["persistence"]["promotion"]["promoted"]


class _DummyCandidate:
    name = "dummy_neural"

    def fit(self, features, targets, *, validation_data=None):
        assert validation_data is not None
        self.width = np.asarray(targets).shape[1]
        return self

    def predict(self, features):
        return np.zeros((len(features), self.width))


class _RefittingCandidate(_DummyCandidate):
    name = "refitting_neural"
    refit_calls = []

    def refit(self, features, targets):
        self.refit_rows = len(features)
        self.refit_calls.append(self.refit_rows)
        self.width = np.asarray(targets).shape[1]
        return self


def test_runner_places_extra_candidate_on_shared_outer_folds():
    close = np.arange(1.0, 181.0)
    report = run_baseline_experiment(
        np.column_stack([close, np.ones_like(close)]),
        close,
        feature_names=["Close", "Constant"],
        config=ExperimentConfig(
            lookback=5, horizons=(1, 3), folds=2, min_train_size=50, validation_size=20
        ),
        candidate_factories=(_DummyCandidate,),
    )
    assert len(report["models"]["dummy_neural"]["folds"]) == 2
    assert "evidence" in report["models"]["dummy_neural"]


def test_runner_supports_selection_then_full_outer_refit():
    _RefittingCandidate.refit_calls.clear()
    close = np.arange(1.0, 181.0)
    report = run_baseline_experiment(
        np.column_stack([close, np.ones_like(close)]),
        close,
        feature_names=["Close", "Constant"],
        config=ExperimentConfig(
            lookback=5,
            horizons=(1, 3),
            folds=2,
            min_train_size=50,
            validation_size=20,
        ),
        candidate_factories=(_RefittingCandidate,),
    )
    assert len(report["models"]["refitting_neural"]["folds"]) == 2
    assert len(_RefittingCandidate.refit_calls) == 2
    assert all(rows >= 50 for rows in _RefittingCandidate.refit_calls)
