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
        "hist_gradient_boosting",
    }
    for model_report in report["models"].values():
        assert len(model_report["folds"]) == 2
        assert model_report["aggregate"]["horizons"] == [1, 3]
        assert model_report["aggregate"]["pooled"]["sample_count"] == 80

    assert report["models"]["drift"]["aggregate"]["pooled"]["rmse"] == 0
    assert report["models"]["drift"]["promotion"]["promoted"]
    assert not report["models"]["persistence"]["promotion"]["promoted"]
