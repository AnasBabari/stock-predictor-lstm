import numpy as np
import pytest

from experiments.runner import ExperimentConfig, run_baseline_experiment

_BASELINE_MODELS = {
    "persistence",
    "drift",
    "ridge",
    "elastic_net",
    "hist_gradient_boosting",
}


def _run(**overrides) -> dict:
    close = np.arange(1.0, 181.0)
    features = np.column_stack([close, np.ones_like(close)])
    config = ExperimentConfig(
        lookback=5,
        horizons=(1, 3),
        folds=2,
        min_train_size=50,
        validation_size=20,
        **overrides,
    )
    return run_baseline_experiment(
        features,
        close,
        feature_names=["Close", "Constant"],
        config=config,
    )


def _run_noisy_trend(**overrides) -> dict:
    # Trending series with bounded noise, used for price-space quantile
    # diagnostics regression checks.
    rng = np.random.default_rng(7)
    close = 50.0 + np.arange(180) * 0.5 + rng.uniform(-1.0, 1.0, 180)
    features = np.column_stack([close, np.ones_like(close)])
    config = ExperimentConfig(
        lookback=5,
        horizons=(1, 3),
        folds=2,
        min_train_size=50,
        validation_size=20,
        **overrides,
    )
    return run_baseline_experiment(
        features,
        close,
        feature_names=["Close", "Constant"],
        config=config,
    )


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and np.isfinite(value)


def test_default_report_contains_per_horizon_evidence_and_intervals():
    report = _run()
    assert set(report["models"]) == _BASELINE_MODELS
    for model_name, model_report in report["models"].items():
        # Pre-existing report keys and semantics stay untouched.
        assert len(model_report["folds"]) == 2
        assert "aggregate" in model_report and "promotion" in model_report
        if model_name == "persistence":
            assert "evidence" not in model_report
            assert "evidence_by_horizon" not in model_report
            assert "intervals" not in model_report
            continue
        assert set(model_report["evidence"]) == {
            "loss",
            "mean_improvement",
            "confidence_interval",
            "dm_style_statistic",
            "two_sided_p_value",
            "sample_count",
        }
        horizon_evidence = model_report["evidence_by_horizon"]
        assert set(horizon_evidence) == {"1", "3"}
        for entry in horizon_evidence.values():
            assert set(entry) == {"absolute", "squared", "relative_mae", "relative_rmse"}
            for loss_name in ("absolute", "squared"):
                assert _finite(entry[loss_name]["two_sided_p_value"])
                assert entry[loss_name]["sample_count"] == 40
            for metric_name in ("relative_mae", "relative_rmse"):
                assert _finite(entry[metric_name]["ratio"])
        intervals = model_report["intervals"]
        assert intervals["confidence"] == 0.9
        assert set(intervals["radius"]) == {"1", "3"}
        assert all(_finite(radius) and radius >= 0 for radius in intervals["radius"].values())
        assert 0.0 <= intervals["empirical_coverage"] <= 1.0
        assert _finite(intervals["average_width"])
        assert intervals["sample_count"] == 40


def test_evidence_multiple_comparison_lists_every_collected_p_value():
    report = _run()
    multiple_comparison = report["evidence_multiple_comparison"]
    assert multiple_comparison["method"] == "benjamini_hochberg"
    assert multiple_comparison["q"] == 0.10
    decisions = multiple_comparison["decisions"]
    # 4 non-persistence models x 2 horizons x 2 paired losses.
    expected_count = (len(_BASELINE_MODELS) - 1) * 2 * 2
    assert len(decisions) == expected_count
    collected = [
        model_report["evidence_by_horizon"][str(decision["horizon"])][decision["loss"]][
            "two_sided_p_value"
        ]
        for decision in decisions
        for model_name, model_report in report["models"].items()
        if model_name == decision["model"]
    ]
    assert collected == [decision["p_value"] for decision in decisions]
    assert all(isinstance(decision["rejected"], bool) for decision in decisions)


def test_blends_are_opt_in_and_constrained():
    report = _run()
    assert "blend" not in report

    blended = _run(include_blends=True)
    blend = blended["blend"]
    assert set(blend) == {"shrinkage", "constrained"}
    member_names = _BASELINE_MODELS - {"persistence"}
    assert set(blend["shrinkage"]) == member_names
    for per_horizon in blend["shrinkage"].values():
        assert set(per_horizon) == {"1", "3"}
        for alpha in per_horizon.values():
            assert 0.0 <= alpha <= 1.0
    constrained = blend["constrained"]
    assert set(constrained["weights"]) == member_names
    assert all(weight >= 0.0 for weight in constrained["weights"].values())
    assert sum(constrained["weights"].values()) <= 1.0 + 1e-9
    assert constrained["held_out_fold"] == 2
    for metric in ("relative_mae", "relative_rmse"):
        assert set(constrained[metric]) == {"1", "3"}
        assert all(_finite(value) for value in constrained[metric].values())
    # Blending is diagnostic-only: promotion outcomes do not change.
    for model_name in member_names | {"persistence"}:
        assert (
            blended["models"][model_name]["promotion"] == report["models"][model_name]["promotion"]
        )


def test_quantile_forecaster_is_opt_in_with_diagnostics():
    report = _run()
    assert "quantile_hist_gradient_boosting" not in report["models"]

    quantiled = _run(include_quantiles=True)
    model_report = quantiled["models"]["quantile_hist_gradient_boosting"]
    assert len(model_report["folds"]) == 2
    assert model_report["aggregate"]["horizons"] == [1, 3]
    diagnostics = model_report["quantile_diagnostics"]
    for tau in ("0.05", "0.95"):
        assert set(diagnostics["pinball_loss"][tau]) == {"1", "3"}
        assert all(_finite(value) for value in diagnostics["pinball_loss"][tau].values())
    assert 0.0 <= diagnostics["quantile_crossing_rate"] <= 1.0
    assert 0.0 <= diagnostics["band_coverage"] <= 1.0
    assert diagnostics["sample_count"] == 40


def test_quantile_diagnostics_operate_on_price_scale():
    # Regression: diagnostics must compare price-space bands against price
    # actuals (target-space comparisons made band_coverage identically zero),
    # and pinball losses must sit on the price scale (finite and below the
    # pooled price range of the synthetic trend series).
    quantiled = _run_noisy_trend(include_quantiles=True)
    diagnostics = quantiled["models"]["quantile_hist_gradient_boosting"]["quantile_diagnostics"]
    assert diagnostics["band_coverage"] > 0.3
    assert diagnostics["band_quantiles"] == [0.05, 0.95]
    price_range = 140.0 - 50.0
    for per_horizon in diagnostics["pinball_loss"].values():
        for value in per_horizon.values():
            assert _finite(value)
            assert 0.0 <= value < price_range


def test_drift_diagnostics_are_opt_in_and_finite():
    report = _run()
    assert all("drift" not in model_report for model_report in report["models"].values())

    drifted = _run(include_drift=True)
    for model_name in _BASELINE_MODELS:
        drift = drifted["models"][model_name]["drift"]
        divergence = drift["feature_divergence"]
        assert len(divergence["psi_by_column"]) == 2
        assert _finite(divergence["max_psi"]) and _finite(divergence["mean_psi"])
        residual = drift["residual_drift"]
        assert _finite(residual["first_half_mae"]) and _finite(residual["second_half_mae"])
        assert _finite(residual["difference"])
        assert isinstance(residual["drift_detected"], bool)


def test_small_tcn_is_absent_by_default():
    report = _run()
    assert "small_tcn" not in report["models"]
    assert report["config"]["include_tcn"] is False
    assert set(report["models"]) == _BASELINE_MODELS


def test_hgb_excluded_when_include_hgb_false():
    report = _run(include_hgb=False)
    assert "hist_gradient_boosting" not in report["models"]
    assert report["config"]["include_hgb"] is False
    assert set(report["models"]) == _BASELINE_MODELS - {"hist_gradient_boosting"}


def test_small_tcn_joins_model_loop_when_enabled():
    pytest.importorskip("torch")
    report = _run(include_tcn=True)
    assert set(report["models"]) == _BASELINE_MODELS | {"small_tcn"}
    model_report = report["models"]["small_tcn"]
    assert len(model_report["folds"]) == 2
    assert model_report["aggregate"]["horizons"] == [1, 3]
    assert model_report["aggregate"]["pooled"]["sample_count"] == 80
    assert "promotion" in model_report and "evidence" in model_report
    for horizon in ("1", "3"):
        assert _finite(model_report["aggregate"]["per_horizon"][horizon]["relative_mae"])
        assert _finite(model_report["aggregate"]["per_horizon"][horizon]["relative_rmse"])


def test_default_config_report_keeps_pre_existing_keys():
    report = _run()
    assert set(report["config"]) >= {
        "lookback",
        "horizons",
        "target_type",
        "folds",
        "min_train_size",
        "validation_size",
        "gap",
        "method",
        "seed",
        "seeds",
        "effective_gap",
        "include_blends",
        "include_quantiles",
        "include_drift",
        "include_tcn",
    }
    assert report["config"]["seeds"] == (42,)
    assert set(report["dataset"]) == {
        "samples",
        "feature_count",
        "first_origin_index",
        "last_origin_index",
        "snapshot_id",
    }
    assert report["models"]["drift"]["aggregate"]["pooled"]["rmse"] == 0
    assert report["models"]["drift"]["promotion"]["promoted"]
