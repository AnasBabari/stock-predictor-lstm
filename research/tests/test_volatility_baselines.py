from __future__ import annotations

import numpy as np
from volatility_forecasting.baselines import (
    fit_adaptive_variance_baseline,
    predict_adaptive_variance_baseline,
    variance_baseline_candidates,
)
from volatility_forecasting.data import VolatilityPanelExamples


def _examples(rows: int = 80) -> VolatilityPanelExamples:
    horizons = (1, 3)
    names = ("EWMA_Var", "Vol_C2C_5", "Vol_C2C_20", "Vol_C2C_60")
    features = np.ones((rows, 2, len(names)), dtype=np.float32)
    features[:, -1, 0] = 0.01
    features[:, -1, 1:] = 0.1
    har = np.full((rows, len(horizons)), (0.02, 0.06), dtype=np.float32)
    realized = np.full((rows, len(horizons)), (0.01, 0.03), dtype=np.float32)
    return VolatilityPanelExamples(
        features=features,
        baseline_variance=har,
        realized_variance=realized,
        cumulative_returns=np.zeros_like(realized),
        direction_classes=np.ones_like(realized, dtype=np.int64),
        tickers=np.full(rows, "AAA"),
        origin_dates=np.datetime64("2025-01-01") + np.arange(rows).astype("timedelta64[D]"),
        origin_closes=np.full(rows, 100.0),
        horizons=horizons,
        feature_names=names,
    )


def test_candidate_baselines_use_only_the_origin_feature_row() -> None:
    examples = _examples()
    candidates = variance_baseline_candidates(examples)
    np.testing.assert_allclose(candidates["riskmetrics_ewma_c2c"][:3], [[0.01, 0.03]] * 3)
    np.testing.assert_allclose(candidates["rolling_c2c_20"][:3], [[0.01, 0.03]] * 3)


def test_adaptive_baseline_selection_is_frozen_before_prediction() -> None:
    examples = _examples()
    calibration = np.arange(0, 40)
    evaluation = np.arange(40, 80)
    selection = fit_adaptive_variance_baseline(examples, calibration)
    predicted = predict_adaptive_variance_baseline(examples, evaluation, selection)
    np.testing.assert_allclose(predicted, examples.realized_variance[evaluation], rtol=1e-6)

    # Changing evaluation outcomes cannot alter the pre-fitted selection or
    # its forecast because prediction consumes features and frozen parameters.
    changed = _examples()
    changed.realized_variance[evaluation] *= 100.0
    repeated = predict_adaptive_variance_baseline(changed, evaluation, selection)
    np.testing.assert_allclose(repeated, predicted)


def test_missing_candidate_features_fail_safely_to_calibrated_har() -> None:
    examples = _examples()
    stripped = VolatilityPanelExamples(
        features=np.ones((80, 2, 1), dtype=np.float32),
        baseline_variance=examples.baseline_variance,
        realized_variance=examples.realized_variance,
        cumulative_returns=examples.cumulative_returns,
        direction_classes=examples.direction_classes,
        tickers=examples.tickers,
        origin_dates=examples.origin_dates,
        origin_closes=examples.origin_closes,
        horizons=examples.horizons,
        feature_names=("other",),
    )
    selection = fit_adaptive_variance_baseline(stripped, np.arange(40))
    assert all(item.family == "causal_log_har" for item in selection.horizons)
    forecast = predict_adaptive_variance_baseline(stripped, np.arange(40, 80), selection)
    np.testing.assert_allclose(forecast, stripped.realized_variance[40:80], rtol=1e-6)
