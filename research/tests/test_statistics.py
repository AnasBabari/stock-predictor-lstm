from __future__ import annotations

import numpy as np
import pytest
from stock_autoresearch.statistics import (
    block_bootstrap_interval,
    dm_style_statistic,
    fold_metric_evidence,
)


def test_block_bootstrap_interval_brackets_the_mean() -> None:
    rng = np.random.default_rng(0)
    values = 1.0 + np.cumsum(rng.normal(0.0, 0.1, 120))
    interval = block_bootstrap_interval(values, resamples=200, block_length=10, seed=3)
    assert interval["lower"] <= interval["estimate"] <= interval["upper"]
    assert interval["block_length"] == 10
    assert interval["resamples"] == 200


def test_block_bootstrap_interval_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        block_bootstrap_interval([1.0])
    with pytest.raises(ValueError):
        block_bootstrap_interval([1.0, np.nan])
    with pytest.raises(ValueError):
        block_bootstrap_interval([1.0, 2.0], confidence=1.5)
    with pytest.raises(ValueError):
        block_bootstrap_interval([1.0, 2.0], resamples=0)
    with pytest.raises(ValueError):
        block_bootstrap_interval([1.0, 2.0], block_length=0)


def test_dm_style_statistic_is_small_for_zero_mean_noise() -> None:
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 1.0, 400)
    baseline = rng.normal(0.0, 1.0, 400) ** 2
    candidate = baseline + noise
    result = dm_style_statistic(candidate, baseline)
    assert abs(result["statistic"]) < 2.0
    assert result["two_sided_p_value"] > 0.05
    assert result["sample_count"] == 400


def test_dm_style_statistic_detects_systematic_loss_gap() -> None:
    rng = np.random.default_rng(2)
    baseline = rng.normal(0.0, 1.0, 400) ** 2
    candidate = baseline + 0.5 + rng.normal(0.0, 0.2, 400)
    result = dm_style_statistic(candidate, baseline)
    assert result["statistic"] > 2.0
    assert result["two_sided_p_value"] < 0.05


def test_dm_style_statistic_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        dm_style_statistic([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        dm_style_statistic([1.0, np.inf], [1.0, 2.0])


def test_fold_metric_evidence_degrades_gracefully_on_tiny_inputs() -> None:
    folds = [{"relative_rmse": 0.9}, {"relative_rmse": 1.1}, {"relative_rmse": 0.95}]
    evidence = fold_metric_evidence(folds)
    assert evidence["reliable"] is False
    assert evidence["confidence_interval"] is None
    assert evidence["estimate"] == pytest.approx(np.mean([0.9, 1.1, 0.95]))
    assert evidence["fold_count"] == 3


def test_fold_metric_evidence_bootstraps_sufficient_folds() -> None:
    rng = np.random.default_rng(3)
    folds = [{"relative_rmse": float(value)} for value in 0.9 + rng.normal(0.0, 0.03, 8)]
    evidence = fold_metric_evidence(folds, resamples=200, block_length=2)
    assert evidence["reliable"] is True
    interval = evidence["confidence_interval"]
    assert interval is not None
    assert interval["lower"] <= evidence["estimate"] <= interval["upper"]


def test_fold_metric_evidence_accepts_plain_floats_and_validates() -> None:
    evidence = fold_metric_evidence(
        [0.8, 0.9, 1.0, 0.85, 0.95], metric="relative_rmse", resamples=50
    )
    assert evidence["metric"] == "relative_rmse"
    with pytest.raises(ValueError):
        fold_metric_evidence([{"mae": 1.0}])
    with pytest.raises(ValueError):
        fold_metric_evidence([])
