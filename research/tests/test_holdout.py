"""Synthetic-fixture tests for the locked holdout confirmation module."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from stock_autoresearch.candidates import Candidate, RidgeCandidate
from stock_autoresearch.holdout import (
    HoldoutResult,
    MultiWindowResult,
    evaluate_holdout,
    evaluate_multi_window_holdout,
)


class ConstantCandidate(Candidate):
    """Deterministic predictor used for the hand-computed tiny case."""

    name = "constant"

    def __init__(self, value: float):
        self.value = value

    def fit(self, x: np.ndarray, y: np.ndarray) -> ConstantCandidate:
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(len(x), self.value, dtype=np.float64)

    def describe(self) -> dict[str, object]:
        return {"family": self.name, "value": self.value}


class RecordingRidge(RidgeCandidate):
    """Ridge candidate that records its fitted coefficients for leakage checks."""

    def fit(self, x: np.ndarray, y: np.ndarray) -> RecordingRidge:
        super().fit(x, y)
        self.recorded_coef = np.array(self._model.coef_, copy=True)
        self.recorded_intercept = float(self._model.intercept_)
        return self


def make_drift_frame(rows: int = 900, seed: int = 0) -> pd.DataFrame:
    """Slowly varying signal drives the drift so a linear model can learn it.

    Return at row t is ``0.004 * signal[t-1] + noise``; because the signal
    moves slowly, the cumulative horizon return is roughly ``0.004 * horizon *
    signal[origin]`` and is predictable from the features visible at the
    origin, so a linear candidate beats the zero-return persistence baseline.
    """
    rng = np.random.default_rng(seed)
    state = 1.0
    signal = np.empty(rows)
    for t in range(rows):
        state = float(np.clip(state + rng.normal(0.0, 0.05), -1.0, 1.0))
        signal[t] = state
    returns = np.zeros(rows)
    returns[1:] = 0.004 * signal[:-1] + rng.normal(0.0, 0.005, rows - 1)
    close = 100.0 * np.exp(np.cumsum(returns))
    frame = pd.DataFrame(
        {"signal": signal, "noise": rng.normal(0.0, 1.0, rows), "Close": close},
        index=pd.date_range("2020-01-01", periods=rows, freq="B"),
    )
    return frame


def make_noise_frame(rows: int = 900, seed: int = 1) -> pd.DataFrame:
    """Pure-noise random walk with unpredictable features."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, rows)))
    frame = pd.DataFrame(
        {
            "signal": rng.normal(0.0, 1.0, rows),
            "noise": rng.normal(0.0, 1.0, rows),
            "Close": close,
        },
        index=pd.date_range("2020-01-01", periods=rows, freq="B"),
    )
    return frame


def make_single_regime_frame(rows: int = 1200, cutoff: int = 620, seed: int = 3) -> pd.DataFrame:
    """Predictable drift only before ``cutoff``; pure noise afterwards.

    Only the early test block can show a real edge, so the multi-window
    majority rule must reject this series despite one strong window.
    """
    rng = np.random.default_rng(seed)
    state = 1.0
    signal = np.empty(rows)
    for t in range(rows):
        state = float(np.clip(state + rng.normal(0.0, 0.05), -1.0, 1.0))
        signal[t] = state
    returns = rng.normal(0.0, 0.012, rows)
    returns[1:cutoff] = 0.004 * signal[: cutoff - 1] + rng.normal(0.0, 0.005, cutoff - 1)
    returns[0] = 0.0
    close = 100.0 * np.exp(np.cumsum(returns))
    return pd.DataFrame(
        {"signal": signal, "noise": rng.normal(0.0, 1.0, rows), "Close": close},
        index=pd.date_range("2020-01-01", periods=rows, freq="B"),
    )


def test_drift_series_linear_model_beats_persistence() -> None:
    frame = make_drift_frame()
    result = evaluate_holdout(
        frame, lambda seed: RidgeCandidate(), horizon=5, holdout_rows=120, window=60
    )
    assert result.sample_count == 120
    assert result.train_count == len(frame) - 5 - 60 - 120 - 5
    assert result.relative_mae < 1.0
    assert result.relative_rmse < 1.0
    assert result.mae == pytest.approx(result.relative_mae * result.persistence_mae)
    assert result.rmse == pytest.approx(result.relative_rmse * result.persistence_rmse)
    # CI point estimates must agree with the relative point estimates.
    assert result.mae_ci["estimate"] == pytest.approx(result.relative_mae)
    assert result.rmse_ci["estimate"] == pytest.approx(result.relative_rmse)
    assert result.mae_ci["block_length"] >= 5
    assert result.rmse_ci["block_length"] >= 5


def test_tiny_case_matches_hand_computed_values() -> None:
    # log-close path whose one-step log returns are 0.1, 0.2, 0.3, 0.4, 0.5, 0.6.
    log_close = [0.0, 0.1, 0.3, 0.6, 1.0, 1.5, 2.1]
    frame = pd.DataFrame(
        {
            "signal": np.arange(len(log_close), dtype=float),
            "Close": np.exp(log_close),
        },
        index=pd.date_range("2020-01-01", periods=len(log_close), freq="B"),
    )
    # window=2, horizon=1 -> origins 2, 3, 4, 5 with targets 0.3, 0.4, 0.5, 0.6.
    # holdout_rows=2 -> test targets 0.5 and 0.6; purge removes horizon samples
    # so only origin 2 (target 0.3) remains for training.
    result = evaluate_holdout(
        frame, lambda seed: ConstantCandidate(0.2), horizon=1, holdout_rows=2, window=2
    )
    assert result.sample_count == 2
    assert result.train_count == 1
    assert result.mae == pytest.approx((0.3 + 0.4) / 2.0)
    assert result.rmse == pytest.approx(math.sqrt((0.09 + 0.16) / 2.0))
    assert result.persistence_mae == pytest.approx((0.5 + 0.6) / 2.0)
    assert result.persistence_rmse == pytest.approx(math.sqrt((0.25 + 0.36) / 2.0))
    assert result.relative_mae == pytest.approx(0.35 / 0.55)
    assert result.relative_rmse == pytest.approx(math.sqrt(0.125 / 0.305))


def test_pure_noise_metrics_near_or_above_one() -> None:
    frame = make_noise_frame()
    result = evaluate_holdout(
        frame, lambda seed: RidgeCandidate(), horizon=5, holdout_rows=120, window=60
    )
    # On an unpredictable random walk the fitted model cannot beat persistence;
    # any in-sample fit only adds noise, so relative errors stay near/above 1.
    assert result.relative_mae > 0.95
    assert result.relative_rmse > 0.95


def test_test_period_changes_do_not_alter_fitted_model() -> None:
    frame = make_drift_frame(seed=2)
    horizon = 5
    holdout_rows = 120

    fitted: list[RecordingRidge] = []

    def factory(seed: int) -> RecordingRidge:
        candidate = RecordingRidge()
        fitted.append(candidate)
        return candidate

    original = evaluate_holdout(frame, factory, horizon=horizon, holdout_rows=holdout_rows)

    # Perturb every row that belongs to the test period (test origin rows plus
    # the horizon-length target tail): training data must be untouched, so the
    # fitted coefficients must be bit-identical.
    perturbed = frame.copy()
    perturbed.iloc[-(holdout_rows + horizon) :] = (
        perturbed.iloc[-(holdout_rows + horizon) :] * 1.7 + 3.0
    )
    evaluate_holdout(perturbed, factory, horizon=horizon, holdout_rows=holdout_rows)

    assert len(fitted) == 2
    np.testing.assert_array_equal(fitted[0].recorded_coef, fitted[1].recorded_coef)
    assert fitted[0].recorded_intercept == fitted[1].recorded_intercept
    assert original.train_count == 710


def test_verdict_rule() -> None:
    base = {
        "candidate": {"family": "test"},
        "horizon": 10,
        "holdout_rows": 252,
        "window": 60,
        "mae": 0.1,
        "rmse": 0.1,
        "persistence_mae": 0.11,
        "persistence_rmse": 0.11,
        "sample_count": 252,
        "train_count": 1000,
        "mae_difference_ci": {},
        "rmse_difference_ci": {},
    }

    def ci(upper: float) -> dict[str, float]:
        return {"estimate": upper - 0.05, "lower": upper - 0.1, "upper": upper}

    survives = HoldoutResult(
        relative_mae=0.95, relative_rmse=0.96, mae_ci=ci(0.99), rmse_ci=ci(0.99), **base
    )
    assert survives.verdict() == "edge_survives"

    wide_ci = HoldoutResult(
        relative_mae=0.95, relative_rmse=0.96, mae_ci=ci(1.03), rmse_ci=ci(0.99), **base
    )
    assert wide_ci.verdict() == "edge_not_confirmed"

    weak_point = HoldoutResult(
        relative_mae=0.99, relative_rmse=0.96, mae_ci=ci(0.99), rmse_ci=ci(0.99), **base
    )
    assert weak_point.verdict() == "edge_not_confirmed"


# ---------------------------------------------------------------------------
# Multi-window rolling holdout
#
# Rule-fixed placement for rows=1200, horizon=5, min_train_rows=400:
# usable origin region [400, 1195) -> block length 198, edges
# [400, 598, 796, 994, 1195); each block's test origins are its final 126
# origins: [472,598), [670,796), [868,994), [1069,1195).
# ---------------------------------------------------------------------------


def test_multi_window_stable_edge_survives() -> None:
    frame = make_drift_frame(rows=1200, seed=0)
    result = evaluate_multi_window_holdout(frame, lambda seed: RidgeCandidate(), horizon=5)
    assert isinstance(result, MultiWindowResult)
    assert len(result.windows) == 4
    assert result.pooled_sample_count == 504
    # Rule-fixed placement checks.
    assert result.windows[0].test_origin_start == 472
    assert result.windows[0].test_origin_end == 598
    assert result.windows[3].test_origin_end == 1195
    assert result.windows[0].train_count == 407
    for entry in result.windows:
        assert entry.sample_count == 126
    # A stable predictable edge must pass a majority of windows and the CI.
    assert result.windows_passing_gate() >= result.majority_required()
    assert result.majority_required() == 3
    assert result.pooled_relative_mae < 0.98
    assert result.pooled_relative_rmse < 0.98
    assert result.mae_ci["upper"] < 1.0
    assert result.rmse_ci["upper"] < 1.0
    assert result.mae_ci["block_length"] >= 5
    assert result.verdict() == "edge_survives"


def test_multi_window_pure_noise_fails() -> None:
    frame = make_noise_frame(rows=1200, seed=1)
    result = evaluate_multi_window_holdout(frame, lambda seed: RidgeCandidate(), horizon=5)
    assert result.pooled_relative_mae > 0.95
    assert result.pooled_relative_rmse > 0.95
    assert result.verdict() == "edge_not_confirmed"


def test_multi_window_single_regime_artifact_fails_majority() -> None:
    # Signal exists only before row 620: only the first test block
    # ([472, 598)) can show an edge, so the majority rule must reject it.
    frame = make_single_regime_frame()
    result = evaluate_multi_window_holdout(frame, lambda seed: RidgeCandidate(), horizon=5)
    assert result.windows[0].passes_gate()
    assert result.windows_passing_gate() == 1
    assert result.windows_passing_gate() < result.majority_required()
    assert result.verdict() == "edge_not_confirmed"


def test_multi_window_train_isolation_across_windows() -> None:
    frame = make_drift_frame(rows=1200, seed=2)
    horizon = 5

    fitted: list[RecordingRidge] = []

    def factory(seed: int) -> RecordingRidge:
        candidate = RecordingRidge()
        fitted.append(candidate)
        return candidate

    evaluate_multi_window_holdout(frame, factory, horizon=horizon)
    first_run = [
        (np.array(entry.recorded_coef, copy=True), entry.recorded_intercept) for entry in fitted
    ]

    # Perturb every row from the final window's first test origin onward.
    # All four windows' training samples (and their horizon-length targets)
    # end strictly before that row, so every fit must be bit-identical.
    perturbed = frame.copy()
    perturbed.iloc[1069:] = perturbed.iloc[1069:] * 1.7 + 3.0
    evaluate_multi_window_holdout(perturbed, factory, horizon=horizon)

    assert len(fitted) == 8
    for index in range(4):
        np.testing.assert_array_equal(first_run[index][0], fitted[4 + index].recorded_coef)
        assert first_run[index][1] == fitted[4 + index].recorded_intercept
