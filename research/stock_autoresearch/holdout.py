"""Locked out-of-sample holdout confirmation for promoted winners.

Step-24 of the research programme: after the in-sample purged expanding-fold
screen, each promoted winner must still beat the persistence baseline on a
truly unseen forward period.

Locked test-period rule
-----------------------
The final ``holdout_rows`` valid forecast origins of the snapshot are the test
origins. This split is fixed by rule BEFORE any results are observed and must
never be re-tuned: if a winner's edge does not survive this period, the
correct conclusion is that the edge was in-sample overfitting.

Train isolation
---------------
Windowed samples and cumulative log-return targets are built with the locked
``build_examples`` helper from ``data.py`` (imported, never modified), so the
feature/target semantics match the in-sample evaluation exactly. Training uses
every sample whose origin precedes the test period, minus a purge gap of
``horizon`` samples so that no training target (which reads ``horizon`` rows
into the future of its origin) overlaps the first test origin. No test-origin
row can therefore influence the fitted model.

Verdict rule
------------
``edge_survives`` iff ``relative_mae < 0.98`` AND ``relative_rmse < 0.98``
AND the upper bounds of both 95% bootstrap CIs are below 1.0. The point gates
require a real margin, and the CI condition requires the advantage to be
statistically visible on the holdout itself.

Multi-window extension
----------------------
A single 252-session window can produce an incidental pass, so
``evaluate_multi_window_holdout`` spreads the evaluation across several
disjoint test blocks with a RULE-FIXED placement chosen before any results
are observed: the usable origin region starting at row index ``min_train_rows``
is divided into ``window_count`` contiguous equal blocks, and each block's
test origins are its final ``window_rows`` valid forecast origins. Training
expands: window ``k`` trains on everything strictly before its test origins
(minus the same ``horizon`` purge as ``evaluate_holdout``), so later windows
train on more data than earlier ones. The verdict requires BOTH a majority of
per-window point-estimate passes AND pooled block-bootstrap CI upper bounds
below 1.0, which separates a transferable edge from single-window noise.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .candidates import PersistenceCandidate
from .data import Snapshot, build_examples
from .statistics import block_bootstrap_interval

RELATIVE_GATE = 0.98
DEFAULT_HOLDOUT_ROWS = 252
DEFAULT_WINDOW = 60
DEFAULT_WINDOW_COUNT = 4
DEFAULT_WINDOW_ROWS = 126
DEFAULT_MIN_TRAIN_ROWS = 400
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 0


@dataclass(frozen=True)
class HoldoutResult:
    """Locked holdout outcome for one candidate on one snapshot/horizon."""

    candidate: dict[str, object]
    horizon: int
    holdout_rows: int
    window: int
    relative_mae: float
    relative_rmse: float
    mae: float
    rmse: float
    persistence_mae: float
    persistence_rmse: float
    sample_count: int
    train_count: int
    mae_ci: dict[str, float]
    rmse_ci: dict[str, float]
    mae_difference_ci: dict[str, float]
    rmse_difference_ci: dict[str, float]

    def verdict(self) -> str:
        """Return ``edge_survives`` or ``edge_not_confirmed``.

        Rule (locked): ``edge_survives`` iff both relative point estimates are
        below ``RELATIVE_GATE`` (0.98) AND both relative CI upper bounds are
        strictly below 1.0.
        """
        survives = (
            self.relative_mae < RELATIVE_GATE
            and self.relative_rmse < RELATIVE_GATE
            and self.mae_ci["upper"] < 1.0
            and self.rmse_ci["upper"] < 1.0
        )
        return "edge_survives" if survives else "edge_not_confirmed"


def _relative_ci(difference_ci: dict[str, float | int], baseline_mean: float) -> dict[str, float]:
    # rel = mean_cand / mean_base = 1 + mean_diff / mean_base; the baseline
    # mean is the fixed observed constant, so the difference CI maps linearly.
    return {
        "estimate": 1.0 + difference_ci["estimate"] / baseline_mean,
        "lower": 1.0 + difference_ci["lower"] / baseline_mean,
        "upper": 1.0 + difference_ci["upper"] / baseline_mean,
        "confidence": float(difference_ci["confidence"]),
        "resamples": float(difference_ci["resamples"]),
        "block_length": float(difference_ci["block_length"]),
    }


def _relative_squared_ci(
    difference_ci: dict[str, float | int], baseline_mean_sq: float
) -> dict[str, float]:
    # relative_rmse = sqrt(mean_sq_cand / mean_sq_base); transform the squared
    # loss difference CI through the monotone square root, guarding the domain.
    ratio = _relative_ci(difference_ci, baseline_mean_sq)
    return {
        "estimate": float(np.sqrt(max(ratio["estimate"], 0.0))),
        "lower": float(np.sqrt(max(ratio["lower"], 0.0))),
        "upper": float(np.sqrt(max(ratio["upper"], 0.0))),
        "confidence": ratio["confidence"],
        "resamples": ratio["resamples"],
        "block_length": ratio["block_length"],
    }


def evaluate_holdout(
    snapshot: pd.DataFrame,
    factory: Callable[[int], object],
    horizon: int,
    *,
    holdout_rows: int = DEFAULT_HOLDOUT_ROWS,
    window: int = DEFAULT_WINDOW,
) -> HoldoutResult:
    """Evaluate one candidate against persistence on the locked forward holdout.

    The final ``holdout_rows`` valid forecast origins form the test period,
    chosen by rule before seeing any results. Everything before (minus a
    ``horizon``-sample purge) is training data. Targets are cumulative log
    returns from ``Close`` at the requested horizon, exactly as the locked
    ``build_examples`` in ``data.py`` constructs them.
    """
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if window < 1:
        raise ValueError("window must be positive")
    if holdout_rows < 2:
        raise ValueError("holdout_rows must contain at least two test origins")

    feature_names = tuple(column for column in snapshot.columns if column != "Close")
    if not feature_names:
        raise ValueError("snapshot has no feature columns besides Close")
    wrapped = Snapshot(frame=snapshot, snapshot_id="holdout", feature_names=feature_names)
    x, y, _origins = build_examples(wrapped, window=window, horizon=horizon)

    n_samples = len(y)
    if holdout_rows >= n_samples:
        raise ValueError(
            f"holdout_rows={holdout_rows} leaves no training data for {n_samples} samples"
        )
    test_start = n_samples - holdout_rows
    train_end = test_start - horizon  # purge so no training target touches the test period
    if train_end < 1:
        raise ValueError("not enough training rows after the horizon purge gap")

    train_idx = np.arange(0, train_end)
    test_idx = np.arange(test_start, n_samples)

    candidate = factory(0)
    candidate.fit(x[train_idx], y[train_idx])
    predicted = np.asarray(candidate.predict(x[test_idx]), dtype=np.float64)
    baseline = PersistenceCandidate().predict(x[test_idx])

    actual = y[test_idx]
    candidate_abs = np.abs(actual - predicted)
    baseline_abs = np.abs(actual - baseline)
    candidate_sq = np.square(actual - predicted)
    baseline_sq = np.square(actual - baseline)

    baseline_mae = float(np.mean(baseline_abs))
    baseline_mse = float(np.mean(baseline_sq))
    if baseline_mae <= 0.0 or baseline_mse <= 0.0:
        raise ValueError("persistence baseline has zero loss on the holdout; CI undefined")

    model_mae = float(np.mean(candidate_abs))
    model_mse = float(np.mean(candidate_sq))

    # Bootstrap CIs on the per-origin loss differences (candidate - baseline),
    # with block length at least the horizon so overlapping forecast windows
    # stay inside one resampled block.
    block_length = max(int(horizon), 1)
    mae_difference_ci = block_bootstrap_interval(
        candidate_abs - baseline_abs,
        resamples=BOOTSTRAP_RESAMPLES,
        block_length=block_length,
        seed=BOOTSTRAP_SEED,
    )
    rmse_difference_ci = block_bootstrap_interval(
        candidate_sq - baseline_sq,
        resamples=BOOTSTRAP_RESAMPLES,
        block_length=block_length,
        seed=BOOTSTRAP_SEED,
    )

    return HoldoutResult(
        candidate=candidate.describe(),
        horizon=horizon,
        holdout_rows=holdout_rows,
        window=window,
        relative_mae=model_mae / baseline_mae,
        relative_rmse=float(np.sqrt(model_mse / baseline_mse)),
        mae=model_mae,
        rmse=float(np.sqrt(model_mse)),
        persistence_mae=baseline_mae,
        persistence_rmse=float(np.sqrt(baseline_mse)),
        sample_count=int(len(test_idx)),
        train_count=int(len(train_idx)),
        mae_ci=_relative_ci(mae_difference_ci, baseline_mae),
        rmse_ci=_relative_squared_ci(rmse_difference_ci, baseline_mse),
        mae_difference_ci={key: float(value) for key, value in mae_difference_ci.items()},
        rmse_difference_ci={key: float(value) for key, value in rmse_difference_ci.items()},
    )


@dataclass(frozen=True)
class WindowHoldoutResult:
    """Per-window relative losses for one candidate on one test block."""

    window_index: int
    test_origin_start: int
    test_origin_end: int
    train_count: int
    sample_count: int
    relative_mae: float
    relative_rmse: float
    mae: float
    rmse: float
    persistence_mae: float
    persistence_rmse: float

    def passes_gate(self) -> bool:
        """True iff both relative point estimates beat ``RELATIVE_GATE``."""
        return self.relative_mae < RELATIVE_GATE and self.relative_rmse < RELATIVE_GATE


@dataclass(frozen=True)
class MultiWindowResult:
    """Rule-fixed multi-window holdout outcome for one candidate."""

    candidate: dict[str, object]
    horizon: int
    window: int
    window_count: int
    window_rows: int
    min_train_rows: int
    windows: tuple[WindowHoldoutResult, ...]
    pooled_sample_count: int
    pooled_relative_mae: float
    pooled_relative_rmse: float
    pooled_mae: float
    pooled_rmse: float
    pooled_persistence_mae: float
    pooled_persistence_rmse: float
    mae_ci: dict[str, float]
    rmse_ci: dict[str, float]
    mae_difference_ci: dict[str, float]
    rmse_difference_ci: dict[str, float]

    def windows_passing_gate(self) -> int:
        return sum(1 for entry in self.windows if entry.passes_gate())

    def majority_required(self) -> int:
        return self.window_count // 2 + 1

    def verdict(self) -> str:
        """Return ``edge_survives`` or ``edge_not_confirmed``.

        Rule (locked): ``edge_survives`` iff BOTH (a) a strict majority of
        windows pass the 0.98 point gates on BOTH metrics AND (b) the pooled
        95% CI upper bounds are strictly below 1.0 for BOTH metrics.
        """
        majority_ok = self.windows_passing_gate() >= self.majority_required()
        pooled_ci_ok = self.mae_ci["upper"] < 1.0 and self.rmse_ci["upper"] < 1.0
        return "edge_survives" if majority_ok and pooled_ci_ok else "edge_not_confirmed"


def evaluate_multi_window_holdout(
    snapshot: pd.DataFrame,
    factory: Callable[[int], object],
    horizon: int,
    *,
    window_count: int = DEFAULT_WINDOW_COUNT,
    window_rows: int = DEFAULT_WINDOW_ROWS,
    min_train_rows: int = DEFAULT_MIN_TRAIN_ROWS,
    window: int = DEFAULT_WINDOW,
) -> MultiWindowResult:
    """Rule-fixed multi-window rolling holdout against the persistence baseline.

    The usable origin region of the snapshot (row indices from
    ``min_train_rows`` up to the last valid forecast origin) is divided into
    ``window_count`` contiguous equal blocks BEFORE any results are observed.
    Each block's test origins are its final ``window_rows`` valid forecast
    origins; window ``k`` trains on every sample strictly before its test
    origins, minus the same ``horizon``-sample purge used by
    ``evaluate_holdout``. The candidate is fitted fresh per window.

    The pooled estimates concatenate the per-origin loss differences of all
    windows in time order and carry 95% block-bootstrap CIs with block length
    at least ``horizon``. See ``MultiWindowResult.verdict`` for the survival
    rule (majority of window passes AND pooled CI upper bounds below 1.0).
    """
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if window < 1:
        raise ValueError("window must be positive")
    if window_count < 1:
        raise ValueError("window_count must be positive")
    if window_rows < 2:
        raise ValueError("window_rows must contain at least two test origins")
    if min_train_rows < 1:
        raise ValueError("min_train_rows must be positive")

    feature_names = tuple(column for column in snapshot.columns if column != "Close")
    if not feature_names:
        raise ValueError("snapshot has no feature columns besides Close")
    wrapped = Snapshot(
        frame=snapshot, snapshot_id="multi-window-holdout", feature_names=feature_names
    )
    x, y, origins = build_examples(wrapped, window=window, horizon=horizon)

    # Valid forecast origins occupy row indices [window, len(snapshot) - horizon).
    region_start = max(int(min_train_rows), int(origins[0]))
    region_end = int(origins[-1]) + 1
    if region_end - region_start < window_count:
        raise ValueError(
            f"usable origin region [{region_start}, {region_end}) is too small for "
            f"{window_count} windows"
        )
    block_length = (region_end - region_start) // window_count
    if block_length < window_rows:
        raise ValueError(
            f"block length {block_length} cannot hold {window_rows} test origins; "
            "extend the snapshot or reduce window_count/window_rows"
        )

    # Rule-fixed contiguous block edges; the final block absorbs any remainder
    # rows so the placement never depends on observed results.
    edges = [region_start + k * block_length for k in range(window_count)]
    edges.append(region_end)

    window_results: list[WindowHoldoutResult] = []
    pooled_candidate_abs: list[np.ndarray] = []
    pooled_baseline_abs: list[np.ndarray] = []
    pooled_candidate_sq: list[np.ndarray] = []
    pooled_baseline_sq: list[np.ndarray] = []
    last_candidate: object | None = None

    for index in range(window_count):
        test_origin_end = edges[index + 1]
        test_origin_start = test_origin_end - window_rows
        test_start = test_origin_start - window  # sample index of the first test origin
        train_end = test_start - horizon  # purge so no training target touches the test block
        if train_end < 1:
            raise ValueError(f"window {index} has no training rows after the horizon purge gap")
        test_idx = np.arange(test_start, test_origin_end - window)
        train_idx = np.arange(0, train_end)

        candidate = factory(0)
        candidate.fit(x[train_idx], y[train_idx])
        last_candidate = candidate
        predicted = np.asarray(candidate.predict(x[test_idx]), dtype=np.float64)
        baseline = PersistenceCandidate().predict(x[test_idx])

        actual = y[test_idx]
        candidate_abs = np.abs(actual - predicted)
        baseline_abs = np.abs(actual - baseline)
        candidate_sq = np.square(actual - predicted)
        baseline_sq = np.square(actual - baseline)

        baseline_mae = float(np.mean(baseline_abs))
        baseline_mse = float(np.mean(baseline_sq))
        if baseline_mae <= 0.0 or baseline_mse <= 0.0:
            raise ValueError(f"persistence baseline has zero loss in window {index}; CI undefined")
        model_mae = float(np.mean(candidate_abs))
        model_mse = float(np.mean(candidate_sq))

        window_results.append(
            WindowHoldoutResult(
                window_index=index,
                test_origin_start=test_origin_start,
                test_origin_end=test_origin_end,
                train_count=int(len(train_idx)),
                sample_count=int(len(test_idx)),
                relative_mae=model_mae / baseline_mae,
                relative_rmse=float(np.sqrt(model_mse / baseline_mse)),
                mae=model_mae,
                rmse=float(np.sqrt(model_mse)),
                persistence_mae=baseline_mae,
                persistence_rmse=float(np.sqrt(baseline_mse)),
            )
        )
        pooled_candidate_abs.append(candidate_abs)
        pooled_baseline_abs.append(baseline_abs)
        pooled_candidate_sq.append(candidate_sq)
        pooled_baseline_sq.append(baseline_sq)

    all_candidate_abs = np.concatenate(pooled_candidate_abs)
    all_baseline_abs = np.concatenate(pooled_baseline_abs)
    all_candidate_sq = np.concatenate(pooled_candidate_sq)
    all_baseline_sq = np.concatenate(pooled_baseline_sq)

    pooled_baseline_mae = float(np.mean(all_baseline_abs))
    pooled_baseline_mse = float(np.mean(all_baseline_sq))
    pooled_model_mae = float(np.mean(all_candidate_abs))
    pooled_model_mse = float(np.mean(all_candidate_sq))

    block_length = max(int(horizon), 1)
    mae_difference_ci = block_bootstrap_interval(
        all_candidate_abs - all_baseline_abs,
        resamples=BOOTSTRAP_RESAMPLES,
        block_length=block_length,
        seed=BOOTSTRAP_SEED,
    )
    rmse_difference_ci = block_bootstrap_interval(
        all_candidate_sq - all_baseline_sq,
        resamples=BOOTSTRAP_RESAMPLES,
        block_length=block_length,
        seed=BOOTSTRAP_SEED,
    )

    assert last_candidate is not None  # window_count >= 1 guarantees one fit
    return MultiWindowResult(
        candidate=last_candidate.describe(),
        horizon=horizon,
        window=window,
        window_count=window_count,
        window_rows=window_rows,
        min_train_rows=min_train_rows,
        windows=tuple(window_results),
        pooled_sample_count=int(len(all_candidate_abs)),
        pooled_relative_mae=pooled_model_mae / pooled_baseline_mae,
        pooled_relative_rmse=float(np.sqrt(pooled_model_mse / pooled_baseline_mse)),
        pooled_mae=pooled_model_mae,
        pooled_rmse=float(np.sqrt(pooled_model_mse)),
        pooled_persistence_mae=pooled_baseline_mae,
        pooled_persistence_rmse=float(np.sqrt(pooled_baseline_mse)),
        mae_ci=_relative_ci(mae_difference_ci, pooled_baseline_mae),
        rmse_ci=_relative_squared_ci(rmse_difference_ci, pooled_baseline_mse),
        mae_difference_ci={key: float(value) for key, value in mae_difference_ci.items()},
        rmse_difference_ci={key: float(value) for key, value in rmse_difference_ci.items()},
    )
