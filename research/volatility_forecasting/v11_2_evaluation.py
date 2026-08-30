"""Per-horizon scoring, session-block uncertainty, and V11.2 gates."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapInterval:
    mean_delta: float
    ci_lower_95: float
    ci_upper_95: float
    raw_p_value: float
    n_replicates: int
    block_sessions: int
    unique_sessions: int
    stock_origin_observations: int
    seed: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class HorizonGate:
    horizon: int
    candidate: str
    comparator: str
    mean_crps_candidate: float
    mean_crps_comparator: float
    interval: BootstrapInterval
    holm_p_value: float | None
    passed: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["interval"] = self.interval.to_dict()
        return payload


def _session_means(dates: Iterable[str], values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    date_array = np.asarray([str(value) for value in dates], dtype="U64")
    values_array = np.asarray(values, dtype=np.float64)
    if values_array.ndim != 1 or len(date_array) != len(values_array) or not len(values_array):
        raise ValueError("dates and one-dimensional loss arrays must have equal non-zero length")
    if not np.isfinite(values_array).all():
        raise ValueError("loss arrays must be finite")
    sessions = np.array(sorted(set(date_array.tolist())), dtype="U64")
    means = np.asarray(
        [float(np.mean(values_array[date_array == session])) for session in sessions],
        dtype=np.float64,
    )
    return sessions, means


def session_block_bootstrap_ci(
    dates: Iterable[str],
    candidate_losses: np.ndarray,
    comparator_losses: np.ndarray,
    *,
    block_sessions: int = 20,
    n_replicates: int = 10_000,
    seed: int = 42,
) -> BootstrapInterval:
    """Bootstrap paired loss differences in contiguous session blocks.

    Securities sharing an origin session are first averaged, so the estimand
    gives equal weight to sessions rather than treating correlated rows as
    independent observations.
    """
    candidate = np.asarray(candidate_losses, dtype=np.float64)
    comparator = np.asarray(comparator_losses, dtype=np.float64)
    if candidate.shape != comparator.shape:
        raise ValueError("candidate and comparator losses must have equal shape")
    sessions, _ = _session_means(dates, candidate)
    _, candidate_session = _session_means(dates, candidate)
    _, comparator_session = _session_means(dates, comparator)
    deltas = candidate_session - comparator_session
    n_sessions = len(sessions)
    if block_sessions < 1 or block_sessions > n_sessions:
        raise ValueError("bootstrap block length must be within the session count")
    if n_replicates < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    rng = np.random.default_rng(seed)
    boot = np.empty(n_replicates, dtype=np.float64)
    starts_count = int(np.ceil(n_sessions / block_sessions))
    for replicate in range(n_replicates):
        starts = rng.integers(0, n_sessions, size=starts_count)
        indices = np.concatenate(
            [((start + np.arange(block_sessions)) % n_sessions) for start in starts]
        )[:n_sessions]
        boot[replicate] = float(np.mean(deltas[indices]))
    mean_delta = float(np.mean(deltas))
    centered = boot - mean_delta
    # One-sided bootstrap p-value for candidate loss strictly below comparator.
    p_value = (1.0 + float(np.sum(centered <= mean_delta))) / (n_replicates + 1.0)
    return BootstrapInterval(
        mean_delta=mean_delta,
        ci_lower_95=float(np.percentile(boot, 2.5)),
        ci_upper_95=float(np.percentile(boot, 97.5)),
        raw_p_value=min(max(p_value, 0.0), 1.0),
        n_replicates=n_replicates,
        block_sessions=block_sessions,
        unique_sessions=n_sessions,
        stock_origin_observations=len(candidate),
        seed=seed,
    )


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    """Return Holm step-down adjusted p-values in the original order."""
    values = np.asarray([float(value) for value in p_values], dtype=np.float64)
    if len(values) == 0:
        return []
    if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p-values must be finite and in [0, 1]")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(len(values), dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        corrected = (len(values) - rank) * values[index]
        running = max(running, corrected)
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def evaluate_horizon_gates(
    *,
    dates: Iterable[str],
    horizons: Iterable[int],
    candidate: str,
    comparator: str,
    candidate_losses_by_horizon: dict[int, np.ndarray],
    comparator_losses_by_horizon: dict[int, np.ndarray],
    candidate_crps_by_horizon: dict[int, float],
    comparator_crps_by_horizon: dict[int, float],
    qlike_candidate_by_horizon: dict[int, float] | None = None,
    qlike_comparator_by_horizon: dict[int, float] | None = None,
    block_sessions: int = 20,
    n_replicates: int = 10_000,
    seed: int = 42,
) -> list[HorizonGate]:
    """Evaluate four independent horizon gates and apply Holm correction."""
    horizon_list = [int(value) for value in horizons]
    intervals: list[BootstrapInterval] = []
    for horizon in horizon_list:
        intervals.append(
            session_block_bootstrap_ci(
                dates,
                candidate_losses_by_horizon[horizon],
                comparator_losses_by_horizon[horizon],
                block_sessions=block_sessions,
                n_replicates=n_replicates,
                seed=seed,
            )
        )
    adjusted = holm_adjust(interval.raw_p_value for interval in intervals)
    gates: list[HorizonGate] = []
    for horizon, interval, adjusted_p in zip(horizon_list, intervals, adjusted, strict=True):
        qlike_ok = True
        if qlike_candidate_by_horizon is not None and qlike_comparator_by_horizon is not None:
            qlike_ok = (
                qlike_candidate_by_horizon[horizon] <= qlike_comparator_by_horizon[horizon] + 1e-12
            )
        passed = (
            candidate_crps_by_horizon[horizon] < comparator_crps_by_horizon[horizon]
            and interval.ci_upper_95 < 0.0
            and adjusted_p < 0.05
            and qlike_ok
        )
        reason = (
            "passed" if passed else "candidate did not pass paired CRPS, uncertainty, or QLIKE gate"
        )
        gates.append(
            HorizonGate(
                horizon=horizon,
                candidate=candidate,
                comparator=comparator,
                mean_crps_candidate=float(candidate_crps_by_horizon[horizon]),
                mean_crps_comparator=float(comparator_crps_by_horizon[horizon]),
                interval=interval,
                holm_p_value=float(adjusted_p),
                passed=passed,
                reason=reason,
            )
        )
    return gates


def evaluate_m0_adequacy(
    *,
    dates: Iterable[str],
    horizons: Iterable[int],
    har_losses_by_horizon: dict[int, np.ndarray],
    constant_losses_by_horizon: dict[int, np.ndarray],
    persistence_losses_by_horizon: dict[int, np.ndarray],
    har_crps_by_horizon: dict[int, float],
    constant_crps_by_horizon: dict[int, float],
    persistence_crps_by_horizon: dict[int, float],
    block_sessions: int = 20,
    n_replicates: int = 10_000,
    seed: int = 42,
) -> dict[str, list[HorizonGate]]:
    """Pre-register the two four-horizon M0 adequacy comparisons."""
    horizon_list = [int(value) for value in horizons]
    comparisons = [
        evaluate_horizon_gates(
            dates=dates,
            horizons=horizon_list,
            candidate="M0_HAR_BASELINE",
            comparator="ZERO_RETURN_CONST_VAR",
            candidate_losses_by_horizon=har_losses_by_horizon,
            comparator_losses_by_horizon=constant_losses_by_horizon,
            candidate_crps_by_horizon=har_crps_by_horizon,
            comparator_crps_by_horizon=constant_crps_by_horizon,
            block_sessions=block_sessions,
            n_replicates=n_replicates,
            seed=seed,
        ),
        evaluate_horizon_gates(
            dates=dates,
            horizons=horizon_list,
            candidate="M0_HAR_BASELINE",
            comparator="ZERO_RETURN_PERSISTENCE_VOL",
            candidate_losses_by_horizon=har_losses_by_horizon,
            comparator_losses_by_horizon=persistence_losses_by_horizon,
            candidate_crps_by_horizon=har_crps_by_horizon,
            comparator_crps_by_horizon=persistence_crps_by_horizon,
            block_sessions=block_sessions,
            n_replicates=n_replicates,
            seed=seed,
        ),
    ]
    return {"har_vs_constant": comparisons[0], "har_vs_persistence": comparisons[1]}
