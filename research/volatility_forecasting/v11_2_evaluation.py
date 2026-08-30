"""Per-horizon scoring, session-block uncertainty, and V11.2 gates."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np

from .v11_2_protocol import V11_2_MAX_COVERAGE_80, V11_2_MIN_COVERAGE_80


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
    coverage_candidate_80: float | None = None
    coverage_comparator_80: float | None = None

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
    mean_delta = float(np.mean(deltas))
    # Center the observed deltas at the null boundary (zero) for the
    # one-sided test.  Resampling the observed, uncentered deltas would test
    # around the observed effect and produces an invalid p-value.
    null_deltas = deltas - mean_delta
    boot = np.empty(n_replicates, dtype=np.float64)
    null_boot = np.empty(n_replicates, dtype=np.float64)
    starts_count = int(np.ceil(n_sessions / block_sessions))
    for replicate in range(n_replicates):
        starts = rng.integers(0, n_sessions, size=starts_count)
        indices = np.concatenate(
            [((start + np.arange(block_sessions)) % n_sessions) for start in starts]
        )[:n_sessions]
        boot[replicate] = float(np.mean(deltas[indices]))
        null_boot[replicate] = float(np.mean(null_deltas[indices]))
    # One-sided bootstrap p-value for candidate loss strictly below comparator
    # under H0: E[candidate - comparator] = 0.
    p_value = (1.0 + float(np.sum(null_boot <= mean_delta))) / (n_replicates + 1.0)
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
    coverage_candidate_by_horizon: dict[int, float] | None = None,
    coverage_comparator_by_horizon: dict[int, float] | None = None,
    minimum_coverage_80: float = V11_2_MIN_COVERAGE_80,
    maximum_coverage_80: float = V11_2_MAX_COVERAGE_80,
    block_sessions: int = 20,
    n_replicates: int = 10_000,
    seed: int = 42,
) -> list[HorizonGate]:
    """Evaluate four independent horizon gates and apply Holm correction."""
    horizon_list = [int(value) for value in horizons]
    if (qlike_candidate_by_horizon is None) != (qlike_comparator_by_horizon is None):
        raise ValueError("candidate and comparator QLIKE maps must be supplied together")
    if (coverage_candidate_by_horizon is None) != (coverage_comparator_by_horizon is None):
        raise ValueError("candidate and comparator coverage maps must be supplied together")
    if not 0.0 < minimum_coverage_80 < maximum_coverage_80 < 1.0:
        raise ValueError("coverage bounds must satisfy 0 < minimum < maximum < 1")
    date_values = list(dates)
    intervals: list[BootstrapInterval] = []
    for horizon in horizon_list:
        intervals.append(
            session_block_bootstrap_ci(
                date_values,
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
        coverage_candidate = None
        coverage_comparator = None
        coverage_ok = True
        if coverage_candidate_by_horizon is not None and coverage_comparator_by_horizon is not None:
            coverage_candidate = float(coverage_candidate_by_horizon[horizon])
            coverage_comparator = float(coverage_comparator_by_horizon[horizon])
            coverage_ok = minimum_coverage_80 <= coverage_candidate <= maximum_coverage_80
        passed = (
            candidate_crps_by_horizon[horizon] < comparator_crps_by_horizon[horizon]
            and interval.ci_upper_95 < 0.0
            and adjusted_p < 0.05
            and qlike_ok
            and coverage_ok
        )
        reason = (
            "passed"
            if passed
            else "candidate did not pass paired CRPS, uncertainty, QLIKE, or coverage gate"
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
                coverage_candidate_80=coverage_candidate,
                coverage_comparator_80=coverage_comparator,
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
    har_coverage_by_horizon: dict[int, float] | None = None,
    minimum_coverage_80: float = V11_2_MIN_COVERAGE_80,
    maximum_coverage_80: float = V11_2_MAX_COVERAGE_80,
    block_sessions: int = 20,
    n_replicates: int = 10_000,
    seed: int = 42,
) -> dict[str, list[HorizonGate]]:
    """Pre-register the two four-horizon M0 adequacy comparisons.

    When ``har_coverage_by_horizon`` is supplied, the HAR prior must also
    keep its central 80% interval inside the preregistered calibration band.
    The argument is optional for backwards-compatible diagnostic callers, but
    certification and the development runner always provide it.
    """
    horizon_list = [int(value) for value in horizons]
    if not horizon_list or len(set(horizon_list)) != len(horizon_list):
        raise ValueError("horizons must be a non-empty unique sequence")
    if not 0.0 < minimum_coverage_80 < maximum_coverage_80 < 1.0:
        raise ValueError("coverage bounds must satisfy 0 < minimum < maximum < 1")
    if har_coverage_by_horizon is not None:
        missing = [horizon for horizon in horizon_list if horizon not in har_coverage_by_horizon]
        if missing:
            raise ValueError(f"HAR coverage is missing horizons: {missing}")
        if any(
            not np.isfinite(float(har_coverage_by_horizon[horizon]))
            or not 0.0 <= float(har_coverage_by_horizon[horizon]) <= 1.0
            for horizon in horizon_list
        ):
            raise ValueError("HAR coverage values must be finite and in [0, 1]")
    date_values = list(dates)
    comparisons = [
        ("ZERO_RETURN_CONST_VAR", constant_losses_by_horizon, constant_crps_by_horizon),
        ("ZERO_RETURN_PERSISTENCE_VOL", persistence_losses_by_horizon, persistence_crps_by_horizon),
    ]
    intervals: list[BootstrapInterval] = []
    for _comparator, losses, _crps in comparisons:
        for horizon in horizon_list:
            intervals.append(
                session_block_bootstrap_ci(
                    date_values,
                    har_losses_by_horizon[horizon],
                    losses[horizon],
                    block_sessions=block_sessions,
                    n_replicates=n_replicates,
                    seed=seed,
                )
            )
    adjusted = holm_adjust(interval.raw_p_value for interval in intervals)
    output: dict[str, list[HorizonGate]] = {}
    for comparison_index, (comparator, _losses, crps) in enumerate(comparisons):
        gates: list[HorizonGate] = []
        offset = comparison_index * len(horizon_list)
        for horizon_index, horizon in enumerate(horizon_list):
            interval = intervals[offset + horizon_index]
            adjusted_p = adjusted[offset + horizon_index]
            har_coverage = (
                float(har_coverage_by_horizon[horizon])
                if har_coverage_by_horizon is not None
                else None
            )
            coverage_ok = (
                har_coverage is None or minimum_coverage_80 <= har_coverage <= maximum_coverage_80
            )
            passed = (
                har_crps_by_horizon[horizon] < crps[horizon]
                and interval.ci_upper_95 < 0.0
                and adjusted_p < 0.05
                and coverage_ok
            )
            gates.append(
                HorizonGate(
                    horizon=horizon,
                    candidate="M0_HAR_BASELINE",
                    comparator=comparator,
                    mean_crps_candidate=float(har_crps_by_horizon[horizon]),
                    mean_crps_comparator=float(crps[horizon]),
                    interval=interval,
                    holm_p_value=float(adjusted_p),
                    passed=passed,
                    reason=(
                        "passed"
                        if passed
                        else "HAR did not pass the corrected adequacy or coverage gate"
                    ),
                    coverage_candidate_80=har_coverage,
                )
            )
        output[
            "har_vs_constant" if comparator == "ZERO_RETURN_CONST_VAR" else "har_vs_persistence"
        ] = gates
    return output
