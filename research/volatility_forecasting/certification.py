"""One-shot locked certification for the global volatility candidate.

This module does not train, tune, calibrate, or select a model.  It accepts
already-frozen predictions for the two reserves created by
``build_volatility_fold_plan`` and opens those reserves exactly once for final
evidence.  Keeping certification prediction-only prevents accidental
hyperparameter selection on NMM, MSFT, or the terminal temporal year.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from backend.panel.selection import diebold_mariano_hac, holm_correction

from .data import VolatilityPanelExamples
from .evaluation import cluster_losses_by_session, moving_block_ratio_upper_bound
from .folds import VolatilityFoldPlan
from .metrics import DistributionPredictions, horizon_distribution_metrics, qlike_losses

CERTIFICATION_PROTOCOL_VERSION = "global-volatility-locked-cert-v1"


@dataclass(frozen=True)
class LockedCertificationGate:
    """Conservative initial defaults frozen into every certification report."""

    maximum_relative_qlike: float = 1.0
    maximum_ratio_upper_95: float = 1.0
    maximum_required_ticker_relative_qlike: float = 1.05
    significance_level: float = 0.05
    minimum_sessions: int = 126
    minimum_transfer_tickers: int = 2

    def __post_init__(self) -> None:
        if not 0 < self.maximum_relative_qlike <= 1:
            raise ValueError("certification must require QLIKE non-degradation")
        if not 0 < self.maximum_ratio_upper_95 <= 1:
            raise ValueError("certification confidence bound must not exceed one")
        if self.maximum_required_ticker_relative_qlike < 1:
            raise ValueError("required-ticker guardrail cannot be stricter than one")
        if not 0 < self.significance_level < 1:
            raise ValueError("significance level must be in (0, 1)")
        if self.minimum_sessions < 5 or self.minimum_transfer_tickers < 1:
            raise ValueError("certification sample minimums are invalid")


@dataclass(frozen=True)
class LockedPopulationInput:
    """Frozen predictions and matched baselines for one reserved population."""

    population: Literal["temporal", "asset_transfer"]
    indices: np.ndarray
    predictions: DistributionPredictions
    baseline_variance: np.ndarray
    baseline_return_variance: np.ndarray


@dataclass(frozen=True)
class LockedHorizonDecision:
    population: str
    horizon: int
    decision: Literal["pass", "fail"]
    rows: int
    sessions: int
    tickers: int
    relative_qlike: float
    ratio_upper_95: float
    dm_statistic: float
    dm_p_value: float
    holm_significant: bool
    required_ticker_relative_qlike: dict[str, float]
    reasons: tuple[str, ...]
    metrics: dict[str, float | int]


@dataclass(frozen=True)
class LockedCertificationReport:
    certification_protocol_version: str
    model_identity: str
    development_evidence_sha256: str
    status: Literal["passed", "failed"]
    certification_start: str
    required_asset_holdouts: tuple[str, ...]
    gate: dict[str, float | int]
    decisions: tuple[LockedHorizonDecision, ...]
    certified_horizons: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _validate_population(
    population: LockedPopulationInput,
    *,
    expected_indices: np.ndarray,
    examples: VolatilityPanelExamples,
    gate: LockedCertificationGate,
) -> None:
    indices = np.asarray(population.indices, dtype=np.int64)
    expected = np.asarray(expected_indices, dtype=np.int64)
    if indices.ndim != 1 or not np.array_equal(indices, expected):
        raise ValueError(f"{population.population} indices do not match the locked reserve")
    rows = len(indices)
    horizons = len(examples.horizons)
    if population.predictions.variance.shape != (rows, horizons):
        raise ValueError(f"{population.population} prediction shape is incompatible")
    for name, values in (
        ("baseline_variance", population.baseline_variance),
        ("baseline_return_variance", population.baseline_return_variance),
    ):
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.shape != (rows, horizons):
            raise ValueError(f"{population.population} {name} shape is incompatible")
        if not np.isfinite(matrix).all() or (matrix <= 0).any():
            raise ValueError(f"{population.population} {name} must be finite and positive")
    sessions = np.unique(examples.origin_dates[indices])
    if len(sessions) < gate.minimum_sessions:
        raise ValueError(f"{population.population} reserve has too few sessions")
    if population.population == "asset_transfer":
        ticker_count = len(np.unique(examples.tickers[indices]))
        if ticker_count < gate.minimum_transfer_tickers:
            raise ValueError("asset-transfer reserve has too few tickers")


def certify_locked_predictions(
    *,
    examples: VolatilityPanelExamples,
    fold_plan: VolatilityFoldPlan,
    temporal: LockedPopulationInput,
    asset_transfer: LockedPopulationInput,
    model_identity: str,
    development_evidence_sha256: str,
    gate: LockedCertificationGate | None = None,
    required_asset_holdouts: tuple[str, ...] = ("NMM", "MSFT"),
    resamples: int = 2000,
    seed: int = 42,
) -> LockedCertificationReport:
    """Open both locked reserves and issue horizon-specific release decisions.

    The caller must provide an immutable model identity and SHA-256 digest of
    the development report that selected it.  The function rejects reordered,
    subsetted, or augmented holdout rows so a favorable slice cannot be used as
    a substitute for the complete reserve.
    """

    settings = gate or LockedCertificationGate()
    if not model_identity.strip():
        raise ValueError("model identity is required")
    if len(development_evidence_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in development_evidence_sha256
    ):
        raise ValueError("development evidence must be a lowercase SHA-256 digest")
    if temporal.population != "temporal" or asset_transfer.population != "asset_transfer":
        raise ValueError("certification population labels are incorrect")
    _validate_population(
        temporal,
        expected_indices=fold_plan.temporal_certification_indices,
        examples=examples,
        gate=settings,
    )
    _validate_population(
        asset_transfer,
        expected_indices=fold_plan.asset_transfer_certification_indices,
        examples=examples,
        gate=settings,
    )
    transfer_tickers = {str(value).upper() for value in examples.tickers[asset_transfer.indices]}
    missing_required = sorted(set(required_asset_holdouts) - transfer_tickers)
    if missing_required:
        raise ValueError(f"required asset holdouts are absent: {', '.join(missing_required)}")

    populations = (temporal, asset_transfer)
    evidence: list[dict[str, object]] = []
    p_values: list[float] = []
    for population in populations:
        indices = population.indices
        metrics = horizon_distribution_metrics(
            predictions=population.predictions,
            baseline_variance=population.baseline_variance,
            baseline_return_variance=population.baseline_return_variance,
            realized_variance=examples.realized_variance[indices],
            cumulative_returns=examples.cumulative_returns[indices],
            direction_classes=examples.direction_classes[indices],
            horizons=examples.horizons,
        )
        candidate_losses = qlike_losses(
            population.predictions.variance,
            examples.realized_variance[indices],
        )
        baseline_losses = qlike_losses(
            population.baseline_variance,
            examples.realized_variance[indices],
        )
        candidate_sessions, session_dates = cluster_losses_by_session(
            candidate_losses,
            examples.origin_dates[indices],
        )
        baseline_sessions, baseline_dates = cluster_losses_by_session(
            baseline_losses,
            examples.origin_dates[indices],
        )
        if not np.array_equal(session_dates, baseline_dates):
            raise RuntimeError("certification loss sessions do not align")
        for column, horizon in enumerate(examples.horizons):
            dm_stat, dm_p = diebold_mariano_hac(
                candidate_sessions[:, column],
                baseline_sessions[:, column],
                max_lag=max(1, horizon - 1),
            )
            upper = moving_block_ratio_upper_bound(
                candidate_sessions[:, column],
                baseline_sessions[:, column],
                resamples=resamples,
                block_length=max(5, horizon),
                seed=seed + len(evidence),
            )
            required_ratios: dict[str, float] = {}
            if population.population == "asset_transfer":
                for ticker in required_asset_holdouts:
                    mask = examples.tickers[indices] == ticker
                    candidate_mean = float(np.mean(candidate_losses[mask, column]))
                    baseline_mean = float(np.mean(baseline_losses[mask, column]))
                    required_ratios[ticker] = (
                        candidate_mean / baseline_mean if baseline_mean > 0 else float("inf")
                    )
            evidence.append(
                {
                    "population": population.population,
                    "horizon": horizon,
                    "rows": len(indices),
                    "sessions": len(session_dates),
                    "tickers": len(np.unique(examples.tickers[indices])),
                    "relative_qlike": float(metrics[column]["relative_qlike"]),
                    "ratio_upper_95": upper,
                    "dm_statistic": float(dm_stat),
                    "dm_p_value": float(dm_p),
                    "required_ticker_relative_qlike": required_ratios,
                    "metrics": metrics[column],
                }
            )
            p_values.append(float(dm_p))

    significant = holm_correction(p_values, alpha=settings.significance_level)
    decisions: list[LockedHorizonDecision] = []
    for row, holm_significant in zip(evidence, significant, strict=True):
        reasons: list[str] = []
        if float(row["relative_qlike"]) >= settings.maximum_relative_qlike:
            reasons.append("locked relative QLIKE did not beat the matched baseline")
        if float(row["ratio_upper_95"]) >= settings.maximum_ratio_upper_95:
            reasons.append("locked confidence bound did not beat the matched baseline")
        if float(row["dm_statistic"]) >= 0 or not holm_significant:
            reasons.append("locked QLIKE improvement was not Holm-significant")
        required_ratios = row["required_ticker_relative_qlike"]
        if any(
            value > settings.maximum_required_ticker_relative_qlike
            for value in required_ratios.values()
        ):
            reasons.append("NMM or MSFT exceeded the required-ticker degradation guardrail")
        decisions.append(
            LockedHorizonDecision(
                population=str(row["population"]),
                horizon=int(row["horizon"]),
                decision="fail" if reasons else "pass",
                rows=int(row["rows"]),
                sessions=int(row["sessions"]),
                tickers=int(row["tickers"]),
                relative_qlike=float(row["relative_qlike"]),
                ratio_upper_95=float(row["ratio_upper_95"]),
                dm_statistic=float(row["dm_statistic"]),
                dm_p_value=float(row["dm_p_value"]),
                holm_significant=bool(holm_significant),
                required_ticker_relative_qlike=dict(required_ratios),
                reasons=tuple(reasons),
                metrics=dict(row["metrics"]),
            )
        )

    certified = tuple(
        horizon
        for horizon in examples.horizons
        if all(decision.decision == "pass" for decision in decisions if decision.horizon == horizon)
    )
    return LockedCertificationReport(
        certification_protocol_version=CERTIFICATION_PROTOCOL_VERSION,
        model_identity=model_identity,
        development_evidence_sha256=development_evidence_sha256,
        status="passed" if certified == examples.horizons else "failed",
        certification_start=str(fold_plan.certification_start),
        required_asset_holdouts=required_asset_holdouts,
        gate=asdict(settings),
        decisions=tuple(decisions),
        certified_horizons=certified,
    )
