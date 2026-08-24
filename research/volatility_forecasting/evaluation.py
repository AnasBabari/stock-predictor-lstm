"""Fold execution and fail-closed promotion for the volatility TCN."""

from __future__ import annotations

import gc
from dataclasses import dataclass

import numpy as np
import torch

from backend.panel.selection import diebold_mariano_hac, holm_correction

from .contracts import VolatilityForecastProtocol, VolatilityPromotionGate
from .data import VolatilityPanelExamples
from .folds import VolatilityFoldPlan
from .metrics import (
    DistributionPredictions,
    horizon_distribution_metrics,
    qlike_losses,
)
from .model import (
    BaselineResidualTCNConfig,
    TorchTrainingConfig,
    TrainingResult,
    VolatilityLossWeights,
    train_baseline_residual_tcn,
)


@dataclass(frozen=True)
class FoldEvidence:
    fold: int
    validation_start: str
    validation_end: str
    rows: int
    best_epoch: int
    duration_seconds: float
    parameter_count: int
    metrics: tuple[dict[str, float | int], ...]


@dataclass(frozen=True)
class HorizonPromotionDecision:
    horizon: int
    promoted: bool
    reasons: tuple[str, ...]
    folds_beating_baseline: int
    worst_fold_relative_qlike: float
    relative_qlike: float
    relative_qlike_upper_95: float
    relative_gaussian_crps: float
    dm_statistic: float
    dm_p_value: float
    holm_significant: bool
    coverage_80: float


@dataclass(frozen=True)
class DevelopmentEvaluation:
    protocol_version: str
    seed: int
    folds: tuple[FoldEvidence, ...]
    pooled_metrics: tuple[dict[str, float | int], ...]
    promotion: tuple[HorizonPromotionDecision, ...]
    oof_indices: np.ndarray
    predictions: DistributionPredictions


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=-1, keepdims=True)


def predict_distribution(
    result: TrainingResult,
    examples: VolatilityPanelExamples,
    indices: np.ndarray,
    *,
    batch_size: int = 2048,
) -> DistributionPredictions:
    """Run bounded fold inference and return CPU NumPy arrays."""
    device = next(result.model.parameters()).device
    scaled = result.scaler.transform(examples.features[indices])
    variances: list[np.ndarray] = []
    locations: list[np.ndarray] = []
    logits_rows: list[np.ndarray] = []
    result.model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            stop = start + batch_size
            x = torch.from_numpy(scaled[start:stop]).to(device)
            baseline = torch.from_numpy(examples.baseline_variance[indices[start:stop]]).to(device)
            forecast_var, location, logits, _ = result.model(x, baseline)
            variances.append(forecast_var.cpu().numpy())
            locations.append(location.cpu().numpy())
            logits_rows.append(logits.cpu().numpy())
    return DistributionPredictions(
        variance=np.concatenate(variances, axis=0).astype(np.float64),
        return_location=np.concatenate(locations, axis=0).astype(np.float64),
        direction_probabilities=_softmax(np.concatenate(logits_rows, axis=0)).astype(np.float64),
    )


def moving_block_ratio_upper_bound(
    candidate_losses: np.ndarray,
    baseline_losses: np.ndarray,
    *,
    resamples: int = 1000,
    block_length: int = 10,
    confidence: float = 0.95,
    seed: int = 42,
) -> float:
    """Upper confidence bound for mean(candidate loss)/mean(baseline loss)."""
    candidate = np.asarray(candidate_losses, dtype=np.float64)
    baseline = np.asarray(baseline_losses, dtype=np.float64)
    if candidate.shape != baseline.shape or candidate.ndim != 1 or len(candidate) < 5:
        raise ValueError("paired loss arrays must be one-dimensional, matched, and length >= 5")
    if resamples < 100 or block_length < 1:
        raise ValueError("bootstrap requires at least 100 resamples and positive blocks")
    rng = np.random.default_rng(seed)
    length = len(candidate)
    block = min(block_length, length)
    ratios = np.empty(resamples, dtype=np.float64)
    for sample in range(resamples):
        selected: list[int] = []
        while len(selected) < length:
            start = int(rng.integers(0, length))
            selected.extend((start + offset) % length for offset in range(block))
        indices = np.asarray(selected[:length], dtype=np.int64)
        denominator = float(np.mean(baseline[indices]))
        ratios[sample] = (
            float(np.mean(candidate[indices])) / denominator
            if denominator > 0 and np.isfinite(denominator)
            else float("inf")
        )
    return float(np.quantile(ratios, confidence))


def assess_promotion(
    *,
    pooled_metrics: tuple[dict[str, float | int], ...],
    fold_metrics: tuple[tuple[dict[str, float | int], ...], ...],
    candidate_qlike_losses: np.ndarray,
    baseline_qlike_losses: np.ndarray,
    horizons: tuple[int, ...],
    gate: VolatilityPromotionGate | None = None,
    resamples: int = 1000,
    seed: int = 42,
) -> tuple[HorizonPromotionDecision, ...]:
    """Apply QLIKE, CRPS, calibration, fold, bootstrap, and DM gates."""
    settings = gate or VolatilityPromotionGate()
    if candidate_qlike_losses.shape != baseline_qlike_losses.shape:
        raise ValueError("candidate and baseline QLIKE loss matrices must match")
    if candidate_qlike_losses.shape[1] != len(horizons):
        raise ValueError("loss matrix horizon count does not match contract")

    dm_rows: list[tuple[float, float]] = []
    upper_bounds: list[float] = []
    for column, horizon in enumerate(horizons):
        dm_rows.append(
            diebold_mariano_hac(
                candidate_qlike_losses[:, column],
                baseline_qlike_losses[:, column],
                max_lag=max(1, horizon - 1),
            )
        )
        upper_bounds.append(
            moving_block_ratio_upper_bound(
                candidate_qlike_losses[:, column],
                baseline_qlike_losses[:, column],
                resamples=resamples,
                block_length=max(5, horizon),
                seed=seed + column,
            )
        )
    holm = holm_correction([row[1] for row in dm_rows], alpha=settings.significance_level)

    decisions: list[HorizonPromotionDecision] = []
    for column, horizon in enumerate(horizons):
        pooled = pooled_metrics[column]
        fold_relative = [float(metrics[column]["relative_qlike"]) for metrics in fold_metrics]
        folds_beating = sum(value < 1.0 for value in fold_relative)
        worst_fold = max(fold_relative)
        relative_qlike = float(pooled["relative_qlike"])
        relative_crps = float(pooled["relative_gaussian_crps"])
        coverage_80 = float(pooled["coverage_80"])
        dm_stat, dm_p = dm_rows[column]
        reasons: list[str] = []
        if relative_qlike >= settings.maximum_relative_qlike:
            reasons.append("pooled relative QLIKE did not clear the improvement gate")
        if upper_bounds[column] >= 1.0:
            reasons.append("bootstrap upper confidence bound did not beat the baseline")
        if folds_beating < settings.minimum_folds_beating_baseline:
            reasons.append("too few expanding folds beat the matched baseline")
        if worst_fold > settings.maximum_worst_fold_relative_qlike:
            reasons.append("worst fold exceeded the stability guardrail")
        if relative_crps >= settings.maximum_relative_gaussian_crps:
            reasons.append("probabilistic return CRPS did not beat the baseline")
        if (
            not settings.minimum_interval_coverage_80
            <= coverage_80
            <= settings.maximum_interval_coverage_80
        ):
            reasons.append("80% interval coverage is outside the calibration band")
        if dm_stat >= 0 or not holm[column]:
            reasons.append("paired QLIKE improvement is not Holm-significant")
        decisions.append(
            HorizonPromotionDecision(
                horizon=horizon,
                promoted=not reasons,
                reasons=tuple(reasons),
                folds_beating_baseline=folds_beating,
                worst_fold_relative_qlike=worst_fold,
                relative_qlike=relative_qlike,
                relative_qlike_upper_95=upper_bounds[column],
                relative_gaussian_crps=relative_crps,
                dm_statistic=float(dm_stat),
                dm_p_value=float(dm_p),
                holm_significant=bool(holm[column]),
                coverage_80=coverage_80,
            )
        )
    return tuple(decisions)


def evaluate_tcn_development(
    examples: VolatilityPanelExamples,
    fold_plan: VolatilityFoldPlan,
    protocol: VolatilityForecastProtocol,
    *,
    model_config: BaselineResidualTCNConfig | None = None,
    training_config: TorchTrainingConfig | None = None,
    loss_weights: VolatilityLossWeights | None = None,
    promotion_gate: VolatilityPromotionGate | None = None,
    seed: int = 42,
    device: str | None = None,
    resamples: int = 1000,
) -> DevelopmentEvaluation:
    """Train one model per expanding fold and pool untouched OOF evidence."""
    architecture = model_config or BaselineResidualTCNConfig(
        feature_count=examples.features.shape[-1],
        horizon_count=len(protocol.horizons),
    )
    fold_evidence: list[FoldEvidence] = []
    fold_metric_rows: list[tuple[dict[str, float | int], ...]] = []
    oof_indices: list[np.ndarray] = []
    oof_variance: list[np.ndarray] = []
    oof_location: list[np.ndarray] = []
    oof_direction: list[np.ndarray] = []

    for fold in fold_plan.folds:
        train = fold.train_indices
        validation = fold.validation_indices
        trained = train_baseline_residual_tcn(
            train_features=examples.features[train],
            train_baseline_variance=examples.baseline_variance[train],
            train_realized_variance=examples.realized_variance[train],
            train_cumulative_returns=examples.cumulative_returns[train],
            train_direction_classes=examples.direction_classes[train],
            validation_features=examples.features[validation],
            validation_baseline_variance=examples.baseline_variance[validation],
            validation_realized_variance=examples.realized_variance[validation],
            validation_cumulative_returns=examples.cumulative_returns[validation],
            validation_direction_classes=examples.direction_classes[validation],
            model_config=architecture,
            training_config=training_config,
            loss_weights=loss_weights,
            seed=seed,
            device=device,
        )
        predictions = predict_distribution(trained, examples, validation)
        metrics = tuple(
            horizon_distribution_metrics(
                predictions=predictions,
                baseline_variance=examples.baseline_variance[validation],
                realized_variance=examples.realized_variance[validation],
                cumulative_returns=examples.cumulative_returns[validation],
                direction_classes=examples.direction_classes[validation],
                horizons=protocol.horizons,
            )
        )
        fold_metric_rows.append(metrics)
        fold_evidence.append(
            FoldEvidence(
                fold=fold.fold,
                validation_start=str(fold.validation_start),
                validation_end=str(fold.validation_end),
                rows=len(validation),
                best_epoch=trained.best_epoch,
                duration_seconds=trained.duration_seconds,
                parameter_count=trained.parameter_count,
                metrics=metrics,
            )
        )
        oof_indices.append(validation)
        oof_variance.append(predictions.variance)
        oof_location.append(predictions.return_location)
        oof_direction.append(predictions.direction_probabilities)
        del trained
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    indices = np.concatenate(oof_indices)
    order = np.lexsort((examples.tickers[indices], examples.origin_dates[indices]))
    indices = indices[order]
    pooled_predictions = DistributionPredictions(
        variance=np.concatenate(oof_variance, axis=0)[order],
        return_location=np.concatenate(oof_location, axis=0)[order],
        direction_probabilities=np.concatenate(oof_direction, axis=0)[order],
    )
    pooled_metrics = tuple(
        horizon_distribution_metrics(
            predictions=pooled_predictions,
            baseline_variance=examples.baseline_variance[indices],
            realized_variance=examples.realized_variance[indices],
            cumulative_returns=examples.cumulative_returns[indices],
            direction_classes=examples.direction_classes[indices],
            horizons=protocol.horizons,
        )
    )
    candidate_losses = qlike_losses(
        pooled_predictions.variance,
        examples.realized_variance[indices],
    )
    baseline_losses = qlike_losses(
        examples.baseline_variance[indices],
        examples.realized_variance[indices],
    )
    promotion = assess_promotion(
        pooled_metrics=pooled_metrics,
        fold_metrics=tuple(fold_metric_rows),
        candidate_qlike_losses=candidate_losses,
        baseline_qlike_losses=baseline_losses,
        horizons=protocol.horizons,
        gate=promotion_gate,
        resamples=resamples,
        seed=seed,
    )
    return DevelopmentEvaluation(
        protocol_version=protocol.protocol_version,
        seed=seed,
        folds=tuple(fold_evidence),
        pooled_metrics=pooled_metrics,
        promotion=promotion,
        oof_indices=indices,
        predictions=pooled_predictions,
    )
