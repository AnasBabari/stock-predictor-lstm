"""Fold execution and fail-closed promotion for the volatility TCN."""

from __future__ import annotations

import gc
from dataclasses import dataclass

import numpy as np
import torch

from backend.panel.selection import diebold_mariano_hac, holm_correction

from .contracts import VolatilityForecastProtocol, VolatilityPromotionGate
from .data import VolatilityPanelExamples
from .folds import VolatilityFoldPlan, build_inner_training_split
from .metrics import (
    DistributionPredictions,
    fit_crps_variance_scale,
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
    fit_rows: int
    early_stopping_rows: int
    fit_end: str
    early_stopping_start: str
    early_stopping_end: str
    validation_start: str
    validation_end: str
    rows: int
    best_epoch: int
    duration_seconds: float
    parameter_count: int
    return_variance_scale: tuple[float, ...]
    baseline_return_variance_scale: tuple[float, ...]
    training_history: tuple[dict[str, float], ...]
    metrics: tuple[dict[str, float | int], ...]


@dataclass(frozen=True)
class HorizonPromotionDecision:
    horizon: int
    volatility_promoted: bool
    return_distribution_promoted: bool
    return_location_promoted: bool
    direction_promoted: bool
    promoted: bool
    reasons: tuple[str, ...]
    return_distribution_reasons: tuple[str, ...]
    return_location_reasons: tuple[str, ...]
    direction_reasons: tuple[str, ...]
    folds_beating_baseline: int
    return_folds_beating_baseline: int
    worst_fold_relative_qlike: float
    relative_qlike: float
    relative_qlike_upper_95: float
    relative_gaussian_crps: float
    relative_variance_only_gaussian_crps: float
    relative_return_mae: float
    relative_return_rmse: float
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
    """Promote variance independently from auxiliary return and direction heads."""
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
        relative_full_model_crps = float(pooled["relative_gaussian_crps"])
        relative_variance_crps = float(pooled["relative_variance_only_gaussian_crps"])
        relative_return_mae = float(pooled["relative_return_mae"])
        relative_return_rmse = float(pooled["relative_return_rmse"])
        coverage_80 = float(pooled["variance_only_coverage_80"])
        dm_stat, dm_p = dm_rows[column]
        volatility_reasons: list[str] = []
        if relative_qlike >= settings.maximum_relative_qlike:
            volatility_reasons.append("pooled relative QLIKE did not clear the improvement gate")
        if upper_bounds[column] >= 1.0:
            volatility_reasons.append("bootstrap upper confidence bound did not beat the baseline")
        if folds_beating < settings.minimum_folds_beating_baseline:
            volatility_reasons.append("too few expanding folds beat the matched baseline")
        if worst_fold > settings.maximum_worst_fold_relative_qlike:
            volatility_reasons.append("worst fold exceeded the stability guardrail")
        distribution_reasons: list[str] = []
        if relative_variance_crps >= settings.maximum_relative_variance_only_crps:
            distribution_reasons.append(
                "candidate variance CRPS around the zero-return baseline did not improve"
            )
        if (
            not settings.minimum_interval_coverage_80
            <= coverage_80
            <= settings.maximum_interval_coverage_80
        ):
            distribution_reasons.append(
                "zero-centred 80% interval coverage is outside the calibration band"
            )
        if dm_stat >= 0 or not holm[column]:
            volatility_reasons.append("paired QLIKE improvement is not Holm-significant")

        return_fold_relative = [
            (
                float(metrics[column]["relative_return_mae"]),
                float(metrics[column]["relative_return_rmse"]),
            )
            for metrics in fold_metrics
        ]
        return_folds_beating = sum(mae < 1.0 and rmse < 1.0 for mae, rmse in return_fold_relative)
        return_reasons: list[str] = []
        if relative_return_mae >= settings.maximum_relative_return_mae:
            return_reasons.append("return-location MAE did not beat zero return")
        if relative_return_rmse >= settings.maximum_relative_return_rmse:
            return_reasons.append("return-location RMSE did not beat zero return")
        if return_folds_beating < settings.minimum_return_folds_beating_baseline:
            return_reasons.append("return location was not stable across expanding folds")

        # Direction remains an auxiliary diagnostic until its matched
        # pre-evaluation class-prevalence baseline is threaded through every
        # fold. It must fail closed rather than inherit volatility promotion.
        direction_reasons = (
            "direction head is diagnostic until its matched pre-evaluation baseline gate is implemented",
        )
        volatility_promoted = not volatility_reasons
        decisions.append(
            HorizonPromotionDecision(
                horizon=horizon,
                volatility_promoted=volatility_promoted,
                return_distribution_promoted=not distribution_reasons,
                return_location_promoted=not return_reasons,
                direction_promoted=False,
                promoted=volatility_promoted,
                reasons=tuple(volatility_reasons),
                return_distribution_reasons=tuple(distribution_reasons),
                return_location_reasons=tuple(return_reasons),
                direction_reasons=direction_reasons,
                folds_beating_baseline=folds_beating,
                return_folds_beating_baseline=return_folds_beating,
                worst_fold_relative_qlike=worst_fold,
                relative_qlike=relative_qlike,
                relative_qlike_upper_95=upper_bounds[column],
                relative_gaussian_crps=relative_full_model_crps,
                relative_variance_only_gaussian_crps=relative_variance_crps,
                relative_return_mae=relative_return_mae,
                relative_return_rmse=relative_return_rmse,
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
    oof_return_variance: list[np.ndarray] = []
    oof_baseline_return_variance: list[np.ndarray] = []

    for fold in fold_plan.folds:
        inner = build_inner_training_split(examples, fold.train_indices, protocol)
        train = inner.fit_indices
        early_stopping = inner.early_stopping_indices
        validation = fold.validation_indices
        trained = train_baseline_residual_tcn(
            train_features=examples.features[train],
            train_baseline_variance=examples.baseline_variance[train],
            train_realized_variance=examples.realized_variance[train],
            train_cumulative_returns=examples.cumulative_returns[train],
            train_direction_classes=examples.direction_classes[train],
            validation_features=examples.features[early_stopping],
            validation_baseline_variance=examples.baseline_variance[early_stopping],
            validation_realized_variance=examples.realized_variance[early_stopping],
            validation_cumulative_returns=examples.cumulative_returns[early_stopping],
            validation_direction_classes=examples.direction_classes[early_stopping],
            model_config=architecture,
            training_config=training_config,
            loss_weights=loss_weights,
            seed=seed,
            device=device,
        )
        calibration_predictions = predict_distribution(trained, examples, early_stopping)
        return_variance_scale = fit_crps_variance_scale(
            calibration_predictions.variance,
            examples.cumulative_returns[early_stopping],
        )
        baseline_return_variance_scale = fit_crps_variance_scale(
            examples.baseline_variance[early_stopping],
            examples.cumulative_returns[early_stopping],
        )
        raw_predictions = predict_distribution(trained, examples, validation)
        predictions = DistributionPredictions(
            variance=raw_predictions.variance,
            return_location=raw_predictions.return_location,
            direction_probabilities=raw_predictions.direction_probabilities,
            return_variance=raw_predictions.variance * return_variance_scale,
        )
        baseline_return_variance = (
            examples.baseline_variance[validation] * baseline_return_variance_scale
        )
        metrics = tuple(
            horizon_distribution_metrics(
                predictions=predictions,
                baseline_variance=examples.baseline_variance[validation],
                baseline_return_variance=baseline_return_variance,
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
                fit_rows=len(train),
                early_stopping_rows=len(early_stopping),
                fit_end=str(inner.fit_end),
                early_stopping_start=str(inner.early_stopping_start),
                early_stopping_end=str(inner.early_stopping_end),
                validation_start=str(fold.validation_start),
                validation_end=str(fold.validation_end),
                rows=len(validation),
                best_epoch=trained.best_epoch,
                duration_seconds=trained.duration_seconds,
                parameter_count=trained.parameter_count,
                return_variance_scale=tuple(float(value) for value in return_variance_scale),
                baseline_return_variance_scale=tuple(
                    float(value) for value in baseline_return_variance_scale
                ),
                training_history=trained.history,
                metrics=metrics,
            )
        )
        oof_indices.append(validation)
        oof_variance.append(predictions.variance)
        oof_location.append(predictions.return_location)
        oof_direction.append(predictions.direction_probabilities)
        oof_return_variance.append(predictions.return_variance)
        oof_baseline_return_variance.append(baseline_return_variance)
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
        return_variance=np.concatenate(oof_return_variance, axis=0)[order],
    )
    pooled_baseline_return_variance = np.concatenate(oof_baseline_return_variance, axis=0)[order]
    pooled_metrics = tuple(
        horizon_distribution_metrics(
            predictions=pooled_predictions,
            baseline_variance=examples.baseline_variance[indices],
            baseline_return_variance=pooled_baseline_return_variance,
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
