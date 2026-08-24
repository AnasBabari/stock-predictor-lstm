"""Freeze a development-selected candidate before opening locked reserves.

The release candidate is fitted only on development assets and dates.  A
purged terminal development slice is used for early stopping and output
calibration; neither the temporal certification year nor unseen assets can
influence weights, scaler statistics, baseline selection, or calibration.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import asdict, dataclass

import numpy as np
import torch

from .baselines import (
    AdaptiveBaselineSelection,
    fit_adaptive_variance_baseline,
    predict_adaptive_variance_baseline,
)
from .contracts import VolatilityForecastProtocol
from .data import VolatilityPanelExamples
from .evaluation import predict_distribution
from .folds import InnerTrainingSplit, VolatilityFoldPlan, build_inner_training_split
from .metrics import (
    DistributionPredictions,
    fit_crps_variance_scale,
    fit_qlike_variance_scale,
)
from .model import (
    BaselineResidualTCNConfig,
    TorchTrainingConfig,
    TrainingResult,
    train_baseline_residual_tcn,
)


@dataclass(frozen=True)
class FrozenCandidate:
    """Exact pre-certification model plus pre-certification calibration state."""

    training: TrainingResult
    architecture: BaselineResidualTCNConfig
    fit_split: InnerTrainingSplit
    seed: int
    epoch_budget: int
    variance_scale: np.ndarray
    return_variance_scale: np.ndarray
    comparison_baseline: AdaptiveBaselineSelection
    baseline_return_variance_scale: np.ndarray
    model_identity: str

    def predict(
        self,
        examples: VolatilityPanelExamples,
        indices: np.ndarray,
        *,
        news_features: np.ndarray | None = None,
    ) -> DistributionPredictions:
        raw = predict_distribution(
            self.training,
            examples,
            indices,
            news_features=news_features,
        )
        variance = raw.variance * self.variance_scale
        return_variance = variance * self.return_variance_scale
        return DistributionPredictions(
            variance=variance,
            return_location=raw.return_location,
            direction_probabilities=raw.direction_probabilities,
            return_variance=return_variance,
        )

    def matched_baselines(
        self,
        examples: VolatilityPanelExamples,
        indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        variance = predict_adaptive_variance_baseline(
            examples,
            indices,
            self.comparison_baseline,
        )
        return variance, variance * self.baseline_return_variance_scale


@dataclass(frozen=True)
class FrozenEnsemble:
    """One production candidate composed of fixed-seed members."""

    members: tuple[FrozenCandidate, ...]
    model_identity: str

    def __post_init__(self) -> None:
        if not self.members:
            raise ValueError("frozen ensemble requires at least one member")
        seeds = tuple(member.seed for member in self.members)
        if tuple(sorted(set(seeds))) != seeds:
            raise ValueError("ensemble members must have unique increasing seeds")

    def predict(
        self,
        examples: VolatilityPanelExamples,
        indices: np.ndarray,
        *,
        news_features: np.ndarray | None = None,
    ) -> DistributionPredictions:
        predictions = [
            member.predict(examples, indices, news_features=news_features)
            for member in self.members
        ]
        return DistributionPredictions(
            variance=np.mean([item.variance for item in predictions], axis=0),
            return_location=np.mean([item.return_location for item in predictions], axis=0),
            direction_probabilities=np.mean(
                [item.direction_probabilities for item in predictions],
                axis=0,
            ),
            return_variance=np.mean([item.return_variance for item in predictions], axis=0),
        )

    def matched_baselines(
        self,
        examples: VolatilityPanelExamples,
        indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        baselines = [member.matched_baselines(examples, indices) for member in self.members]
        reference_variance, reference_return = baselines[0]
        if any(
            not np.allclose(variance, reference_variance, rtol=1e-10, atol=1e-12)
            or not np.allclose(return_variance, reference_return, rtol=1e-10, atol=1e-12)
            for variance, return_variance in baselines[1:]
        ):
            raise RuntimeError("ensemble members do not share one matched baseline contract")
        return reference_variance, reference_return


def ensemble_identity(members: tuple[FrozenCandidate, ...]) -> str:
    if not members:
        raise ValueError("cannot identify an empty ensemble")
    ordered = sorted((member.seed, member.model_identity) for member in members)
    if len({seed for seed, _identity in ordered}) != len(ordered):
        raise ValueError("ensemble member seeds must be unique")
    payload = json.dumps(ordered, separators=(",", ":")).encode("utf-8")
    return f"global-volatility-ensemble:{hashlib.sha256(payload).hexdigest()}"


def derive_epoch_budget(development_record: dict[str, object]) -> int:
    """Derive a robust fixed budget from the five untouched fold choices."""
    folds = development_record.get("folds")
    if not isinstance(folds, list) or len(folds) < 3:
        raise ValueError("development evidence must contain at least three folds")
    epochs: list[int] = []
    for fold in folds:
        if not isinstance(fold, dict):
            raise ValueError("development fold evidence is malformed")
        value = fold.get("best_epoch")
        if not isinstance(value, int) or value < 1:
            raise ValueError("development fold has an invalid best epoch")
        epochs.append(value)
    # Lower median for an even count is deterministic and conservative.
    ordered = sorted(epochs)
    return ordered[(len(ordered) - 1) // 2]


def certification_development_split(
    examples: VolatilityPanelExamples,
    fold_plan: VolatilityFoldPlan,
    protocol: VolatilityForecastProtocol,
) -> InnerTrainingSplit:
    """Return the only rows eligible for pre-certification fitting/calibration."""
    eligible = np.flatnonzero(
        np.isin(examples.tickers, fold_plan.train_tickers)
        & (examples.origin_dates < fold_plan.certification_start)
    )
    if len(eligible) == 0:
        raise ValueError("no development rows precede the certification boundary")
    return build_inner_training_split(examples, eligible, protocol)


def _model_identity(
    training: TrainingResult,
    *,
    architecture: BaselineResidualTCNConfig,
    seed: int,
    epoch_budget: int,
    variance_scale: np.ndarray,
    return_variance_scale: np.ndarray,
    comparison_baseline: AdaptiveBaselineSelection,
    baseline_return_variance_scale: np.ndarray,
) -> str:
    state = io.BytesIO()
    torch.save(training.model.state_dict(), state)
    metadata = {
        "architecture": asdict(architecture),
        "seed": seed,
        "epoch_budget": epoch_budget,
        "market_scaler": training.scaler.to_dict(),
        "news_scaler": training.news_scaler.to_dict() if training.news_scaler else None,
        "variance_scale": variance_scale.tolist(),
        "return_variance_scale": return_variance_scale.tolist(),
        "comparison_baseline": [asdict(value) for value in comparison_baseline.horizons],
        "baseline_return_variance_scale": baseline_return_variance_scale.tolist(),
    }
    payload = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(state.getvalue() + payload).hexdigest()
    return f"global-volatility:{digest}"


def fit_frozen_candidate(
    *,
    examples: VolatilityPanelExamples,
    fold_plan: VolatilityFoldPlan,
    protocol: VolatilityForecastProtocol,
    development_record: dict[str, object],
    architecture: BaselineResidualTCNConfig,
    seed: int,
    device: str,
    batch_size: int = 512,
    news_features: np.ndarray | None = None,
) -> FrozenCandidate:
    """Fit and calibrate the exact candidate that may enter locked certification."""
    epoch_budget = derive_epoch_budget(development_record)
    split = certification_development_split(examples, fold_plan, protocol)
    fit = split.fit_indices
    calibration = split.early_stopping_indices
    if architecture.news_feature_count:
        if news_features is None or news_features.shape != (
            len(examples.features),
            architecture.news_feature_count,
        ):
            raise ValueError("news candidate requires one aligned feature row per example")
    elif news_features is not None:
        raise ValueError("market-only candidate cannot receive news features")
    training = train_baseline_residual_tcn(
        train_features=examples.features[fit],
        train_baseline_variance=examples.baseline_variance[fit],
        train_realized_variance=examples.realized_variance[fit],
        train_cumulative_returns=examples.cumulative_returns[fit],
        train_direction_classes=examples.direction_classes[fit],
        validation_features=examples.features[calibration],
        validation_baseline_variance=examples.baseline_variance[calibration],
        validation_realized_variance=examples.realized_variance[calibration],
        validation_cumulative_returns=examples.cumulative_returns[calibration],
        validation_direction_classes=examples.direction_classes[calibration],
        train_news_features=news_features[fit] if news_features is not None else None,
        validation_news_features=(
            news_features[calibration] if news_features is not None else None
        ),
        model_config=architecture,
        training_config=TorchTrainingConfig(
            maximum_epochs=epoch_budget,
            patience=max(epoch_budget, 1),
            batch_size=batch_size,
            use_amp=device == "cuda",
        ),
        seed=seed,
        device=device,
    )
    raw = predict_distribution(
        training,
        examples,
        calibration,
        news_features=news_features,
    )
    variance_scale = fit_qlike_variance_scale(
        raw.variance,
        examples.realized_variance[calibration],
        session_labels=examples.origin_dates[calibration],
    )
    calibrated_variance = raw.variance * variance_scale
    return_variance_scale = fit_crps_variance_scale(
        calibrated_variance,
        examples.cumulative_returns[calibration],
    )
    comparison = fit_adaptive_variance_baseline(examples, calibration)
    baseline_variance = predict_adaptive_variance_baseline(examples, calibration, comparison)
    baseline_return_scale = fit_crps_variance_scale(
        baseline_variance,
        examples.cumulative_returns[calibration],
    )
    identity = _model_identity(
        training,
        architecture=architecture,
        seed=seed,
        epoch_budget=epoch_budget,
        variance_scale=variance_scale,
        return_variance_scale=return_variance_scale,
        comparison_baseline=comparison,
        baseline_return_variance_scale=baseline_return_scale,
    )
    return FrozenCandidate(
        training=training,
        architecture=architecture,
        fit_split=split,
        seed=seed,
        epoch_budget=epoch_budget,
        variance_scale=variance_scale,
        return_variance_scale=return_variance_scale,
        comparison_baseline=comparison,
        baseline_return_variance_scale=baseline_return_scale,
        model_identity=identity,
    )


def fit_frozen_ensemble(
    *,
    examples: VolatilityPanelExamples,
    fold_plan: VolatilityFoldPlan,
    protocol: VolatilityForecastProtocol,
    development_records: dict[int, dict[str, object]],
    architecture: BaselineResidualTCNConfig,
    device: str,
    batch_size: int = 512,
    news_features: np.ndarray | None = None,
) -> FrozenEnsemble:
    """Fit the fixed-seed ensemble selected by the frozen development report."""
    if tuple(sorted(development_records)) != protocol.seeds:
        raise ValueError(f"development records must cover frozen seeds {protocol.seeds}")
    members = tuple(
        fit_frozen_candidate(
            examples=examples,
            fold_plan=fold_plan,
            protocol=protocol,
            development_record=development_records[seed],
            architecture=architecture,
            seed=seed,
            device=device,
            batch_size=batch_size,
            news_features=news_features,
        )
        for seed in protocol.seeds
    )
    return FrozenEnsemble(members=members, model_identity=ensemble_identity(members))
