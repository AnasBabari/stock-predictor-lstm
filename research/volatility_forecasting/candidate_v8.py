"""Real CUDA-trainable v8 development candidates.

The module intentionally receives only train and validation identities.  It
does not accept temporal-test or asset-transfer indices, which keeps model
fitting, calibration, and selection structurally separated from one-shot
certification.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .baselines import fit_adaptive_variance_baseline, predict_adaptive_variance_baseline
from .data import VolatilityPanelExamples
from .evaluation import (
    cluster_losses_by_session,
    moving_block_ratio_upper_bound,
    predict_distribution,
)
from .folds import InnerTrainingSplit
from .metrics import (
    fit_crps_variance_scale,
    fit_qlike_variance_scale,
    horizon_distribution_metrics,
    qlike_losses,
)
from .model import (
    BaselineResidualTCNConfig,
    TorchTrainingConfig,
    VolatilityLossWeights,
    train_baseline_residual_tcn,
)
from .refit import FrozenCandidate, FrozenEnsemble, candidate_identity


@dataclass(frozen=True)
class V8ValidationPartitions:
    calibration_indices: np.ndarray
    selection_indices: np.ndarray
    calibration_end: np.datetime64
    selection_start: np.datetime64


@dataclass(frozen=True)
class V8MemberEvidence:
    seed: int
    eligible: bool
    best_epoch: int
    duration_seconds: float
    metrics: tuple[dict[str, float | int], ...]
    ratio_upper_95: tuple[float, ...]
    reasons: tuple[str, ...]


def v8_ensemble_identity(members: tuple[FrozenCandidate, ...]) -> str:
    """Content identity for one ordered v8 fixed-seed ensemble."""
    ordered = tuple(sorted(members, key=lambda member: member.seed))
    if not ordered or len({member.seed for member in ordered}) != len(ordered):
        raise ValueError("v8 ensemble members must have unique seeds")
    identity_payload = json.dumps(
        [(member.seed, member.model_identity) for member in ordered],
        separators=(",", ":"),
    ).encode("utf-8")
    families = {member.architecture.news_feature_count > 0 for member in ordered}
    if len(families) != 1:
        raise ValueError("v8 ensemble cannot mix market-only and news-enabled members")
    family = "news-transfer" if families == {True} else "numeric"
    return f"global-volatility-v8-{family}:{hashlib.sha256(identity_payload).hexdigest()}"


def split_validation_for_selection(
    examples: VolatilityPanelExamples,
    validation_indices: np.ndarray,
    *,
    calibration_fraction: float = 0.65,
    embargo_sessions: int = 30,
) -> V8ValidationPartitions:
    """Create disjoint calibration and untouched selection regions inside validation."""
    indices = np.asarray(validation_indices, dtype=np.int64)
    if indices.ndim != 1 or not len(indices) or len(np.unique(indices)) != len(indices):
        raise ValueError("validation indices must be a non-empty unique vector")
    if not 0.5 <= calibration_fraction <= 0.8:
        raise ValueError("calibration fraction must be in [0.5, 0.8]")
    if embargo_sessions < max(examples.horizons):
        raise ValueError("validation embargo must cover the maximum horizon")
    dates = np.unique(examples.origin_dates[indices])
    dates.sort()
    boundary = int(len(dates) * calibration_fraction)
    calibration_end_position = boundary - max(examples.horizons) - 1
    selection_start_position = boundary + embargo_sessions
    if calibration_end_position < 5 or selection_start_position >= len(dates) - 5:
        raise ValueError("validation region is too short for calibration, purge, and selection")
    calibration_end = dates[calibration_end_position]
    selection_start = dates[selection_start_position]
    row_dates = examples.origin_dates[indices]
    calibration = indices[row_dates <= calibration_end]
    selection = indices[row_dates >= selection_start]
    if not len(calibration) or not len(selection) or np.intersect1d(calibration, selection).size:
        raise RuntimeError("internal validation partitioning failed")
    return V8ValidationPartitions(
        calibration_indices=calibration,
        selection_indices=selection,
        calibration_end=calibration_end,
        selection_start=selection_start,
    )


def _fit_member(
    *,
    examples: VolatilityPanelExamples,
    train_indices: np.ndarray,
    partitions: V8ValidationPartitions,
    architecture: BaselineResidualTCNConfig,
    seed: int,
    device: str,
    maximum_epochs: int,
    patience: int,
    batch_size: int,
    loss_weights: VolatilityLossWeights,
    training_config: TorchTrainingConfig | None = None,
    news_features: np.ndarray | None = None,
) -> FrozenCandidate:
    train = np.asarray(train_indices, dtype=np.int64)
    calibration = partitions.calibration_indices
    settings = training_config or TorchTrainingConfig(
        maximum_epochs=maximum_epochs,
        patience=patience,
        batch_size=batch_size,
        use_amp=device == "cuda",
    )
    training = train_baseline_residual_tcn(
        train_features=examples.features[train],
        train_baseline_variance=examples.baseline_variance[train],
        train_realized_variance=examples.realized_variance[train],
        train_cumulative_returns=examples.cumulative_returns[train],
        train_direction_classes=examples.direction_classes[train],
        validation_features=examples.features[calibration],
        validation_baseline_variance=examples.baseline_variance[calibration],
        validation_realized_variance=examples.realized_variance[calibration],
        validation_cumulative_returns=examples.cumulative_returns[calibration],
        validation_direction_classes=examples.direction_classes[calibration],
        train_news_features=news_features[train] if news_features is not None else None,
        validation_news_features=(
            news_features[calibration] if news_features is not None else None
        ),
        model_config=architecture,
        training_config=settings,
        loss_weights=loss_weights,
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
    calibrated = raw.variance * variance_scale
    return_variance_scale = fit_crps_variance_scale(
        calibrated,
        examples.cumulative_returns[calibration],
    )
    comparison = fit_adaptive_variance_baseline(examples, calibration)
    baseline = predict_adaptive_variance_baseline(examples, calibration, comparison)
    baseline_return_scale = fit_crps_variance_scale(
        baseline,
        examples.cumulative_returns[calibration],
    )
    fit_split = InnerTrainingSplit(
        fit_indices=train,
        early_stopping_indices=calibration,
        fit_end=np.max(examples.origin_dates[train]),
        early_stopping_start=np.min(examples.origin_dates[calibration]),
        early_stopping_end=np.max(examples.origin_dates[calibration]),
    )
    identity = candidate_identity(
        training,
        architecture=architecture,
        seed=seed,
        epoch_budget=settings.maximum_epochs,
        variance_scale=variance_scale,
        return_variance_scale=return_variance_scale,
        comparison_baseline=comparison,
        baseline_return_variance_scale=baseline_return_scale,
        loss_weights=loss_weights,
    )
    return FrozenCandidate(
        training=training,
        architecture=architecture,
        fit_split=fit_split,
        seed=seed,
        epoch_budget=settings.maximum_epochs,
        variance_scale=variance_scale,
        return_variance_scale=return_variance_scale,
        comparison_baseline=comparison,
        baseline_return_variance_scale=baseline_return_scale,
        model_identity=identity,
        loss_weights=loss_weights,
    )


def _evaluate_member(
    candidate: FrozenCandidate,
    examples: VolatilityPanelExamples,
    indices: np.ndarray,
    *,
    required_horizons: tuple[int, ...],
    maximum_relative_qlike: float = 0.98,
    maximum_ratio_upper_95: float = 1.0,
    news_features: np.ndarray | None = None,
) -> V8MemberEvidence:
    predictions = candidate.predict(examples, indices, news_features=news_features)
    baseline, baseline_return = candidate.matched_baselines(examples, indices)
    metrics = tuple(
        horizon_distribution_metrics(
            predictions=predictions,
            baseline_variance=baseline,
            baseline_return_variance=baseline_return,
            realized_variance=examples.realized_variance[indices],
            cumulative_returns=examples.cumulative_returns[indices],
            direction_classes=examples.direction_classes[indices],
            horizons=examples.horizons,
        )
    )
    candidate_losses = qlike_losses(predictions.variance, examples.realized_variance[indices])
    baseline_losses = qlike_losses(baseline, examples.realized_variance[indices])
    candidate_sessions, candidate_dates = cluster_losses_by_session(
        candidate_losses,
        examples.origin_dates[indices],
    )
    baseline_sessions, baseline_dates = cluster_losses_by_session(
        baseline_losses,
        examples.origin_dates[indices],
    )
    if not np.array_equal(candidate_dates, baseline_dates):
        raise RuntimeError("validation loss sessions are misaligned")
    upper = tuple(
        moving_block_ratio_upper_bound(
            candidate_sessions[:, column],
            baseline_sessions[:, column],
            resamples=1000,
            block_length=max(5, horizon),
            seed=20260827 + candidate.seed + column,
        )
        for column, horizon in enumerate(examples.horizons)
    )
    reasons: list[str] = []
    for column, horizon in enumerate(examples.horizons):
        if horizon not in required_horizons:
            continue
        relative = float(metrics[column]["relative_qlike"])
        if relative > maximum_relative_qlike:
            reasons.append(f"h{horizon} relative QLIKE {relative:.6f} > {maximum_relative_qlike}")
        if upper[column] > maximum_ratio_upper_95:
            reasons.append(
                f"h{horizon} QLIKE ratio upper95 {upper[column]:.6f} > {maximum_ratio_upper_95}"
            )
    return V8MemberEvidence(
        seed=candidate.seed,
        eligible=not reasons,
        best_epoch=candidate.training.best_epoch,
        duration_seconds=candidate.training.duration_seconds,
        metrics=metrics,
        ratio_upper_95=upper,
        reasons=tuple(reasons),
    )


def train_v8_ensemble(
    *,
    examples: VolatilityPanelExamples,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    seeds: tuple[int, ...],
    required_horizons: tuple[int, ...],
    device: str = "cuda",
    maximum_epochs: int = 60,
    patience: int = 8,
    batch_size: int = 512,
    architecture: BaselineResidualTCNConfig | None = None,
    loss_weights: VolatilityLossWeights | None = None,
    training_config: TorchTrainingConfig | None = None,
    news_features: np.ndarray | None = None,
) -> tuple[FrozenEnsemble, tuple[V8MemberEvidence, ...], V8ValidationPartitions]:
    partitions = split_validation_for_selection(examples, validation_indices)
    architecture = architecture or BaselineResidualTCNConfig(
        feature_count=examples.features.shape[-1],
        horizon_count=len(examples.horizons),
        window_size=examples.features.shape[1],
    )
    if architecture.news_feature_count:
        if news_features is None or news_features.shape != (
            len(examples.features),
            architecture.news_feature_count,
        ):
            raise ValueError("news candidate requires one aligned feature row per example")
        if not np.isfinite(news_features).all():
            raise ValueError("news candidate features must be finite")
    elif news_features is not None:
        raise ValueError("market-only candidate cannot receive news features")
    weights = loss_weights or VolatilityLossWeights()
    members: list[FrozenCandidate] = []
    evidence: list[V8MemberEvidence] = []
    for seed in seeds:
        member = _fit_member(
            examples=examples,
            train_indices=train_indices,
            partitions=partitions,
            architecture=architecture,
            seed=seed,
            device=device,
            maximum_epochs=maximum_epochs,
            patience=patience,
            batch_size=batch_size,
            loss_weights=weights,
            training_config=training_config,
            news_features=news_features,
        )
        members.append(member)
        evidence.append(
            _evaluate_member(
                member,
                examples,
                partitions.selection_indices,
                required_horizons=required_horizons,
                news_features=news_features,
            )
        )
    ordered = tuple(sorted(members, key=lambda member: member.seed))
    model_identity = v8_ensemble_identity(ordered)
    ensemble = FrozenEnsemble(members=ordered, model_identity=model_identity)
    return ensemble, tuple(evidence), partitions


def train_v8_numeric_ensemble(
    *,
    examples: VolatilityPanelExamples,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    seeds: tuple[int, ...],
    required_horizons: tuple[int, ...],
    device: str = "cuda",
    maximum_epochs: int = 60,
    patience: int = 8,
    batch_size: int = 512,
    architecture: BaselineResidualTCNConfig | None = None,
    loss_weights: VolatilityLossWeights | None = None,
    training_config: TorchTrainingConfig | None = None,
    news_features: np.ndarray | None = None,
) -> tuple[FrozenEnsemble, tuple[V8MemberEvidence, ...], V8ValidationPartitions]:
    """Compatibility entry point that rejects accidental news fusion."""

    if news_features is not None:
        raise ValueError("numeric v8 training cannot accept news features")
    if architecture is not None and architecture.news_feature_count:
        raise ValueError("numeric v8 training requires a market-only architecture")
    return train_v8_ensemble(
        examples=examples,
        train_indices=train_indices,
        validation_indices=validation_indices,
        seeds=seeds,
        required_horizons=required_horizons,
        device=device,
        maximum_epochs=maximum_epochs,
        patience=patience,
        batch_size=batch_size,
        architecture=architecture,
        loss_weights=loss_weights,
        training_config=training_config,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _persist_member_rows(
    output: Path,
    *,
    ensemble: FrozenEnsemble,
    evidence: tuple[V8MemberEvidence, ...],
    filename_prefix: str = "",
) -> list[dict[str, object]]:
    evidence_by_seed = {row.seed: row for row in evidence}
    member_seeds = {member.seed for member in ensemble.members}
    if member_seeds != set(evidence_by_seed):
        raise ValueError("candidate members and validation evidence have different seeds")
    rows: list[dict[str, object]] = []
    for member in ensemble.members:
        weights_name = f"{filename_prefix}seed-{member.seed}.pt"
        weights_path = output / weights_name
        torch.save(member.training.model.state_dict(), weights_path)
        rows.append(
            {
                "seed": member.seed,
                "model_identity": member.model_identity,
                "weights_file": weights_name,
                "weights_sha256": _sha256_file(weights_path),
                "epoch_budget": member.epoch_budget,
                "best_epoch": member.training.best_epoch,
                "market_scaler": member.training.scaler.to_dict(),
                "news_scaler": (
                    member.training.news_scaler.to_dict()
                    if member.training.news_scaler is not None
                    else None
                ),
                "variance_scale": member.variance_scale.tolist(),
                "return_variance_scale": member.return_variance_scale.tolist(),
                "baseline_return_variance_scale": member.baseline_return_variance_scale.tolist(),
                "comparison_baseline": [
                    asdict(value) for value in member.comparison_baseline.horizons
                ],
                "loss_weights": asdict(member.loss_weights),
                "fit_end": str(member.fit_split.fit_end),
                "calibration_start": str(member.fit_split.early_stopping_start),
                "calibration_end": str(member.fit_split.early_stopping_end),
                "validation_evidence": asdict(evidence_by_seed[member.seed]),
            }
        )
    return rows


def save_v8_development_candidate(
    output: Path,
    *,
    ensemble: FrozenEnsemble,
    evidence: tuple[V8MemberEvidence, ...],
    protocol: dict[str, object],
    split_manifest: dict[str, object],
    split_manifest_sha256: str,
    panel_checksum: str,
    universe_manifest_sha256: str,
    news_snapshot_checksum: str,
    universe_certifiable: bool,
    training_config: dict[str, object] | None = None,
    news_matrix_sha256: str | None = None,
    news_feature_names: tuple[str, ...] = (),
    news_ablation_evidence: tuple[dict[str, object], ...] | None = None,
    numeric_companion_ensemble: FrozenEnsemble | None = None,
    numeric_companion_evidence: tuple[V8MemberEvidence, ...] | None = None,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError("candidate output must be a new immutable directory")
    is_news_candidate = ensemble.members[0].architecture.news_feature_count > 0
    if is_news_candidate:
        if not news_matrix_sha256 or not news_feature_names or not news_ablation_evidence:
            raise ValueError(
                "news candidate requires matrix identity, schema, and ablation evidence"
            )
        if numeric_companion_ensemble is None or numeric_companion_evidence is None:
            raise ValueError("news candidate requires a frozen numeric companion")
        if any(
            member.architecture.news_feature_count
            for member in numeric_companion_ensemble.members
        ):
            raise ValueError("numeric companion cannot contain news features")
        if {member.seed for member in numeric_companion_ensemble.members} != {
            member.seed for member in ensemble.members
        }:
            raise ValueError("news and numeric companion seeds must match")
        news_promoted = all(row.get("promoted") is True for row in news_ablation_evidence)
    else:
        if (
            news_matrix_sha256
            or news_feature_names
            or news_ablation_evidence
            or numeric_companion_ensemble is not None
            or numeric_companion_evidence is not None
        ):
            raise ValueError("market-only candidate cannot persist news evidence")
        news_promoted = True
    # All argument validation completes before the first filesystem mutation so
    # a rejected candidate never leaves a partially populated artifact directory.
    output.mkdir(parents=True)
    member_rows = _persist_member_rows(output, ensemble=ensemble, evidence=evidence)
    companion_ensemble = numeric_companion_ensemble if is_news_candidate else None
    companion_evidence = numeric_companion_evidence if is_news_candidate else None
    companion_payload = None
    companion_selected = True
    if companion_ensemble is not None and companion_evidence is not None:
        companion_rows = _persist_member_rows(
            output,
            ensemble=companion_ensemble,
            evidence=companion_evidence,
            filename_prefix="numeric-",
        )
        companion_selected = all(row.eligible for row in companion_evidence)
        companion_payload = {
            "model_identity": companion_ensemble.model_identity,
            "architecture": asdict(companion_ensemble.members[0].architecture),
            "members": companion_rows,
            "validation_selected": companion_selected,
            "role": "predeclared_numeric_fallback_companion",
        }
    eligible = (
        universe_certifiable
        and all(row.eligible for row in evidence)
        and news_promoted
        and companion_selected
    )
    role = (
        "prospective_v8_development_candidate" if eligible else "rejected_v8_development_evidence"
    )
    manifest: dict[str, object] = {
        "artifact_role": role,
        "release_eligible": False,
        "model_identity": ensemble.model_identity,
        "model_version": protocol["model_version"],
        "protocol_version": protocol["protocol_version"],
        "protocol": protocol,
        "architecture": asdict(ensemble.members[0].architecture),
        "training_config": training_config,
        "members": member_rows,
        "panel_checksum": panel_checksum,
        "universe_manifest_sha256": universe_manifest_sha256,
        "news_snapshot_checksum": news_snapshot_checksum,
        "news_matrix_sha256": news_matrix_sha256,
        "news_feature_names": list(news_feature_names),
        "news_ablation_evidence": news_ablation_evidence,
        "numeric_companion": companion_payload,
        "split_manifest": split_manifest,
        "split_manifest_sha256": split_manifest_sha256,
        "universe_certifiable": universe_certifiable,
        "validation_selected": (
            all(row.eligible for row in evidence) and news_promoted and companion_selected
        ),
        "strict_release_policy": {
            "unsigned": True,
            "sealed_test_required": True,
            "partial_release_allowed": False,
            "placeholder_members_allowed": False,
        },
    }
    (output / "candidate-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return manifest
