"""Shared walk-forward runner for offline baseline and candidate experiments."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from evaluation.evidence import paired_loss_evidence
from evaluation.metrics import evaluate_forecast_horizons
from evaluation.promotion import PromotionPolicy, assess_promotion
from evaluation.splits import purged_tail_split
from experiments.baselines import (
    DriftForecaster,
    HistogramGradientBoostingForecaster,
    PersistenceForecaster,
    RidgeForecaster,
)
from experiments.contracts import FoldPlan, build_experiment_dataset
from experiments.targets import TargetType, reconstruct_prices


@dataclass(frozen=True)
class ExperimentConfig:
    lookback: int = 60
    horizons: tuple[int, ...] = (1, 5, 20)
    target_type: TargetType = "log_return"
    folds: int = 5
    min_train_size: int = 300
    validation_size: int = 60
    gap: int | None = None
    method: str = "expanding"
    seed: int = 42

    @property
    def effective_gap(self) -> int:
        return max(self.horizons) if self.gap is None else self.gap


def _scale_windows(
    train_features: np.ndarray, validation_features: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    feature_count = train_features.shape[2]
    scaler = MinMaxScaler()
    scaler.fit(train_features.reshape(-1, feature_count))
    scaled_train = scaler.transform(train_features.reshape(-1, feature_count)).reshape(
        train_features.shape
    )
    scaled_validation = scaler.transform(validation_features.reshape(-1, feature_count)).reshape(
        validation_features.shape
    )
    return scaled_train, scaled_validation


def _round_metric_tree(value):
    if isinstance(value, dict):
        return {key: _round_metric_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_metric_tree(item) for item in value]
    if isinstance(value, float):
        return round(value, 8)
    return value


def run_baseline_experiment(
    feature_values,
    close_values,
    *,
    feature_names: list[str],
    config: ExperimentConfig | None = None,
    dates=None,
    snapshot_id: str | None = None,
    candidate_factories: tuple[Callable[[], object], ...] = (),
) -> dict:
    """Compare baselines on identical purged walk-forward observations."""

    selected = config or ExperimentConfig()
    if "Close" not in feature_names:
        raise ValueError("feature_names must identify the Close feature.")
    date_index = (
        pd.DatetimeIndex(dates)
        if dates is not None
        else pd.date_range("1970-01-01", periods=len(close_values), freq="D")
    )
    dataset = build_experiment_dataset(
        feature_values,
        close_values,
        dates=date_index,
        feature_names=feature_names,
        lookback=selected.lookback,
        horizons=selected.horizons,
        target_type=selected.target_type,
        snapshot_id=snapshot_id,
    )
    if len(feature_names) != dataset.features.shape[2]:
        raise ValueError("feature_names must match the feature matrix.")

    fold_plan = FoldPlan.create(
        dataset,
        folds=selected.folds,
        min_train_size=selected.min_train_size,
        validation_size=selected.validation_size,
        gap=selected.effective_gap,
        method=selected.method,
    )
    splits = [(fold.training_indices, fold.validation_indices) for fold in fold_plan.folds]
    model_reports: dict[str, dict] = {}
    pooled_rows: dict[str, dict[str, list[np.ndarray]]] = {}
    pooled_scale_end = int(dataset.origin_indices[splits[-1][0][-1]] + max(selected.horizons))
    pooled_scale_series = np.asarray(close_values, dtype=float)[: pooled_scale_end + 1]

    for fold_number, (training, validation) in enumerate(splits, start=1):
        raw_train = dataset.features[training]
        raw_validation = dataset.features[validation]
        scaled_train, scaled_validation = _scale_windows(raw_train, raw_validation)
        training_targets = dataset.targets[training]
        validation_origins = dataset.origins[validation]
        validation_actual = dataset.actual_prices[validation]
        fold_scale_end = int(dataset.origin_indices[training[-1]] + max(selected.horizons))
        fold_scale_series = np.asarray(close_values, dtype=float)[: fold_scale_end + 1]

        candidates = {
            "persistence": PersistenceForecaster().predict_prices(
                origins=validation_origins, horizons=selected.horizons
            ),
            "drift": DriftForecaster(feature_names.index("Close")).predict_prices(
                raw_validation,
                origins=validation_origins,
                horizons=selected.horizons,
            ),
        }
        for model in (
            RidgeForecaster().fit(scaled_train, training_targets),
            HistogramGradientBoostingForecaster(random_state=selected.seed).fit(
                scaled_train, training_targets
            ),
        ):
            predicted_targets = model.predict(scaled_validation)
            candidates[model.name] = reconstruct_prices(
                validation_origins,
                predicted_targets,
                selected.target_type,
            )

        # Neural (or other) candidates use the exact outer folds as every
        # baseline. Their early-stopping validation is a purged tail of the
        # outer training partition, never the outer validation observations.
        for factory in candidate_factories:
            candidate = factory()
            candidate_name = getattr(candidate, "name", candidate.__class__.__name__)
            if str(candidate_name) in candidates:
                raise ValueError(f"Duplicate experiment candidate name: {candidate_name}")
            inner_training, inner_validation = purged_tail_split(
                len(raw_train), validation_fraction=0.15, purge=selected.effective_gap
            )
            inner_raw_train = raw_train[inner_training]
            inner_raw_validation = raw_train[inner_validation]
            feature_count = inner_raw_train.shape[2]
            scaler = MinMaxScaler().fit(inner_raw_train.reshape(-1, feature_count))
            inner_scaled_train = scaler.transform(
                inner_raw_train.reshape(-1, feature_count)
            ).reshape(inner_raw_train.shape)
            inner_scaled_validation = scaler.transform(
                inner_raw_validation.reshape(-1, feature_count)
            ).reshape(inner_raw_validation.shape)
            candidate.fit(
                inner_scaled_train,
                training_targets[inner_training],
                validation_data=(inner_scaled_validation, training_targets[inner_validation]),
            )
            if hasattr(candidate, "refit"):
                candidate.refit(scaled_train, training_targets)
                candidate_validation = scaled_validation
            else:
                # Lightweight third-party adapters may not support a second
                # fit. They remain on the purged inner training scaler.
                candidate_validation = scaler.transform(
                    raw_validation.reshape(-1, feature_count)
                ).reshape(raw_validation.shape)
            predicted_targets = candidate.predict(candidate_validation)
            candidates[str(candidate_name)] = reconstruct_prices(
                validation_origins, predicted_targets, selected.target_type
            )

        for model_name, predicted_prices in candidates.items():
            fold_report = evaluate_forecast_horizons(
                validation_actual,
                predicted_prices,
                validation_origins,
                horizons=selected.horizons,
                scale_series=fold_scale_series,
            )
            report = model_reports.setdefault(model_name, {"folds": []})
            report["folds"].append(
                {
                    "fold": fold_number,
                    "train_samples": int(len(training)),
                    "validation_samples": int(len(validation)),
                    "train_index_start": int(training[0]),
                    "train_index_end": int(training[-1]),
                    "validation_index_start": int(validation[0]),
                    "validation_index_end": int(validation[-1]),
                    **fold_report,
                }
            )
            rows = pooled_rows.setdefault(
                model_name,
                {"actual": [], "predicted": [], "origins": []},
            )
            rows["actual"].append(validation_actual)
            rows["predicted"].append(predicted_prices)
            rows["origins"].append(validation_origins)

    promotion_policy = PromotionPolicy(
        minimum_winning_folds=min(4, selected.folds),
    )
    persistence_rows = pooled_rows["persistence"]
    persistence_predicted = np.concatenate(persistence_rows["predicted"])
    for model_name, report in model_reports.items():
        rows = pooled_rows[model_name]
        aggregate = evaluate_forecast_horizons(
            np.concatenate(rows["actual"]),
            np.concatenate(rows["predicted"]),
            np.concatenate(rows["origins"]),
            horizons=selected.horizons,
            scale_series=pooled_scale_series,
        )
        decision = assess_promotion(
            aggregate["pooled"],
            [fold["pooled"] for fold in report["folds"]],
            policy=promotion_policy,
        )
        report["aggregate"] = aggregate
        report["promotion"] = {
            "promoted": decision.promoted,
            "reasons": list(decision.reasons),
        }
        if model_name != "persistence":
            # Pooled observations are ordered chronologically by fold and origin.
            # The bootstrap supports, but does not determine, the promotion gate.
            report["evidence"] = paired_loss_evidence(
                np.concatenate(rows["actual"]),
                np.concatenate(rows["predicted"]),
                persistence_predicted,
                loss="absolute",
                horizon=max(selected.horizons),
                resamples=250,
                seed=selected.seed,
            )

    result = {
        "config": {
            **asdict(selected),
            "effective_gap": selected.effective_gap,
        },
        "dataset": {
            "samples": int(len(dataset.features)),
            "feature_count": int(dataset.features.shape[2]),
            "first_origin_index": int(dataset.origin_indices[0]),
            "last_origin_index": int(dataset.origin_indices[-1]),
            "snapshot_id": dataset.snapshot_id,
        },
        "models": model_reports,
    }
    return _round_metric_tree(result)
