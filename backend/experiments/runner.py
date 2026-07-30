"""Shared walk-forward runner for offline baseline and candidate experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from evaluation.metrics import evaluate_forecast_horizons
from evaluation.promotion import PromotionPolicy, assess_promotion
from evaluation.splits import generate_walk_forward_splits
from experiments.baselines import (
    DriftForecaster,
    HistogramGradientBoostingForecaster,
    PersistenceForecaster,
    RidgeForecaster,
)
from experiments.targets import TargetType, build_supervised_dataset, reconstruct_prices


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
) -> dict:
    """Compare baselines on identical purged walk-forward observations."""

    selected = config or ExperimentConfig()
    if "Close" not in feature_names:
        raise ValueError("feature_names must identify the Close feature.")
    dataset = build_supervised_dataset(
        feature_values,
        close_values,
        lookback=selected.lookback,
        horizons=selected.horizons,
        target_type=selected.target_type,
    )
    if len(feature_names) != dataset.features.shape[2]:
        raise ValueError("feature_names must match the feature matrix.")

    splits = generate_walk_forward_splits(
        len(dataset.features),
        folds=selected.folds,
        min_train_size=selected.min_train_size,
        validation_size=selected.validation_size,
        gap=selected.effective_gap,
        method=selected.method,
    )
    model_reports: dict[str, dict] = {}
    pooled_rows: dict[str, dict[str, list[np.ndarray]]] = {}
    initial_scale_end = int(dataset.origin_indices[splits[0][0][-1]] + max(selected.horizons))
    pooled_scale_series = np.asarray(close_values, dtype=float)[: initial_scale_end + 1]

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
        },
        "models": model_reports,
    }
    return _round_metric_tree(result)
