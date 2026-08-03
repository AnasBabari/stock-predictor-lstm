"""Shared walk-forward runner for offline baseline and candidate experiments."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from evaluation.blending import fit_constrained_blend, fit_shrinkage_alpha
from evaluation.conformal import calibrate_intervals, interval_diagnostics, prediction_intervals
from evaluation.drift import feature_divergence, residual_drift
from evaluation.evidence import (
    benjamini_hochberg,
    paired_loss_evidence,
    relative_ratio_evidence,
)
from evaluation.metrics import evaluate_forecast_horizons, pinball_loss
from evaluation.promotion import PromotionPolicy, assess_promotion
from evaluation.seeds import aggregate_seed_runs
from evaluation.splits import purged_tail_split
from experiments.baselines import (
    DriftForecaster,
    ElasticNetForecaster,
    HistogramGradientBoostingForecaster,
    PersistenceForecaster,
    QuantileForecaster,
    RidgeForecaster,
    SmallTCNForecaster,
    quantile_crossing_rate,
)
from experiments.contracts import FoldPlan, build_experiment_dataset
from experiments.targets import TargetType, reconstruct_prices, transform_price_targets

# Deterministic models produce identical predictions for every seed, so they
# are evaluated once and shared across the seed loop.
DETERMINISTIC_MODELS = frozenset({"persistence", "drift", "ridge", "elastic_net"})


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
    seeds: tuple[int, ...] = (42,)
    # Opt-in diagnostics; all are additive report keys and never influence
    # fold plans, aggregate metrics, or promotion decisions.
    include_blends: bool = False
    include_quantiles: bool = False
    include_drift: bool = False
    include_tcn: bool = False

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


def _price_log_returns(prices, origins) -> np.ndarray:
    """Convert a (rows, horizons) price matrix to log returns versus origins."""

    origin_array = np.asarray(origins, dtype=float).reshape(-1)
    return np.log(np.asarray(prices, dtype=float) / origin_array[:, None])


def _round_metric_tree(value):
    if isinstance(value, dict):
        return {key: _round_metric_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_metric_tree(item) for item in value]
    if isinstance(value, float):
        return round(value, 8)
    return value


def _instantiate_candidate(factory: Callable, seed: int) -> object:
    """Build a candidate, passing the seed only to factories that accept one."""

    try:
        accepts_seed = bool(inspect.signature(factory).parameters)
    except (TypeError, ValueError):
        accepts_seed = False
    return factory(seed) if accepts_seed else factory()


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

    # Multi-seed runs re-fit only the stochastic models (hist_gradient_boosting
    # and the neural candidate factories) once per seed on the shared dataset
    # and fold plan. Single-seed runs keep the historical seed semantics: the
    # loop degenerates to (selected.seed,), which is also the HGB random_state
    # and the bootstrap evidence seed.
    multi_seed = len(selected.seeds) > 1
    seed_loop = tuple(selected.seeds) if multi_seed else (selected.seed,)
    # Per-seed pooled rows for stochastic models (multi-seed runs only). A
    # None entry marks a seed run that failed and counts toward failure_count.
    seed_pooled_rows: dict[str, dict[int, dict[str, list[np.ndarray]] | None]] = {}
    factory_names: list[str] = []
    # Per-fold pooled quantile predictions (include_quantiles only). Only the
    # first seed run appends, matching the pooled_rows first-seed convention.
    quantile_predictions_by_fold: list[np.ndarray] = []
    # Quantile levels of the fitted quantile forecaster, captured with the
    # first-seed predictions so diagnostics never hardcode column positions.
    quantile_tau_levels: tuple[float, ...] | None = None

    for fold_number, (training, validation) in enumerate(splits, start=1):
        raw_train = dataset.features[training]
        raw_validation = dataset.features[validation]
        scaled_train, scaled_validation = _scale_windows(raw_train, raw_validation)
        training_targets = dataset.targets[training]
        validation_origins = dataset.origins[validation]
        validation_actual = dataset.actual_prices[validation]
        fold_scale_end = int(dataset.origin_indices[training[-1]] + max(selected.horizons))
        fold_scale_series = np.asarray(close_values, dtype=float)[: fold_scale_end + 1]

        # Default arguments bind the fold-local values so the helpers never
        # observe a later iteration's state (and keep ruff B023 quiet).
        def record_fold_prediction(
            model_name: str,
            predicted_prices,
            *,
            fold_number: int = fold_number,
            training=training,
            validation=validation,
            validation_actual=validation_actual,
            validation_origins=validation_origins,
            fold_scale_series=fold_scale_series,
        ) -> None:
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

        def append_seed_rows(
            model_name: str,
            seed: int,
            predicted_prices,
            *,
            validation_actual=validation_actual,
            validation_origins=validation_origins,
        ) -> None:
            model_rows = seed_pooled_rows.setdefault(model_name, {})
            rows = model_rows.get(seed)
            if rows is None:
                if seed in model_rows:
                    # A None entry marks a seed run that already failed on an
                    # earlier fold; skip its late successes silently.
                    return
                rows = {"actual": [], "predicted": [], "origins": []}
                model_rows[seed] = rows
            rows["actual"].append(validation_actual)
            rows["predicted"].append(predicted_prices)
            rows["origins"].append(validation_origins)

        # Deterministic baselines are fitted once per fold and shared across
        # every seed in the loop.
        deterministic_candidates = {
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
            ElasticNetForecaster().fit(scaled_train, training_targets),
        ):
            predicted_targets = model.predict(scaled_validation)
            deterministic_candidates[model.name] = reconstruct_prices(
                validation_origins,
                predicted_targets,
                selected.target_type,
            )
        for model_name, predicted_prices in deterministic_candidates.items():
            record_fold_prediction(model_name, predicted_prices)

        for seed_index, seed in enumerate(seed_loop):
            first_seed_run = seed_index == 0
            stochastic_candidates: dict[str, np.ndarray] = {}
            failed_models: list[str] = []

            try:
                boosting = HistogramGradientBoostingForecaster(random_state=seed).fit(
                    scaled_train, training_targets
                )
                predicted_targets = boosting.predict(scaled_validation)
                stochastic_candidates[boosting.name] = reconstruct_prices(
                    validation_origins,
                    predicted_targets,
                    selected.target_type,
                )
            except Exception:
                if first_seed_run:
                    raise
                failed_models.append("hist_gradient_boosting")

            if selected.include_quantiles:
                try:
                    quantile_forecaster = QuantileForecaster(random_state=seed).fit(
                        scaled_train, training_targets
                    )
                    quantile_predictions = quantile_forecaster.predict(scaled_validation)
                    median_index = min(
                        range(len(quantile_forecaster.quantiles)),
                        key=lambda index: abs(quantile_forecaster.quantiles[index] - 0.5),
                    )
                    stochastic_candidates[quantile_forecaster.name] = reconstruct_prices(
                        validation_origins,
                        quantile_predictions[:, :, median_index],
                        selected.target_type,
                    )
                    if first_seed_run:
                        quantile_predictions_by_fold.append(quantile_predictions)
                        quantile_tau_levels = quantile_forecaster.quantiles
                except Exception:
                    if first_seed_run:
                        raise
                    failed_models.append(QuantileForecaster.name)

            if selected.include_tcn:
                # TCN training is stochastic, so the challenger lives in the
                # seed loop like HGB and stays out of DETERMINISTIC_MODELS.
                try:
                    tcn_forecaster = SmallTCNForecaster(seed=seed).fit(
                        scaled_train, training_targets
                    )
                    predicted_targets = tcn_forecaster.predict(scaled_validation)
                    stochastic_candidates[tcn_forecaster.name] = reconstruct_prices(
                        validation_origins,
                        predicted_targets,
                        selected.target_type,
                    )
                except Exception:
                    if first_seed_run:
                        raise
                    failed_models.append(SmallTCNForecaster.name)

            # Neural (or other) candidates use the exact outer folds as every
            # baseline. Their early-stopping validation is a purged tail of the
            # outer training partition, never the outer validation observations.
            for factory_index, factory in enumerate(candidate_factories):
                try:
                    candidate = _instantiate_candidate(factory, seed)
                    candidate_name = getattr(candidate, "name", candidate.__class__.__name__)
                    if (
                        str(candidate_name) in deterministic_candidates
                        or str(candidate_name) in stochastic_candidates
                    ):
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
                        validation_data=(
                            inner_scaled_validation,
                            training_targets[inner_validation],
                        ),
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
                    stochastic_candidates[str(candidate_name)] = reconstruct_prices(
                        validation_origins, predicted_targets, selected.target_type
                    )
                    if first_seed_run:
                        factory_names.append(str(candidate_name))
                except Exception:
                    if first_seed_run:
                        raise
                    if factory_index < len(factory_names):
                        failed_models.append(factory_names[factory_index])

            for model_name, predicted_prices in stochastic_candidates.items():
                # For multi-seed runs the reported folds/aggregate/promotion/
                # evidence come from the FIRST seed to preserve the historical
                # report schema; seed_summary is additive-only.
                if first_seed_run:
                    record_fold_prediction(model_name, predicted_prices)
                if multi_seed:
                    append_seed_rows(model_name, seed, predicted_prices)
            if multi_seed:
                for model_name in failed_models:
                    seed_pooled_rows.setdefault(model_name, {})[seed] = None

    promotion_policy = PromotionPolicy(
        minimum_winning_folds=min(4, selected.folds),
    )
    persistence_rows = pooled_rows["persistence"]
    persistence_predicted = np.concatenate(persistence_rows["predicted"])
    p_value_entries: list[dict] = []
    for model_name, report in model_reports.items():
        rows = pooled_rows[model_name]
        # Multi-seed runs: pooled_rows holds first-seed predictions only (see
        # record_fold_prediction), so every additive per-model key computed
        # here follows the same first-seed convention as seed_summary.
        pooled_actual = np.concatenate(rows["actual"])
        pooled_predicted = np.concatenate(rows["predicted"])
        aggregate = evaluate_forecast_horizons(
            pooled_actual,
            pooled_predicted,
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
                pooled_actual,
                pooled_predicted,
                persistence_predicted,
                loss="absolute",
                horizon=max(selected.horizons),
                resamples=250,
                seed=selected.seed,
            )
            horizon_evidence: dict[str, dict] = {}
            for column, horizon in enumerate(selected.horizons):
                horizon_actual = pooled_actual[:, column]
                horizon_predicted = pooled_predicted[:, column]
                horizon_baseline = persistence_predicted[:, column]
                entry = {
                    "absolute": paired_loss_evidence(
                        horizon_actual,
                        horizon_predicted,
                        horizon_baseline,
                        loss="absolute",
                        horizon=horizon,
                        resamples=250,
                        seed=selected.seed,
                    ),
                    "squared": paired_loss_evidence(
                        horizon_actual,
                        horizon_predicted,
                        horizon_baseline,
                        loss="squared",
                        horizon=horizon,
                        resamples=250,
                        seed=selected.seed,
                    ),
                    "relative_mae": relative_ratio_evidence(
                        horizon_actual,
                        horizon_predicted,
                        horizon_baseline,
                        metric="mae",
                        horizon=horizon,
                        resamples=250,
                        seed=selected.seed,
                    ),
                    "relative_rmse": relative_ratio_evidence(
                        horizon_actual,
                        horizon_predicted,
                        horizon_baseline,
                        metric="rmse",
                        horizon=horizon,
                        resamples=250,
                        seed=selected.seed,
                    ),
                }
                horizon_evidence[str(horizon)] = entry
                for loss_name in ("absolute", "squared"):
                    p_value_entries.append(
                        {
                            "model": model_name,
                            "horizon": int(horizon),
                            "loss": loss_name,
                            "p_value": entry[loss_name]["two_sided_p_value"],
                        }
                    )
            report["evidence_by_horizon"] = horizon_evidence
            # Split-conformal intervals calibrated on this model's own pooled
            # out-of-fold price residuals. Persistence is excluded: its
            # residuals are the reference baseline, not a calibrated model.
            calibration = calibrate_intervals(pooled_actual, pooled_predicted, coverages=(0.9,))
            conformal_bounds = prediction_intervals(pooled_predicted, calibration, coverage=0.9)
            diagnostics = interval_diagnostics(pooled_actual, conformal_bounds)
            report["intervals"] = {
                "confidence": 0.9,
                "radius": {
                    str(horizon): float(radius)
                    for horizon, radius in zip(
                        selected.horizons, calibration["radii"]["0.9"], strict=True
                    )
                },
                "empirical_coverage": diagnostics["empirical_coverage"],
                "average_width": diagnostics["average_width"],
                "sample_count": calibration["calibration_count"],
            }

    if multi_seed:
        # Additive cross-seed summary for stochastic models only. Each seed's
        # pooled relative metrics are computed on that seed's pooled rows with
        # the same evaluator used for the aggregate report above.
        for model_name, report in model_reports.items():
            if model_name in DETERMINISTIC_MODELS:
                continue
            per_seed_summaries: list[dict[str, float] | None] = []
            for seed in seed_loop:
                rows = seed_pooled_rows.get(model_name, {}).get(seed)
                if rows is None:
                    per_seed_summaries.append(None)
                    continue
                pooled_metrics = evaluate_forecast_horizons(
                    np.concatenate(rows["actual"]),
                    np.concatenate(rows["predicted"]),
                    np.concatenate(rows["origins"]),
                    horizons=selected.horizons,
                    scale_series=pooled_scale_series,
                )["pooled"]
                per_seed_summaries.append(
                    {
                        "relative_mae": pooled_metrics["relative_mae"],
                        "relative_rmse": pooled_metrics["relative_rmse"],
                    }
                )
            report["seed_summary"] = aggregate_seed_runs(per_seed_summaries)

    # FDR control across every per-horizon paired-loss p value collected
    # above; decisions are reported only and never feed promotion.
    evidence_multiple_comparison: dict | None = None
    if p_value_entries:
        rejected = benjamini_hochberg([entry["p_value"] for entry in p_value_entries], q=0.10)
        evidence_multiple_comparison = {
            "method": "benjamini_hochberg",
            "q": 0.10,
            "decisions": [
                {**entry, "rejected": bool(decision)}
                for entry, decision in zip(p_value_entries, rejected, strict=True)
            ],
        }

    if selected.include_quantiles and quantile_predictions_by_fold:
        # Pooled quantile forecasts come from the first-seed outer folds only,
        # matching the pooled_rows convention used by every per-model key.
        quantile_report = model_reports[QuantileForecaster.name]
        pooled_quantiles = np.concatenate(quantile_predictions_by_fold, axis=0)
        quantile_actual = np.concatenate(pooled_rows[QuantileForecaster.name]["actual"])
        quantile_origins = np.concatenate(pooled_rows[QuantileForecaster.name]["origins"])
        # Stored quantile predictions live in target space; diagnostics compare
        # prices, so reconstruct each quantile column into price units first.
        pooled_quantile_prices = np.stack(
            [
                reconstruct_prices(
                    quantile_origins, pooled_quantiles[:, :, index], selected.target_type
                )
                for index in range(pooled_quantiles.shape[2])
            ],
            axis=2,
        )
        tau_levels = quantile_tau_levels or (0.05, 0.5, 0.95)

        def tau_index(tau: float) -> int:
            # Match requested tau levels to actual quantile tuple positions.
            return int(min(range(len(tau_levels)), key=lambda index: abs(tau_levels[index] - tau)))

        pinball_by_tau = {
            str(tau): {
                str(horizon): pinball_loss(
                    quantile_actual[:, column],
                    pooled_quantile_prices[:, column, tau_index(tau)],
                    tau,
                )
                for column, horizon in enumerate(selected.horizons)
            }
            for tau in (0.05, 0.95)
        }
        band_lower = pooled_quantile_prices[:, :, tau_index(min(tau_levels))]
        band_upper = pooled_quantile_prices[:, :, tau_index(max(tau_levels))]
        quantile_report["quantile_diagnostics"] = {
            "pinball_loss": pinball_by_tau,
            "quantile_crossing_rate": quantile_crossing_rate(pooled_quantiles),
            "band_coverage": float(
                np.mean((quantile_actual >= band_lower) & (quantile_actual <= band_upper))
            ),
            "band_quantiles": [min(tau_levels), max(tau_levels)],
            "sample_count": int(len(quantile_actual)),
        }

    if selected.include_drift:
        # Input drift is model-independent: first-fold training windows versus
        # every fold's pooled validation windows, both raw dataset slices.
        input_divergence = feature_divergence(
            dataset.features[splits[0][0]],
            np.concatenate([dataset.features[validation] for _, validation in splits]),
        )
        for model_name, report in model_reports.items():
            rows = pooled_rows[model_name]
            pooled_actual = np.concatenate(rows["actual"])
            pooled_predicted = np.concatenate(rows["predicted"])
            pooled_magnitude = np.mean(np.abs(pooled_actual - pooled_predicted), axis=1)
            report["drift"] = {
                "feature_divergence": input_divergence,
                "residual_drift": residual_drift(
                    pooled_magnitude, resamples=250, seed=selected.seed
                ),
            }

    blend_report: dict | None = None
    if selected.include_blends:
        # Blends are diagnostic-only: weights and shrinkage factors are fitted
        # on earlier folds and never influence aggregate metrics or promotion.
        blend_member_names = [name for name in model_reports if name != "persistence"]
        shrinkage: dict[str, dict[str, float]] = {}
        constrained: dict | None = None
        if blend_member_names and len(splits) >= 2:
            for model_name in blend_member_names:
                rows = pooled_rows[model_name]
                # Shrinkage calibration uses every pooled fold except the
                # chronological last one, which stays held out.
                fit_predicted = _price_log_returns(
                    np.concatenate(rows["predicted"][:-1]),
                    np.concatenate(rows["origins"][:-1]),
                )
                fit_actual = _price_log_returns(
                    np.concatenate(rows["actual"][:-1]),
                    np.concatenate(rows["origins"][:-1]),
                )
                shrinkage[model_name] = {
                    str(horizon): fit_shrinkage_alpha(
                        fit_predicted[:, column], fit_actual[:, column]
                    )
                    for column, horizon in enumerate(selected.horizons)
                }
            member_fit_columns = []
            member_held_columns = []
            for model_name in blend_member_names:
                rows = pooled_rows[model_name]
                fit_targets = transform_price_targets(
                    np.concatenate(rows["origins"][:-1]),
                    np.concatenate(rows["predicted"][:-1]),
                    selected.target_type,
                )
                member_fit_columns.append(fit_targets.reshape(-1))
                member_held_columns.append(
                    transform_price_targets(
                        rows["origins"][-1], rows["predicted"][-1], selected.target_type
                    )
                )
            fit_actual_targets = transform_price_targets(
                np.concatenate(persistence_rows["origins"][:-1]),
                np.concatenate(persistence_rows["actual"][:-1]),
                selected.target_type,
            ).reshape(-1)
            blend_weights = fit_constrained_blend(
                np.column_stack(member_fit_columns), fit_actual_targets
            )
            held_blend = np.zeros_like(member_held_columns[0])
            for weight, member in zip(blend_weights, member_held_columns, strict=True):
                held_blend += weight * member
            held_origins = persistence_rows["origins"][-1]
            blend_prices = reconstruct_prices(held_origins, held_blend, selected.target_type)
            blend_metrics = evaluate_forecast_horizons(
                persistence_rows["actual"][-1],
                blend_prices,
                held_origins,
                horizons=selected.horizons,
                scale_series=pooled_scale_series,
            )
            constrained = {
                "weights": {
                    name: float(weight)
                    for name, weight in zip(blend_member_names, blend_weights, strict=True)
                },
                "held_out_fold": len(splits),
                "relative_mae": {
                    str(horizon): blend_metrics["per_horizon"][str(horizon)]["relative_mae"]
                    for horizon in selected.horizons
                },
                "relative_rmse": {
                    str(horizon): blend_metrics["per_horizon"][str(horizon)]["relative_rmse"]
                    for horizon in selected.horizons
                },
            }
        blend_report = {"shrinkage": shrinkage, "constrained": constrained}

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
    if evidence_multiple_comparison is not None:
        result["evidence_multiple_comparison"] = evidence_multiple_comparison
    if blend_report is not None:
        result["blend"] = blend_report
    return _round_metric_tree(result)
