"""Run the V11.2 numeric-only development stage on the RTX host.

The command consumes only ``development/`` files produced by
``prepare_v11_2_dataset.py``.  It cannot accept a holdout key or a sealed
payload path, and it writes a pre-unseal comparison plus one route per horizon.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "research"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from backend.panel.features import DEPLOYABLE_FEATURE_COLUMNS_V5  # noqa: E402
from research.volatility_forecasting.global_multimodal_trainer_v11 import (  # noqa: E402
    EconometricHARBaseline,
)
from research.volatility_forecasting.model import RobustSequenceScaler  # noqa: E402
from research.volatility_forecasting.v11_2_evidence import (  # noqa: E402
    seed_evidence_from_forecast,
    write_development_report,
    write_seed_evidence,
)
from research.volatility_forecasting.v11_2_freezer import (  # noqa: E402
    V112Route,
    freeze_routing_bundle,
)
from research.volatility_forecasting.v11_2_protocol import (  # noqa: E402
    V11_2_EMBARGO_SESSIONS,
    V11_2_MAX_HORIZON,
    V112Protocol,
    require_numeric_only,
)
from research.volatility_forecasting.v11_2_sealed_store import load_v112_development  # noqa: E402
from research.volatility_forecasting.v11_2_trainer import (  # noqa: E402
    evaluate_residual_model,
    fit_histgb_location,
    fit_ridge_location,
    make_forecast,
    select_per_horizon_challengers,
    train_epoch_zero_residual_model,
)

# V11.2 consumes the deployable v5 stationary feature order.  The legacy
# V11.1 HAR helper expects its own 34-column schema, so never pass v5 columns
# to its hard-coded positions (23, 24, 25): those are liquidity features here.
# Use the named v5 volatility columns explicitly instead.
_V112_FEATURE_COUNT = len(DEPLOYABLE_FEATURE_COLUMNS_V5)
_V112_HAR_INDICES = (13, 15, 16)  # Vol_C2C_5, Vol_C2C_20, Vol_C2C_60
_V112_DAILY_RETURN_INDEX = 0  # Return_1D


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git_provenance() -> tuple[str, bool]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        return sha, dirty
    except (OSError, subprocess.SubprocessError):
        return "unknown_git_sha", True


def _last_row(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim == 3:
        return values[:, -1, :]
    if values.ndim == 2:
        return values
    raise ValueError("features must have shape [rows, features] or [rows, window, features]")


def _fit_har(
    train_features: np.ndarray, train_rv: np.ndarray, eval_features: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    train_last = _last_row(train_features)
    eval_last = _last_row(eval_features)
    if train_last.shape[1] != _V112_FEATURE_COUNT:
        raise ValueError("V11.2 numeric features must contain exactly the 26 deployable v5 columns")
    har_features_train = train_last[:, _V112_HAR_INDICES]
    har_features_eval = eval_last[:, _V112_HAR_INDICES]
    har = EconometricHARBaseline()
    # The V11.1 class consumes three variance predictors at positions 23–25;
    # adapt its stable regression implementation to the correctly named v5
    # volatility predictors rather than changing the V11.1 contract.
    train_projection = np.zeros((len(har_features_train), _V112_FEATURE_COUNT), dtype=np.float64)
    eval_projection = np.zeros((len(har_features_eval), _V112_FEATURE_COUNT), dtype=np.float64)
    train_projection[:, 23:26] = har_features_train
    eval_projection[:, 23:26] = har_features_eval
    har.fit(train_projection, train_rv)
    return har.predict_variance(train_projection), har.predict_variance(eval_projection)


def _persistence_variance(features: np.ndarray, horizons: tuple[int, ...]) -> np.ndarray:
    last = _last_row(features)
    if last.shape[1] != _V112_FEATURE_COUNT:
        raise ValueError("V11.2 numeric features must contain exactly the 26 deployable v5 columns")
    latest_daily_variance = np.maximum(last[:, _V112_DAILY_RETURN_INDEX] ** 2, 1e-8)
    return np.column_stack([latest_daily_variance * horizon for horizon in horizons])


def _inner_development_indices(dates: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Split training rows by sessions with purge and embargo for ranking only."""
    sessions = sorted(set(dates))
    cut = int(len(sessions) * 0.80)
    train_end = cut - V11_2_MAX_HORIZON
    validation_start = cut + V11_2_EMBARGO_SESSIONS
    if train_end < 100 or validation_start >= len(sessions):
        raise ValueError("V11.2 training partition is too small for inner selection")
    train_sessions = set(sessions[:train_end])
    validation_sessions = set(sessions[validation_start:])
    train_idx = np.asarray(
        [i for i, date in enumerate(dates) if date in train_sessions], dtype=np.int64
    )
    validation_idx = np.asarray(
        [i for i, date in enumerate(dates) if date in validation_sessions], dtype=np.int64
    )
    return train_idx, validation_idx


def _training_only_ranking_scores(
    *,
    train_features: np.ndarray,
    train_returns: np.ndarray,
    train_rv: np.ndarray,
    train_dates: tuple[str, ...],
    horizon_column: int,
    horizon: int,
    max_epochs: int,
    patience: int,
    device: str | None,
) -> dict[str, float]:
    """Rank fixed candidate families using only an inner chronological split."""
    inner_train, inner_validation = _inner_development_indices(train_dates)
    inner_scaler = RobustSequenceScaler.fit(train_features[inner_train])
    inner_train_x = inner_scaler.transform(train_features[inner_train])
    inner_validation_x = inner_scaler.transform(train_features[inner_validation])
    inner_har_train, inner_har_validation = _fit_har(
        train_features[inner_train], train_rv[inner_train], train_features[inner_validation]
    )
    y_train = train_returns[inner_train, horizon_column]
    y_validation = train_returns[inner_validation, horizon_column]
    rv_validation = train_rv[inner_validation, horizon_column]
    scores: dict[str, float] = {}
    ridge = fit_ridge_location(
        inner_train_x,
        y_train,
        inner_validation_x,
        inner_har_validation[:, horizon_column],
        y_validation,
        rv_validation,
        horizon=horizon,
    )
    histgb = fit_histgb_location(
        inner_train_x,
        y_train,
        inner_validation_x,
        inner_har_validation[:, horizon_column],
        y_validation,
        rv_validation,
        horizon=horizon,
    )
    scores[ridge.family] = float(np.mean(ridge.crps))
    scores[histgb.family] = float(np.mean(histgb.crps))
    neural_training = train_epoch_zero_residual_model(
        x_train=inner_train_x,
        base_variance_train=inner_har_train[:, horizon_column],
        returns_train=y_train,
        rv_train=train_rv[inner_train, horizon_column],
        x_validation=inner_validation_x,
        base_variance_validation=inner_har_validation[:, horizon_column],
        returns_validation=y_validation,
        rv_validation=rv_validation,
        max_epochs=max_epochs,
        patience=patience,
        seed=42,
        device=device,
    )
    neural = evaluate_residual_model(
        neural_training,
        x_eval=inner_validation_x,
        base_variance_eval=inner_har_validation[:, horizon_column],
        returns_eval=y_validation,
        rv_eval=rv_validation,
        horizon=horizon,
    )
    scores[neural.family] = float(np.mean(neural.crps))
    return scores


def _save_pickle(value: Any, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = pickle.dumps(value, protocol=5)
    path.write_bytes(data)
    return _digest_bytes(data)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument(
        "--device", default=None, help="cuda, cpu, or omitted for automatic selection"
    )
    args = parser.parse_args()

    protocol = V112Protocol()
    require_numeric_only(None)
    development = load_v112_development(args.dataset_dir)
    if development.protocol_id != protocol.protocol_id:
        raise SystemExit("development dataset protocol does not match V11.2")
    if len(development.train_dates) == 0 or len(development.validation_dates) == 0:
        raise SystemExit("development dataset has an empty train or validation partition")
    if development.train_returns.ndim != 2 or development.train_returns.shape[1] != len(
        protocol.horizons
    ):
        raise SystemExit("V11.2 targets must contain exactly four horizons")

    output = args.output_dir
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit("V11.2 development output must be absent or an empty directory")
    evidence_dir = output / "seed_evidence"
    model_dir = output / "models"
    output.mkdir(parents=True, exist_ok=True)
    train_scaler = RobustSequenceScaler.fit(development.train_features)
    train_x = train_scaler.transform(development.train_features)
    validation_x = train_scaler.transform(development.validation_features)
    scaler_payload = train_scaler.to_dict()
    scaler_bytes = json.dumps(scaler_payload, indent=2, sort_keys=True).encode("utf-8")
    scaler_digest = _digest_bytes(scaler_bytes)
    (output / "numeric_scaler.json").write_bytes(scaler_bytes)

    train_har, validation_har = _fit_har(
        development.train_features, development.train_rv, development.validation_features
    )
    validation_persistence = _persistence_variance(
        development.validation_features, protocol.horizons
    )
    constant_variance = np.mean(development.train_rv, axis=0)
    validation_constant = np.tile(constant_variance, (len(development.validation_dates), 1))

    all_comparisons: dict[str, Any] = {
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.digest(),
        "panel_sha256": development.panel_sha256,
        "split_sha256": development.split_sha256,
        "sealed_test_status": "LOCKED_UNOPENED",
        "news_mode": protocol.news_mode,
        "train_stock_origin_observations": len(development.train_dates),
        "validation_stock_origin_observations": len(development.validation_dates),
        "train_unique_sessions": len(set(development.train_dates)),
        "validation_unique_sessions": len(set(development.validation_dates)),
        "horizons": {},
    }
    routes: list[V112Route] = []
    seed_evidence_digests: list[str] = []
    har_forecasts_by_horizon: dict[int, Any] = {}
    constant_forecasts_by_horizon: dict[int, Any] = {}
    persistence_forecasts_by_horizon: dict[int, Any] = {}
    candidates_by_horizon: dict[int, dict[str, Any]] = {}
    seed_results_by_horizon: dict[int, dict[int, Any]] = {}
    ranking_scores_by_horizon: dict[int, dict[str, float]] = {}
    inner_split_by_horizon: dict[int, dict[str, Any]] = {}

    for column, horizon in enumerate(protocol.horizons):
        train_returns = development.train_returns[:, column]
        validation_returns = development.validation_returns[:, column]
        train_rv = development.train_rv[:, column]
        validation_rv = development.validation_rv[:, column]
        har_forecast = make_forecast(
            "M0_HAR_BASELINE",
            horizon,
            np.zeros(len(validation_returns)),
            validation_har[:, column],
            validation_returns,
            validation_rv,
        )
        candidates = {
            "RIDGE_LOCATION_HAR_SCALE": fit_ridge_location(
                train_x,
                train_returns,
                validation_x,
                validation_har[:, column],
                validation_returns,
                validation_rv,
                horizon=horizon,
            ),
            "HISTGB_LOCATION_HAR_SCALE": fit_histgb_location(
                train_x,
                train_returns,
                validation_x,
                validation_har[:, column],
                validation_returns,
                validation_rv,
                horizon=horizon,
            ),
        }
        seed_results: dict[int, Any] = {}
        for seed in protocol.seeds:
            training = train_epoch_zero_residual_model(
                x_train=train_x,
                base_variance_train=train_har[:, column],
                returns_train=train_returns,
                rv_train=train_rv,
                x_validation=validation_x,
                base_variance_validation=validation_har[:, column],
                returns_validation=validation_returns,
                rv_validation=validation_rv,
                max_epochs=args.max_epochs,
                patience=args.patience,
                seed=seed,
                device=args.device,
            )
            forecast = evaluate_residual_model(
                training,
                x_eval=validation_x,
                base_variance_eval=validation_har[:, column],
                returns_eval=validation_returns,
                rv_eval=validation_rv,
                horizon=horizon,
            )
            seed_results[seed] = (training, forecast)
            seed_path = evidence_dir / f"horizon_{horizon}" / f"seed_{seed}.json"
            seed_evidence_digests.append(
                write_seed_evidence(
                    seed_evidence_from_forecast(
                        seed=seed,
                        horizon=horizon,
                        forecast=forecast,
                        dates=development.validation_dates,
                        training=training,
                    ),
                    seed_path,
                )
            )
            if seed == protocol.canonical_seed:
                candidates["M1_NUMERIC_RESIDUAL"] = forecast
                state_path = model_dir / f"horizon_{horizon}" / "seed_42.pt"
                state_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(training.model.state_dict(), state_path)

        # Simple baselines are recorded but can never be promoted as learned.
        constant_forecast = make_forecast(
            "ZERO_RETURN_CONST_VAR",
            horizon,
            np.zeros(len(validation_returns)),
            validation_constant[:, column],
            validation_returns,
            validation_rv,
        )
        persistence_forecast = make_forecast(
            "ZERO_RETURN_PERSISTENCE_VOL",
            horizon,
            np.zeros(len(validation_returns)),
            validation_persistence[:, column],
            validation_returns,
            validation_rv,
        )
        ranking_scores = _training_only_ranking_scores(
            train_features=development.train_features,
            train_returns=development.train_returns,
            train_rv=development.train_rv,
            train_dates=development.train_dates,
            horizon_column=column,
            horizon=horizon,
            max_epochs=args.max_epochs,
            patience=args.patience,
            device=args.device,
        )
        inner_train_indices, inner_validation_indices = _inner_development_indices(
            development.train_dates
        )
        inner_train_sessions = sorted(
            {development.train_dates[index] for index in inner_train_indices}
        )
        inner_validation_sessions = sorted(
            {development.train_dates[index] for index in inner_validation_indices}
        )
        har_forecasts_by_horizon[horizon] = har_forecast
        constant_forecasts_by_horizon[horizon] = constant_forecast
        persistence_forecasts_by_horizon[horizon] = persistence_forecast
        candidates_by_horizon[horizon] = candidates
        seed_results_by_horizon[horizon] = seed_results
        ranking_scores_by_horizon[horizon] = ranking_scores
        inner_split_by_horizon[horizon] = {
            "train_sessions": [inner_train_sessions[0], inner_train_sessions[-1]],
            "validation_sessions": [
                inner_validation_sessions[0],
                inner_validation_sessions[-1],
            ],
            "train_session_count": len(inner_train_sessions),
            "validation_session_count": len(inner_validation_sessions),
            "purge_sessions": V11_2_MAX_HORIZON,
            "embargo_sessions": V11_2_EMBARGO_SESSIONS,
        }

    # Candidate ranking is horizon-specific, but the inferential gate is
    # family-wise across all four horizons.  Do this once before any route or
    # selection record is frozen so no route can accidentally use an
    # uncorrected one-horizon p-value.
    selections = select_per_horizon_challengers(
        dates=list(development.validation_dates),
        horizons=protocol.horizons,
        har_by_horizon=har_forecasts_by_horizon,
        candidates_by_horizon=candidates_by_horizon,
        ranking_scores_by_horizon=ranking_scores_by_horizon,
        block_sessions=protocol.bootstrap_block_sessions,
        n_replicates=protocol.bootstrap_replicates,
        seed=protocol.bootstrap_seed,
    )

    for _column, horizon in enumerate(protocol.horizons):
        har_forecast = har_forecasts_by_horizon[horizon]
        constant_forecast = constant_forecasts_by_horizon[horizon]
        persistence_forecast = persistence_forecasts_by_horizon[horizon]
        candidates = candidates_by_horizon[horizon]
        seed_results = seed_results_by_horizon[horizon]
        ranking_scores = ranking_scores_by_horizon[horizon]
        selection = selections[horizon]
        m1_better_seeds = sum(
            float(np.mean(result[1].crps)) < float(np.mean(har_forecast.crps))
            for result in seed_results.values()
        )
        if (
            selection.learned_promotion
            and selection.selected_family == "M1_NUMERIC_RESIDUAL"
            and m1_better_seeds < 2
        ):
            selection = selection.__class__(
                horizon,
                "M0_HAR_BASELINE",
                False,
                selection.gates,
                "M1 failed the two-of-three seed stability rule; HAR retained",
            )
        selection_record = {
            "horizon": horizon,
            "selected_family": selection.selected_family,
            "learned_promotion": selection.learned_promotion,
            "reason": selection.reason,
            "gates": [gate.to_dict() for gate in selection.gates],
            "m1_better_seed_count": m1_better_seeds,
            "training_only_ranking_scores": ranking_scores,
            "inner_ranking_split": inner_split_by_horizon[horizon],
        }
        selection_path = output / f"selection_horizon_{horizon}.json"
        selection_bytes = json.dumps(selection_record, indent=2, sort_keys=True).encode("utf-8")
        selection_path.write_bytes(selection_bytes)
        selection_digest = _digest_bytes(selection_bytes)
        artifact_path = ""
        if selection.selected_family == "M1_NUMERIC_RESIDUAL":
            model_digest = _digest_bytes(
                (model_dir / f"horizon_{horizon}" / "seed_42.pt").read_bytes()
            )
            artifact_path = str(Path("models") / f"horizon_{horizon}" / "seed_42.pt")
        elif selection.selected_family in {"RIDGE_LOCATION_HAR_SCALE", "HISTGB_LOCATION_HAR_SCALE"}:
            if selection.selected_family == "RIDGE_LOCATION_HAR_SCALE":
                selected_model = Ridge(alpha=1.0)
                model_filename = f"horizon_{horizon}_ridge.pkl"
            else:
                selected_model = HistGradientBoostingRegressor(max_iter=150, random_state=42)
                model_filename = f"horizon_{horizon}_histgb.pkl"
            selected_model.fit(train_x.reshape(len(train_x), -1), train_returns)
            model_path = model_dir / model_filename
            model_digest = _save_pickle(
                {"model": selected_model, "scaler": scaler_payload, "horizon": horizon},
                model_path,
            )
            artifact_path = str(Path("models") / model_filename)
        else:
            baseline_path = output / "baselines" / f"har_horizon_{horizon}.json"
            baseline_payload = {"family": "M0_HAR_BASELINE", "horizon": horizon}
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            baseline_bytes = json.dumps(baseline_payload, indent=2).encode("utf-8")
            baseline_path.write_bytes(baseline_bytes)
            model_digest = _digest_bytes(baseline_bytes)
            artifact_path = str(Path("baselines") / baseline_path.name)
        routes.append(
            V112Route(
                horizon=horizon,
                family=selection.selected_family,
                model_digest=model_digest,
                scaler_digest=scaler_digest,
                selection_record_digest=selection_digest,
                learned_promotion=selection.learned_promotion,
                artifact_path=artifact_path,
            )
        )
        all_comparisons["horizons"][str(horizon)] = {
            "selected_family": selection.selected_family,
            "learned_promotion": selection.learned_promotion,
            "reason": selection.reason,
            "validation_crps": {
                "HAR": float(np.mean(har_forecast.crps)),
                "constant": float(np.mean(constant_forecast.crps)),
                "persistence": float(np.mean(persistence_forecast.crps)),
                **{family: float(np.mean(value.crps)) for family, value in candidates.items()},
            },
            "validation_qlike": {
                "HAR": float(np.mean(har_forecast.qlike)),
                **{family: float(np.mean(value.qlike)) for family, value in candidates.items()},
            },
            "validation_coverage_80": {
                "HAR": har_forecast.coverage_80,
                **{family: value.coverage_80 for family, value in candidates.items()},
            },
            "m1_better_seed_count": m1_better_seeds,
            "selection_record_sha256": selection_digest,
        }

    comparison_path = output / "v11_2_development_model_comparison.json"
    comparison_sha = write_development_report(all_comparisons, comparison_path)
    sealed_meta = json.loads(
        (args.dataset_dir / "sealed" / "sealed_metadata.json").read_text(encoding="utf-8")
    )
    git_sha, dirty = _git_provenance()
    freeze_routing_bundle(
        protocol=protocol,
        universe_sha256=str(
            json.loads(
                (args.dataset_dir / "manifests" / "universe.json").read_text(encoding="utf-8")
            )["manifest_sha256"]
        ),
        panel_sha256=development.panel_sha256,
        schema_sha256=str(sealed_meta["schema_sha256"]),
        split_sha256=development.split_sha256,
        development_evidence_sha256=comparison_sha,
        routes=routes,
        seed_evidence_sha256=seed_evidence_digests,
        sealed_ciphertext_sha256=str(sealed_meta["ciphertext_sha256"]),
        output_dir=output,
        git_sha=git_sha,
        git_dirty=dirty,
    )
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "comparison_sha256": comparison_sha,
                "sealed_test_status": "LOCKED_UNOPENED",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
