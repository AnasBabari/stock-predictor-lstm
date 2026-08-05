import hashlib
import logging
import platform
from datetime import datetime

import numpy as np
from sklearn.preprocessing import RobustScaler

from config import FEATURES_V4, MAX_FORECAST_DAYS, WINDOW_SIZE
from data_pipeline import fetch_browser_data
from experiments.baselines import ElasticNetForecaster
from experiments.runner import ExperimentConfig, run_baseline_experiment
from experiments.targets import reconstruct_prices
from server_models.contracts import (
    ReproducibilityMetadata,
    RobustScalerParams,
    ServerArtifactKey,
    ServerForecastBundle,
    ServerModelRecord,
    git_commit_short,
)

logger = logging.getLogger(__name__)


def train_server_forecast(ticker: str, registry, storage, signer) -> ServerModelRecord | None:
    """Train, evaluate, and promote a server forecast for a single ticker."""
    logger.info(f"Starting server training job for {ticker}")

    # 1. Fetch raw data matching the exact feature pipeline
    feature_df, closing_prices, dates, metadata = fetch_browser_data(ticker)
    feature_values = feature_df[FEATURES_V4].to_numpy(dtype=np.float64)
    close_values = np.asarray(closing_prices, dtype=np.float64)

    # 2. Run standard walk-forward evaluation on baselines (ridge, elastic_net)
    config = ExperimentConfig(
        lookback=WINDOW_SIZE,
        horizons=tuple(range(1, MAX_FORECAST_DAYS + 1)),
        target_type="log_return",
        include_hgb=False,
    )
    result = run_baseline_experiment(
        feature_values, close_values, feature_names=list(FEATURES_V4), config=config, dates=dates
    )

    # 3. Assess promotion
    best_candidate = None
    best_rmse = float("inf")

    for candidate_name in ("elastic_net",):
        report = result["models"][candidate_name]
        rmse = report["aggregate"]["pooled"]["relative_rmse"]
        mae = report["aggregate"]["pooled"]["relative_mae"]
        if rmse < 0.98 and mae < 0.98 and rmse < best_rmse:
            best_rmse = rmse
            best_candidate = candidate_name

    if not best_candidate:
        logger.warning(f"No candidate passed promotion gates for {ticker}.")
        return None

    logger.info(f"Candidate {best_candidate} passed with RMSE {best_rmse:.4f}")

    # 4. Train best candidate on the entire dataset
    # We must construct the windowed dataset just like walk-forward runner does
    from experiments.contracts import build_experiment_dataset

    dataset = build_experiment_dataset(
        feature_values,
        close_values,
        dates=dates,
        feature_names=list(FEATURES_V4),
        lookback=WINDOW_SIZE,
        horizons=config.horizons,
        target_type="log_return",
    )

    # In runner.py, ElasticNet/Ridge expect _flatten_features
    # We fit a scaler on the entire flattened training data
    feature_count = dataset.features.shape[2]
    flat_features = dataset.features.reshape(-1, feature_count)

    # Use RobustScaler to match the browser's preprocessing style (and metadata contract)
    scaler = RobustScaler()
    scaler.fit(flat_features)
    scaled_features = scaler.transform(flat_features).reshape(dataset.features.shape)

    from typing import Any

    model: Any = ElasticNetForecaster()

    model.fit(scaled_features, dataset.targets)

    # 5. Generate prediction bundle for the most recent observation
    # We use the very last row of dataset.features to predict future horizons
    last_window = dataset.features[-1:]
    last_scaled = scaler.transform(last_window.reshape(-1, feature_count)).reshape(
        last_window.shape
    )

    predicted_targets = model.predict(last_scaled)
    # The last row of dataset corresponds to dataset.origins[-1] (the final close price)
    origin_close = dataset.origins[-1]
    origin_date_val = dataset.origin_dates[-1]
    predicted_prices = reconstruct_prices([origin_close], predicted_targets, "log_return")[0]

    predicted_log_returns = predicted_targets[0]

    from calendars import future_trading_dates

    future_date_strs, _ = future_trading_dates(
        ticker, str(origin_date_val.date()), MAX_FORECAST_DAYS
    )
    future_dates = [datetime.strptime(d, "%Y-%m-%d").date() for d in future_date_strs]

    snapshot_id = metadata.get("snapshot_id", "unknown")
    key = ServerArtifactKey.create(ticker=ticker, snapshot_id=snapshot_id)

    bundle = ServerForecastBundle(
        ticker=ticker,
        version_id=key.version_id,
        origin_close=float(origin_close),
        origin_date=origin_date_val.date(),
        future_dates=future_dates,
        predicted_log_returns=predicted_log_returns.tolist(),
        predicted_prices=predicted_prices.tolist(),
        generated_at=key.trained_at,
    )

    # 6. Prepare ReproducibilityMetadata
    repro = ReproducibilityMetadata(
        feature_names=list(FEATURES_V4),
        scaler=RobustScalerParams(medians=scaler.center_.tolist(), iqrs=scaler.scale_.tolist()),
        metrics=result["models"][best_candidate]["aggregate"]["per_horizon"],
        python_version=platform.python_version(),
        git_commit=git_commit_short(),
    )

    # Serialize bundle and compute digest
    bundle_bytes = bundle.model_dump_json().encode("utf-8")
    digest = hashlib.sha256(bundle_bytes).hexdigest()

    signature = signer.sign(bundle_bytes)

    record = ServerModelRecord(
        key=key, reproducibility=repro, sha256_digest=digest, signature=signature, status="promoted"
    )

    # 7. Atomically save bundle to S3 and record to Postgres
    storage.put_bundle(key.version_id, bundle_bytes)
    registry.promote_model(record)

    logger.info(f"Promoted {best_candidate} for {ticker} (version {key.version_id})")
    return record
