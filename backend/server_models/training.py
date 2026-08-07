import hashlib
import logging
import platform
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from config import FEATURES_V4, MAX_FORECAST_DAYS, WINDOW_SIZE
from data_pipeline import fetch_browser_data
from experiments.baselines import ElasticNetForecaster
from experiments.runner import ExperimentConfig, run_baseline_experiment
from experiments.targets import reconstruct_prices
from server_models.contracts import (
    HISTORY_DISPLAY_WINDOW,
    ReproducibilityMetadata,
    RobustScalerParams,
    ServerArtifactKey,
    ServerForecastBundle,
    ServerModelRecord,
    git_commit_short,
)
from server_models.promotion import assess_server_promotion

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

    # 3. Assess promotion with the harmonized gate set (report-level first so
    # hopeless candidates never pay for a full model fit).
    best_candidate = None
    for candidate_name in ("elastic_net",):
        report = result["models"][candidate_name]
        passed, reasons = assess_server_promotion(report, selected_horizon=MAX_FORECAST_DAYS)
        if passed:
            best_candidate = candidate_name
            break

    if not best_candidate:
        logger.warning(f"No candidate passed promotion gates for {ticker}.")
        return None

    logger.info(f"Candidate {best_candidate} passed with report gates for {ticker}.")

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

    # 5. Generate prediction bundle from the *most recent* observation window.
    # The inference slice is the raw tail of the feature matrix transformed with
    # the fitted scaler — never a stale row of the walk-forward dataset — so the
    # origin is exactly the last close price and the first future trading date.
    if len(feature_values) < WINDOW_SIZE:
        raise ValueError(f"Snapshot for {ticker} has fewer than {WINDOW_SIZE} rows.")
    latest_window = feature_values[-WINDOW_SIZE:]
    latest_scaled = scaler.transform(latest_window.reshape(-1, feature_count)).reshape(
        1, WINDOW_SIZE, feature_count
    )

    predicted_targets = model.predict(latest_scaled)
    origin_close = float(close_values[-1])
    origin_date_val = pd.Timestamp(dates[-1])
    predicted_prices = reconstruct_prices([origin_close], predicted_targets, "log_return")[0]

    # Re-run the full gate set now that a real forecast exists, including the
    # volatility plausibility range learned from observed horizon returns.
    predicted_cumulative_return = float(predicted_targets[0, MAX_FORECAST_DAYS - 1])
    passed, reasons = assess_server_promotion(
        result["models"][best_candidate],
        selected_horizon=MAX_FORECAST_DAYS,
        close_values=close_values,
        predicted_cumulative_return=predicted_cumulative_return,
    )
    if not passed:
        logger.warning(f"Final promotion gates failed for {ticker}: {'; '.join(reasons)}")
        return None

    predicted_log_returns = predicted_targets[0]

    from calendars import future_trading_dates

    future_date_strs, _ = future_trading_dates(
        ticker, str(origin_date_val.date()), MAX_FORECAST_DAYS
    )
    future_dates = [datetime.strptime(d, "%Y-%m-%d").date() for d in future_date_strs]

    if len(close_values) < HISTORY_DISPLAY_WINDOW:
        raise ValueError(
            f"Snapshot for {ticker} has fewer than {HISTORY_DISPLAY_WINDOW} close prices."
        )
    historical_prices = close_values[-HISTORY_DISPLAY_WINDOW:].tolist()
    historical_dates = [pd.Timestamp(d).date() for d in dates[-HISTORY_DISPLAY_WINDOW:]]

    snapshot_id = metadata.get("snapshot_id", "unknown")
    key = ServerArtifactKey.create(ticker=ticker, snapshot_id=snapshot_id)

    bundle = ServerForecastBundle(
        ticker=ticker,
        version_id=key.version_id,
        origin_close=origin_close,
        origin_date=origin_date_val.date(),
        future_dates=future_dates,
        predicted_log_returns=predicted_log_returns.tolist(),
        predicted_prices=predicted_prices.tolist(),
        historical_dates=historical_dates,
        historical_prices=historical_prices,
        evidence={
            "per_horizon": result["models"][best_candidate]["aggregate"]["per_horizon"],
            "pooled": result["models"][best_candidate]["aggregate"]["pooled"],
            "metric_source": "server_purged_walk_forward",
            "family": best_candidate,
        },
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

    signature = signer(bundle_bytes)

    record = ServerModelRecord(
        key=key,
        reproducibility=repro,
        sha256_digest=digest,
        signature=signature,
        status="candidate",
    )

    # 7. Immutable handoff: bundle first (never overwritten), then registry row,
    #    then promotion. Any failure after the bundle write leaves a harmless,
    #    unused immutable artifact rather than a pointer to a missing bundle.
    storage.put_bundle(key.version_id, bundle_bytes)
    registry.insert_artifact(record)
    promoted = registry.promote(key.version_id)

    logger.info(f"Promoted {best_candidate} for {ticker} (version {key.version_id})")
    return promoted
