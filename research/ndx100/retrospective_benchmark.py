"""Point-in-time Nasdaq-100 retrospective benchmark harness (2022 -> 2026-08).

Evaluates classical and machine-learning price forecasters on weekly forecast
origins across point-in-time Nasdaq-100 constituents with strict no-lookahead
enforcement.

Models:
1. naive_flat (persistence / last close)
2. drift_random_walk (random walk with empirical drift)
3. seasonal_naive_5d (5-session cyclical repetition)
4. ridge_lagged_returns (Ridge regression on causal lag features with recursive stepping)
5. hgb_boosting (HistGradientBoosting on causal lag features with recursive stepping)
6. lstm_window (PyTorch LSTM on 20-session windows with recursive stepping)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .data import load_ticker_history
from .universe import (
    assert_survivorship_bias_resistant,
    get_ndx100_constituents,
    get_weekly_origins,
)

logger = logging.getLogger(__name__)

LAGS = (1, 2, 3, 5, 10)
FEATURE_COLUMNS = [f"lag_{lag}" for lag in LAGS] + [
    "vol_5",
    "vol_20",
    "ma5_ratio",
    "ma20_ratio",
    "volume_z",
]
LSTM_WINDOW = 20
DEFAULT_SEED = 41
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "ndx100-v9"


def compute_causal_features(
    closes: np.ndarray,
    volumes: np.ndarray,
    index: pd.Index | None = None,
) -> pd.DataFrame:
    """Compute causal per-session features; every row uses only data at or before it."""
    frame = pd.DataFrame(
        {"close": closes.astype(float), "volume": volumes.astype(float)},
        index=index,
    )
    log_close = np.log(frame["close"])
    log_return = log_close.diff()
    features = pd.DataFrame(index=frame.index)
    for lag in LAGS:
        features[f"lag_{lag}"] = log_return.shift(lag)
    features["vol_5"] = log_return.rolling(5).std()
    features["vol_20"] = log_return.rolling(20).std()
    features["ma5_ratio"] = frame["close"] / frame["close"].rolling(5).mean() - 1.0
    features["ma20_ratio"] = frame["close"] / frame["close"].rolling(20).mean() - 1.0
    volume_mean = frame["volume"].rolling(20).mean()
    features["volume_z"] = frame["volume"] / np.maximum(volume_mean, 1e-8) - 1.0
    return features


def recursive_step_forecast(
    history: pd.DataFrame,
    horizon: int,
    predict_step_fn: Callable[[np.ndarray], float],
) -> np.ndarray:
    """Roll forward one session at a time using causal features."""
    closes = history["close"].to_numpy(dtype=float).copy()
    volumes = history["volume"].to_numpy(dtype=float).copy()
    predictions: list[float] = []

    for _ in range(horizon):
        features = compute_causal_features(closes, volumes)
        row = features.iloc[-1]
        if row[FEATURE_COLUMNS].isna().any():
            # Fallback if insufficient history during recursion
            next_log_return = 0.0
        else:
            next_log_return = float(predict_step_fn(row[FEATURE_COLUMNS].to_numpy(dtype=float)))
            if not np.isfinite(next_log_return):
                next_log_return = 0.0

        next_close = float(closes[-1]) * float(np.exp(np.clip(next_log_return, -0.5, 0.5)))
        predictions.append(next_close)
        closes = np.append(closes, next_close)
        volumes = np.append(volumes, volumes[-1])

    return np.asarray(predictions, dtype=float)


class ReturnLSTM(nn.Module):
    def __init__(self, feature_count: int, hidden: int = 32) -> None:
        super().__init__()
        self.lstm = nn.LSTM(feature_count, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(batch)
        return self.head(output[:, -1, :]).squeeze(-1)


def fit_train_data(
    history: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    features = compute_causal_features(
        history["close"].to_numpy(),
        history["volume"].to_numpy(),
        index=history.index,
    )
    next_return = np.log(history["close"]).diff().shift(-1)
    valid = features[FEATURE_COLUMNS].notna().all(axis=1) & next_return.notna()
    x = features.loc[valid, FEATURE_COLUMNS].to_numpy(dtype=float)
    y = next_return.loc[valid].to_numpy(dtype=float)
    return x, y, features


def forecast_naive_flat(history: pd.DataFrame, horizon: int) -> np.ndarray:
    last_close = float(history["close"].iloc[-1])
    return np.full(horizon, last_close, dtype=float)


def forecast_drift(history: pd.DataFrame, horizon: int) -> np.ndarray:
    last_close = float(history["close"].iloc[-1])
    log_returns = np.log(history["close"]).diff().dropna()
    drift = float(log_returns.mean()) if len(log_returns) > 0 else 0.0
    return last_close * np.exp(drift * np.arange(1, horizon + 1))


def forecast_seasonal_naive(history: pd.DataFrame, horizon: int) -> np.ndarray:
    if len(history) >= horizon:
        return history["close"].iloc[-horizon:].to_numpy(dtype=float)
    last_close = float(history["close"].iloc[-1])
    return np.full(horizon, last_close, dtype=float)


def forecast_ridge(history: pd.DataFrame, horizon: int) -> np.ndarray:
    x, y, _ = fit_train_data(history)
    if len(x) < 20:
        return forecast_naive_flat(history, horizon)
    scaler = StandardScaler().fit(x)
    ridge = Ridge(alpha=1.0).fit(scaler.transform(x), y)

    def predict_step(row: np.ndarray) -> float:
        return float(ridge.predict(scaler.transform(row[np.newaxis, :]))[0])

    return recursive_step_forecast(history, horizon, predict_step)


def forecast_hgb(history: pd.DataFrame, horizon: int, seed: int = DEFAULT_SEED) -> np.ndarray:
    x, y, _ = fit_train_data(history)
    if len(x) < 20:
        return forecast_naive_flat(history, horizon)
    boosting = HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.05,
        max_depth=3,
        l2_regularization=0.1,
        random_state=seed,
    ).fit(x, y)

    def predict_step(row: np.ndarray) -> float:
        return float(boosting.predict(row[np.newaxis, :])[0])

    return recursive_step_forecast(history, horizon, predict_step)


def forecast_lstm(
    history: pd.DataFrame,
    horizon: int,
    seed: int = DEFAULT_SEED,
    epochs: int = 40,
) -> np.ndarray:
    x, y, features = fit_train_data(history)
    if len(x) < LSTM_WINDOW + 10:
        return forecast_naive_flat(history, horizon)

    torch.manual_seed(seed)
    full = features[FEATURE_COLUMNS].to_numpy(dtype=float)
    next_return = np.log(history["close"]).diff().shift(-1).to_numpy(dtype=float)
    valid = features[FEATURE_COLUMNS].notna().all(axis=1) & np.isfinite(next_return)
    valid_indices = np.flatnonzero(valid.to_numpy())

    scaler = StandardScaler().fit(full[valid_indices])
    scaled = scaler.transform(full)

    windows: list[np.ndarray] = []
    labels: list[float] = []
    for i in valid_indices:
        if i < LSTM_WINDOW - 1:
            continue
        win = scaled[i - LSTM_WINDOW + 1 : i + 1]
        if np.isfinite(win).all() and np.isfinite(next_return[i]):
            windows.append(win)
            labels.append(next_return[i])

    if len(windows) < 10:
        return forecast_naive_flat(history, horizon)

    x_tensor = torch.tensor(np.asarray(windows), dtype=torch.float32, device=DEVICE)
    y_tensor = torch.tensor(np.asarray(labels), dtype=torch.float32, device=DEVICE)

    model = ReturnLSTM(len(FEATURE_COLUMNS), hidden=32).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()
    generator = torch.Generator(device="cpu").manual_seed(seed)

    model.train()
    batch_size = 64
    for _ in range(epochs):
        perm = torch.randperm(x_tensor.shape[0], generator=generator).to(DEVICE)
        for start in range(0, x_tensor.shape[0], batch_size):
            batch = perm[start : start + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(x_tensor[batch]), y_tensor[batch])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()

    def predict_step(row: np.ndarray) -> float:
        with torch.no_grad():
            # For LSTM, we need the trailing window of scaled features
            # In recursive step, the model gets the last row; but for full window
            # we scale and feed the tensor
            val = float(
                model.head(
                    model.lstm(
                        torch.tensor(
                            scaler.transform(row[np.newaxis, :])[np.newaxis, ...],
                            dtype=torch.float32,
                            device=DEVICE,
                        )
                    )[0][:, -1, :]
                ).item()
            )
            return val

    # Full window recursive stepping
    closes = history["close"].to_numpy(dtype=float).copy()
    volumes = history["volume"].to_numpy(dtype=float).copy()
    predictions: list[float] = []

    with torch.no_grad():
        for _ in range(horizon):
            cur_features = compute_causal_features(closes, volumes)
            matrix = cur_features[FEATURE_COLUMNS].to_numpy(dtype=float)
            if len(matrix) < LSTM_WINDOW or np.isnan(matrix[-LSTM_WINDOW:]).any():
                next_return_val = 0.0
            else:
                window_tensor = torch.tensor(
                    scaler.transform(matrix[-LSTM_WINDOW:])[np.newaxis, ...],
                    dtype=torch.float32,
                    device=DEVICE,
                )
                next_return_val = float(model(window_tensor).item())
                if not np.isfinite(next_return_val):
                    next_return_val = 0.0
            next_close = float(closes[-1]) * float(np.exp(np.clip(next_return_val, -0.5, 0.5)))
            predictions.append(next_close)
            closes = np.append(closes, next_close)
            volumes = np.append(volumes, volumes[-1])

    return np.asarray(predictions, dtype=float)


@dataclass(frozen=True)
class BenchmarkConfig:
    start_date: str = "2022-01-01"
    end_date: str = "2026-08-28"
    include_neural: bool = False
    sample_stocks: int | None = None
    sample_weeks: int | None = None
    seed: int = DEFAULT_SEED
    results_dir: Path = DEFAULT_RESULTS_DIR


def run_retrospective_benchmark(
    config: BenchmarkConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = config or BenchmarkConfig()
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    origins = get_weekly_origins(cfg.start_date, cfg.end_date)
    if cfg.sample_weeks is not None:
        origins = origins[: cfg.sample_weeks]

    constituents_by_origin: dict[pd.Timestamp, list[str]] = {
        origin: get_ndx100_constituents(origin) for origin, _ in origins
    }

    # Assert PIT survivorship bias resistance
    if cfg.sample_weeks is None and cfg.sample_stocks is None:
        assert_survivorship_bias_resistant(constituents_by_origin)

    models: dict[str, Callable[[pd.DataFrame, int], np.ndarray]] = {
        "naive_flat": forecast_naive_flat,
        "drift_random_walk": forecast_drift,
        "seasonal_naive_5d": forecast_seasonal_naive,
        "ridge_lagged_returns": forecast_ridge,
        "hgb_boosting": lambda h, hz: forecast_hgb(h, hz, seed=cfg.seed),
    }
    if cfg.include_neural:
        models["lstm_window"] = lambda h, hz: forecast_lstm(h, hz, seed=cfg.seed)

    rows: list[dict[str, object]] = []
    total_evals = 0
    t0 = time.perf_counter()

    for origin_idx, (origin_dt, target_dts) in enumerate(origins):
        constituents = constituents_by_origin[origin_dt]
        if cfg.sample_stocks is not None:
            constituents = sorted(
                constituents,
                key=lambda ticker: hashlib.sha256(
                    f"{cfg.seed}|{origin_dt.date()}|{ticker}".encode()
                ).hexdigest(),
            )[: cfg.sample_stocks]

        for ticker in constituents:
            full_history = load_ticker_history(ticker)
            if full_history is None or full_history.empty:
                continue

            # Strict no-lookahead: history strictly at or before origin_dt
            train_history = full_history.loc[full_history.index <= origin_dt]
            if len(train_history) < 60:
                continue

            # Target week actual closes
            target_mask = full_history.index.isin(target_dts)
            target_closes = full_history.loc[target_mask, "close"]
            if len(target_closes) != len(target_dts):
                continue

            actuals = target_closes.to_numpy(dtype=float)
            last_close = float(train_history["close"].iloc[-1])
            actual_direction = int(np.sign(actuals[-1] - last_close))

            for model_name, model_fn in models.items():
                try:
                    preds = model_fn(train_history, len(target_dts))
                    if not np.isfinite(preds).all() or len(preds) != len(actuals):
                        continue
                    pred_direction = int(np.sign(preds[-1] - last_close))
                    direction_hit = int(pred_direction == actual_direction)

                    for session_h, (tgt_d, p_val, a_val) in enumerate(
                        zip(target_dts, preds, actuals, strict=True), start=1
                    ):
                        rows.append(
                            {
                                "origin_date": origin_dt.strftime("%Y-%m-%d"),
                                "ticker": ticker,
                                "model": model_name,
                                "session_h": session_h,
                                "target_date": tgt_d.strftime("%Y-%m-%d"),
                                "predicted_close": float(p_val),
                                "actual_close": float(a_val),
                                "abs_error": float(abs(p_val - a_val)),
                                "sq_error": float((p_val - a_val) ** 2),
                                "ape_pct": float(abs(p_val - a_val) / a_val * 100.0),
                                "direction_hit": direction_hit,
                            }
                        )
                    total_evals += 1
                except Exception as exc:
                    logger.debug(
                        "Forecast failed for %s on %s (%s): %s", ticker, origin_dt, model_name, exc
                    )

        if (origin_idx + 1) % 20 == 0 or (origin_idx + 1) == len(origins):
            logger.info(
                "Processed %d / %d origins (%d evaluations so far)...",
                origin_idx + 1,
                len(origins),
                total_evals,
            )

    elapsed = time.perf_counter() - t0
    df_preds = pd.DataFrame(rows)

    if df_preds.empty:
        raise RuntimeError("Retrospective benchmark produced no predictions!")

    # Save detailed predictions to parquet
    preds_file = cfg.results_dir / "predictions.parquet"
    df_preds.to_parquet(preds_file, index=False)
    logger.info("Saved %d prediction rows to %s", len(df_preds), preds_file)

    # Compute summary ranking across models
    summary_rows: list[dict[str, object]] = []
    for model_name, group in df_preds.groupby("model"):
        # Distinct origin-ticker forecasts
        unique_evals = group.groupby(["origin_date", "ticker"]).first()
        summary_rows.append(
            {
                "model": model_name,
                "eval_count": len(unique_evals),
                "mape_pct": float(group["ape_pct"].mean()),
                "rmse": float(np.sqrt(group["sq_error"].mean())),
                "mae": float(group["abs_error"].mean()),
                "direction_accuracy_pct": float(unique_evals["direction_hit"].mean() * 100.0),
                "duration_seconds": elapsed,
            }
        )

    df_summary = pd.DataFrame(summary_rows).sort_values("mape_pct").reset_index(drop=True)
    summary_json_file = cfg.results_dir / "summary.json"
    summary_dict = {
        "artifact_role": (
            "development_smoke_evidence"
            if cfg.sample_stocks is not None or cfg.sample_weeks is not None
            else "development_retrospective_evidence"
        ),
        "complete_universe_run": cfg.sample_stocks is None and cfg.sample_weeks is None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "period": f"{cfg.start_date} to {cfg.end_date}",
        "total_origins": len(origins),
        "models": df_summary.to_dict(orient="records"),
        "cuda_used": torch.cuda.is_available() and cfg.include_neural,
        "sampling": {
            "stock_limit_per_origin": cfg.sample_stocks,
            "week_limit": cfg.sample_weeks,
            "seed": cfg.seed,
        },
    }
    with summary_json_file.open("w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)

    return df_preds, df_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run NDX-100 retrospective benchmark")
    parser.add_argument(
        "--smoke", action="store_true", help="Run quick smoke test (5 stocks x 8 weeks)"
    )
    parser.add_argument(
        "--sample-stocks",
        type=int,
        default=None,
        help="Limit number of stocks evaluated per origin",
    )
    parser.add_argument(
        "--sample-weeks", type=int, default=None, help="Limit number of weekly origins evaluated"
    )
    parser.add_argument(
        "--include-neural", action="store_true", help="Include PyTorch LSTM window model"
    )
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--end-date", default="2026-08-28")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    sample_stocks = 5 if args.smoke else args.sample_stocks
    sample_weeks = 8 if args.smoke else args.sample_weeks

    config = BenchmarkConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        include_neural=args.include_neural,
        sample_stocks=sample_stocks,
        sample_weeks=sample_weeks,
        seed=args.seed,
    )

    print("\nRunning NDX-100 Retrospective Benchmark...")
    print(f"Period: {config.start_date} to {config.end_date}")
    print(f"Sample stocks: {config.sample_stocks}, Sample weeks: {config.sample_weeks}")
    print(f"Include neural (LSTM): {config.include_neural} (CUDA: {torch.cuda.is_available()})")

    _preds, summary = run_retrospective_benchmark(config)

    print("\n" + "=" * 80)
    print("NDX-100 Retrospective Benchmark Results (Ordered by MAPE):")
    print("=" * 80)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("=" * 80 + "\n")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
