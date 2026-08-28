#!/usr/bin/env python3
"""One-off CSCO multi-strategy backtest for the 2026-07-20..24 target week.

Trains several independently-strategised price forecasters on history that ends
at the last close strictly before the target week (2026-07-17), then produces
recursive 5-session point forecasts for 2026-07-20..2026-07-24 and scores them
against realised closes. Results are ordered by mean absolute percentage error
(MAPE) and reported with RMSE, MAE, and R2.

Deterministic: fixed seeds, GPU (CUDA) torch training with CPU fallback, no
randomness beyond the seeds.

Educational research only -- not financial advice.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

TICKER = "CSCO"
TRAIN_END = pd.Timestamp("2026-07-17")
TARGET_DAYS = pd.bdate_range("2026-07-20", "2026-07-24")
HORIZON = len(TARGET_DAYS)
SEED = 41
LAGS = (1, 2, 3, 5, 10)
WINDOW = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURES = (
    [f"lag_{lag}" for lag in LAGS]
    + ["vol_5", "vol_20", "ma5_ratio", "ma20_ratio", "volume_z"]
)


def _load_history() -> pd.DataFrame:
    raw = yf.download(
        TICKER,
        start="2015-01-01",
        end="2026-07-25",
        auto_adjust=True,
        progress=False,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    frame = raw[["Close", "Volume"]].dropna().copy()
    frame.columns = ["close", "volume"]
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
    if frame.empty:
        raise RuntimeError("no history downloaded for CSCO")
    return frame


def _feature_frame(
    closes: np.ndarray,
    volumes: np.ndarray,
    index: pd.Index | None = None,
) -> pd.DataFrame:
    """Causal per-session features; every row uses only data at or before it."""
    frame = pd.DataFrame(
        {"close": closes, "volume": volumes.astype(float)},
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
    features["volume_z"] = frame["volume"] / volume_mean - 1.0
    return features


def _recursive_forecast(name: str, history: pd.DataFrame, predict_step) -> np.ndarray:
    """Walk the model forward one session at a time for HORIZON sessions.

    ``predict_step(feature_row) -> next_log_return`` sees only features
    computed from the (possibly synthetic) history strictly before the
    forecast session.
    """
    closes = history["close"].to_numpy(dtype=float).copy()
    volumes = history["volume"].to_numpy(dtype=float).copy()
    predictions: list[float] = []
    for _session in range(HORIZON):
        features = _feature_frame(closes, volumes)
        row = features.iloc[-1]
        if row[FEATURES].isna().any():
            raise RuntimeError(f"{name}: feature row is incomplete")
        next_log_return = float(predict_step(row[FEATURES].to_numpy(dtype=float)))
        if not np.isfinite(next_log_return):
            raise RuntimeError(f"{name}: non-finite prediction")
        next_close = float(closes[-1]) * float(np.exp(next_log_return))
        predictions.append(next_close)
        closes = np.append(closes, next_close)
        volumes = np.append(volumes, volumes[-1])
    return np.asarray(predictions, dtype=float)


@dataclass(frozen=True)
class Forecast:
    name: str
    strategy: str
    predictions: np.ndarray


def _train_rows(history: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Train-only feature rows plus next-day log-return targets."""
    features = _feature_frame(
        history["close"].to_numpy(),
        history["volume"].to_numpy(),
        index=history.index,
    )
    next_return = np.log(history["close"]).diff().shift(-1)
    valid = features[FEATURES].notna().all(axis=1) & next_return.notna()
    # Exclude any row whose target session falls inside the target week.
    valid &= history.index.to_numpy() < np.datetime64(TRAIN_END)
    x = features.loc[valid, FEATURES].to_numpy(dtype=float)
    y = next_return.loc[valid].to_numpy(dtype=float)
    return features, x, y


def _fit_supervised(history: pd.DataFrame) -> tuple[StandardScaler, Ridge, HistGradientBoostingRegressor]:
    _features, x, y = _train_rows(history)
    scaler = StandardScaler().fit(x)
    ridge = Ridge(alpha=1.0).fit(scaler.transform(x), y)
    boosting = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_depth=3,
        l2_regularization=0.1,
        random_state=SEED,
    ).fit(x, y)
    return scaler, ridge, boosting


class _ReturnLSTM(nn.Module):
    def __init__(self, feature_count: int, hidden: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(feature_count, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(batch)
        return self.head(output[:, -1, :]).squeeze(-1)


def _train_lstm(history: pd.DataFrame) -> tuple[StandardScaler, _ReturnLSTM]:
    """Train the sequence model on GPU (CUDA) when available, CPU otherwise."""
    torch.manual_seed(SEED)
    features, _x, _y = _train_rows(history)
    next_return = np.log(history["close"]).diff().shift(-1)
    valid = features[FEATURES].notna().all(axis=1) & next_return.notna()
    valid &= history.index.to_numpy() < np.datetime64(TRAIN_END)
    full = features[FEATURES].to_numpy(dtype=float)
    targets = next_return.to_numpy(dtype=float)
    scaler = StandardScaler().fit(full[valid.to_numpy()])
    scaled = scaler.transform(full)

    valid_indices = np.flatnonzero(valid.to_numpy())
    windows = []
    labels = []
    for i in valid_indices:
        if i < WINDOW - 1:
            continue
        window = scaled[i - WINDOW + 1 : i + 1]
        if not np.isfinite(window).all() or not np.isfinite(targets[i]):
            continue
        windows.append(window)
        labels.append(targets[i])
    x = torch.tensor(np.asarray(windows), dtype=torch.float32, device=DEVICE)
    y = torch.tensor(np.asarray(labels), dtype=torch.float32, device=DEVICE)

    model = _ReturnLSTM(len(FEATURES), 32).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    model.train()
    for _epoch in range(60):
        permutation = torch.randperm(x.shape[0], generator=generator).to(DEVICE)
        for start in range(0, x.shape[0], 64):
            batch = permutation[start : start + 64]
            optimizer.zero_grad()
            loss = loss_fn(model(x[batch]), y[batch])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    model.eval()
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise RuntimeError("LSTM training diverged to non-finite weights")
    print(f"trained LSTM on device: {DEVICE}")
    return scaler, model


def _lstm_forecast(
    train_history: pd.DataFrame,
    scaler: StandardScaler,
    model: _ReturnLSTM,
) -> np.ndarray:
    closes = train_history["close"].to_numpy(dtype=float).copy()
    volumes = train_history["volume"].to_numpy(dtype=float).copy()
    predictions: list[float] = []
    with torch.no_grad():
        for _session in range(HORIZON):
            features = _feature_frame(closes, volumes)
            matrix = features[FEATURES].to_numpy(dtype=float)
            window = torch.tensor(
                scaler.transform(matrix[-WINDOW:])[np.newaxis, ...],
                dtype=torch.float32,
                device=DEVICE,
            )
            next_log_return = float(model(window).item())
            if not np.isfinite(next_log_return):
                raise RuntimeError("LSTM produced a non-finite return forecast")
            next_close = float(closes[-1]) * float(np.exp(next_log_return))
            predictions.append(next_close)
            closes = np.append(closes, next_close)
            volumes = np.append(volumes, volumes[-1])
    return np.asarray(predictions, dtype=float)


def main() -> int:
    np.random.seed(SEED)
    history = _load_history()
    train_history = history.loc[history.index <= TRAIN_END]
    actuals = history.loc[TARGET_DAYS, "close"].to_numpy(dtype=float)
    if len(actuals) != HORIZON or np.isnan(actuals).any():
        raise RuntimeError("target-week actual closes are incomplete")

    last_close = float(train_history["close"].iloc[-1])
    drift = float(np.log(train_history["close"]).diff().dropna().mean())
    last_week = train_history["close"].iloc[-HORIZON:].to_numpy(dtype=float)

    scaler, ridge, boosting = _fit_supervised(train_history)
    lstm_scaler, lstm_model = _train_lstm(train_history)

    def ridge_step(row: np.ndarray) -> float:
        return float(ridge.predict(scaler.transform(row[np.newaxis, :]))[0])

    def boosting_step(row: np.ndarray) -> float:
        return float(boosting.predict(row[np.newaxis, :])[0])

    forecasts = [
        Forecast("naive_flat", "persistence (last close)", np.full(HORIZON, last_close)),
        Forecast(
            "drift_random_walk",
            "random walk with drift",
            last_close * np.exp(drift * np.arange(1, HORIZON + 1)),
        ),
        Forecast("seasonal_naive_5d", "last-week close repeat", last_week),
        Forecast(
            "ridge_lagged_returns",
            "ridge on causal lag features",
            _recursive_forecast("ridge", train_history, ridge_step),
        ),
        Forecast(
            "hgb_boosting",
            "gradient boosting on causal lag features",
            _recursive_forecast("boosting", train_history, boosting_step),
        ),
        Forecast(
            "lstm_window",
            f"LSTM on {WINDOW}-session windows ({DEVICE})",
            _lstm_forecast(train_history, lstm_scaler, lstm_model),
        ),
    ]

    print(f"\nCSCO actual closes (target week {TARGET_DAYS[0].date()}..{TARGET_DAYS[-1].date()}):")
    for day, close in zip(TARGET_DAYS, actuals, strict=True):
        print(f"  {day.date()}  {close:.2f}")
    print(f"\nLast training close ({TRAIN_END.date()}): {last_close:.2f}\n")

    rows = []
    for forecast in forecasts:
        error = forecast.predictions - actuals
        if not np.isfinite(forecast.predictions).all():
            raise RuntimeError(f"{forecast.name} produced non-finite predictions: {forecast.predictions}")
        rows.append(
            {
                "model": forecast.name,
                "strategy": forecast.strategy,
                "mape_pct": float(np.mean(np.abs(error) / actuals) * 100.0),
                "rmse": float(np.sqrt(mean_squared_error(actuals, forecast.predictions))),
                "mae": float(mean_absolute_error(actuals, forecast.predictions)),
                "r2": float(r2_score(actuals, forecast.predictions)),
            }
        )
    table = pd.DataFrame(rows).sort_values("mape_pct").reset_index(drop=True)

    print("Per-day predictions:")
    header = "model".ljust(20) + "".join(day.strftime("%Y-%m-%d").rjust(12) for day in TARGET_DAYS)
    print(header)
    print("actual".ljust(20) + "".join(f"{value:12.2f}" for value in actuals))
    for forecast in forecasts:
        print(
            forecast.name.ljust(20)
            + "".join(f"{value:12.2f}" for value in forecast.predictions)
        )

    print("\nResults ordered by percentage error (MAPE):")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
