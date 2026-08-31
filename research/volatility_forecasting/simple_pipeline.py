"""Readable, leakage-safe volatility forecasting pipeline.

This module is the deliberately small research path for the project.  It
operates on one OHLCV frame at a time, defines one canonical realised-
volatility target, and provides matched statistical and learned baselines.
The older ``v8``--``v11`` modules remain available for historical reproduction,
but the active portfolio benchmark should use this module.

The target at origin ``t`` is annualised future realised volatility over the
next ``H`` sessions::

    RV(t, H) = sqrt(252 / H * sum(r[t+1:t+H+1] ** 2))

where ``r[t+1] = log(C[t+1] / C[t])``.  Every feature is computed using rows
through ``t`` only.  Splits add an ``H``-session embargo so a training label
cannot overlap the first validation/test origin.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler

PIPELINE_VERSION = "simple-volatility-v1"
TARGET_VERSION = "future-realized-volatility-annualized-v1"
DEFAULT_ANNUALIZATION = 252.0
_EPS = 1e-12


@dataclass(frozen=True)
class VolatilityConfig:
    """Small, explicit configuration for one reproducible experiment."""

    horizon: int = 5
    lookback: int = 22
    annualization: float = DEFAULT_ANNUALIZATION
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    embargo_sessions: int | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        if self.horizon < 1 or self.lookback < 2:
            raise ValueError("horizon and lookback must be positive")
        if not math.isfinite(self.annualization) or self.annualization <= 0:
            raise ValueError("annualization must be finite and positive")
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must be in (0, 1)")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in (0, 1)")
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("train and validation fractions must leave a test set")
        if self.embargo_sessions is not None and self.embargo_sessions < self.horizon:
            raise ValueError("embargo_sessions must be at least the forecast horizon")

    @property
    def embargo(self) -> int:
        return max(self.horizon, int(self.embargo_sessions or 0))


@dataclass(frozen=True)
class ChronologicalSplit:
    """Disjoint chronological indices with explicit purge/embargo gaps."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    train_end: int
    validation_end: int
    embargo_sessions: int

    def __post_init__(self) -> None:
        groups = (self.train, self.validation, self.test)
        if any(np.asarray(group).ndim != 1 for group in groups):
            raise ValueError("split indices must be one-dimensional")
        if any(len(np.unique(group)) != len(group) for group in groups):
            raise ValueError("split indices must be unique")
        if set(self.train) & set(self.validation) or set(self.train) & set(self.test):
            raise ValueError("chronological split partitions overlap")
        if set(self.validation) & set(self.test):
            raise ValueError("chronological split partitions overlap")
        if not len(self.train) or not len(self.validation) or not len(self.test):
            raise ValueError("chronological split requires non-empty partitions")
        if self.embargo_sessions < 1:
            raise ValueError("split must retain a positive label embargo")


def chronological_split(
    n_rows: int,
    *,
    horizon: int,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    embargo_sessions: int | None = None,
) -> ChronologicalSplit:
    """Return a 70/15/15-style split with labels purged at both boundaries.

    An origin ``t`` labels prices through ``t + horizon``.  The first
    validation/test origin therefore starts at least ``horizon`` rows after
    the preceding partition's final origin.
    """

    if n_rows < 1 or horizon < 1:
        raise ValueError("n_rows and horizon must be positive")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("split fractions must be in (0, 1)")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("split fractions must leave a test partition")
    embargo = max(horizon, int(embargo_sessions or 0))
    raw_train_end = int(np.floor(n_rows * train_fraction))
    raw_validation_end = int(np.floor(n_rows * (train_fraction + validation_fraction)))
    validation_start = raw_train_end + embargo
    test_start = raw_validation_end + embargo
    if raw_train_end < 1 or validation_start >= raw_validation_end or test_start >= n_rows:
        raise ValueError("not enough rows for requested chronological split and embargo")
    train = np.arange(0, raw_train_end, dtype=np.int64)
    validation = np.arange(validation_start, raw_validation_end, dtype=np.int64)
    test = np.arange(test_start, n_rows, dtype=np.int64)
    return ChronologicalSplit(
        train=train,
        validation=validation,
        test=test,
        train_end=raw_train_end,
        validation_end=raw_validation_end,
        embargo_sessions=embargo,
    )


def validate_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalise and validate a single historical OHLCV frame."""

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("OHLCV frame must be a non-empty DataFrame")
    out = frame.copy()
    rename = {str(column).strip().lower(): column for column in out.columns}
    aliases = {
        "date": "Date",
        "datetime": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "adj close": "Adj Close",
        "adjusted_close": "Adj Close",
        "volume": "Volume",
    }
    selected: dict[str, Any] = {}
    for key, canonical in aliases.items():
        source = rename.get(key)
        if source is not None and canonical not in selected:
            selected[canonical] = source
    if "Close" not in selected:
        raise ValueError("OHLCV frame must contain a Close column")
    if not isinstance(out.index, pd.DatetimeIndex):
        date_source = selected.get("Date")
        if date_source is None:
            raise ValueError("OHLCV frame needs a DatetimeIndex or Date column")
        out.index = pd.to_datetime(out[date_source], errors="raise", utc=True).dt.tz_localize(None)
    else:
        out.index = pd.to_datetime(out.index, errors="raise").tz_localize(None)
    out = out.rename(columns={source: canonical for canonical, source in selected.items()})
    out = out.sort_index()
    if out.index.has_duplicates or not out.index.is_monotonic_increasing:
        raise ValueError("OHLCV timestamps must be unique and increasing")
    required = ["Close"]
    optional = [name for name in ("Open", "High", "Low", "Volume", "Adj Close") if name in out]
    for name in required + optional:
        out[name] = pd.to_numeric(out[name], errors="coerce")
    if not np.isfinite(out[required + optional].to_numpy(dtype=float)).all():
        raise ValueError("OHLCV values must be finite")
    if (out["Close"] <= 0).any():
        raise ValueError("Close values must be positive")
    if "Volume" in out and (out["Volume"] < 0).any():
        raise ValueError("Volume values cannot be negative")
    if {"Open", "High", "Low"}.issubset(out.columns):
        if (out["High"] < out[["Open", "Close"]].max(axis=1)).any():
            raise ValueError("High must be at least Open and Close")
        if (out["Low"] > out[["Open", "Close"]].min(axis=1)).any():
            raise ValueError("Low must be at most Open and Close")
    return out


def realised_volatility(
    close: pd.Series | np.ndarray, horizon: int, *, annualization: float = 252.0
) -> np.ndarray:
    """Return annualised future realised volatility at every possible origin."""

    prices = np.asarray(close, dtype=np.float64).reshape(-1)
    if horizon < 1 or len(prices) <= horizon:
        raise ValueError("close history is too short for the requested horizon")
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("close prices must be finite and positive")
    returns = np.log(prices[1:] / prices[:-1])
    squared = returns**2
    prefix = np.concatenate(([0.0], np.cumsum(squared)))
    target = np.full(len(prices), np.nan, dtype=np.float64)
    for origin in range(len(prices) - horizon):
        total = prefix[origin + horizon] - prefix[origin]
        target[origin] = math.sqrt(max(float(total) * annualization / horizon, _EPS))
    return target


def build_feature_frame(frame: pd.DataFrame, *, annualization: float = 252.0) -> pd.DataFrame:
    """Build causal market features; no value reads beyond the current row."""

    data = validate_ohlcv(frame)
    close = data["Close"]
    log_close = np.log(close)
    returns = log_close.diff()
    out = pd.DataFrame(index=data.index)
    out["return_1d"] = returns
    out["abs_return_1d"] = returns.abs()
    for window in (5, 22, 60):
        out[f"realized_vol_{window}"] = returns.pow(2).rolling(
            window, min_periods=window
        ).mean().pow(0.5) * math.sqrt(annualization)
    out["ewma_vol"] = returns.pow(2).ewm(alpha=1 - 0.94, adjust=False, min_periods=5).mean().pow(
        0.5
    ) * math.sqrt(annualization)
    out["return_mean_5"] = returns.rolling(5, min_periods=5).mean()
    out["return_mean_22"] = returns.rolling(22, min_periods=22).mean()
    out["return_std_22"] = returns.rolling(22, min_periods=22).std()
    if {"High", "Low"}.issubset(data.columns):
        out["log_range"] = np.log(data["High"] / data["Low"])
    else:
        out["log_range"] = returns.abs()
    if "Open" in data.columns:
        out["overnight_return"] = np.log(data["Open"] / close.shift(1))
    else:
        out["overnight_return"] = 0.0
    if "Volume" in data.columns:
        out["log_volume_change"] = np.log1p(data["Volume"]).diff()
    else:
        out["log_volume_change"] = 0.0
    return out.replace([np.inf, -np.inf], np.nan)


@dataclass(frozen=True)
class VolatilityExamples:
    """Sequence examples and the causal quantities used by baselines."""

    sequences: np.ndarray
    target: np.ndarray
    dates: np.ndarray
    feature_names: tuple[str, ...]
    har_features: np.ndarray
    current_volatility: np.ndarray
    rolling_mean_volatility: np.ndarray
    ewma_volatility: np.ndarray
    origin_close: np.ndarray | None = None
    future_close: np.ndarray | None = None

    def __post_init__(self) -> None:
        rows = len(self.sequences)
        if self.sequences.ndim != 3 or self.sequences.shape[0] != rows:
            raise ValueError("sequences must have shape [rows, lookback, features]")
        arrays = [
            self.target,
            self.dates,
            self.har_features,
            self.current_volatility,
            self.rolling_mean_volatility,
            self.ewma_volatility,
        ]
        if self.origin_close is not None:
            arrays.append(self.origin_close)
        if self.future_close is not None:
            arrays.append(self.future_close)
        if any(len(values) != rows for values in arrays):
            raise ValueError("example arrays must have matching row counts")
        if not np.isfinite(self.sequences).all() or not np.isfinite(self.target).all():
            raise ValueError("examples contain non-finite values")
        if (self.target <= 0).any():
            raise ValueError("volatility targets must be positive")


def build_examples(
    frame: pd.DataFrame,
    config: VolatilityConfig | None = None,
) -> VolatilityExamples:
    """Construct causal lookback sequences and strictly future targets."""

    settings = config or VolatilityConfig()
    data = validate_ohlcv(frame)
    features = build_feature_frame(data, annualization=settings.annualization)
    target = realised_volatility(
        data["Close"], settings.horizon, annualization=settings.annualization
    )
    feature_names = tuple(features.columns)
    values = features.to_numpy(dtype=np.float64)
    har = features[["realized_vol_5", "realized_vol_22", "realized_vol_60"]].to_numpy(
        dtype=np.float64
    )
    rows: list[np.ndarray] = []
    targets: list[float] = []
    dates: list[np.datetime64] = []
    har_rows: list[np.ndarray] = []
    current: list[float] = []
    rolling: list[float] = []
    ewma: list[float] = []
    origin_closes: list[float] = []
    future_closes: list[float] = []
    first = max(settings.lookback - 1, 60)
    for origin in range(first, len(data) - settings.horizon):
        window = values[origin - settings.lookback + 1 : origin + 1]
        target_value = target[origin]
        if not (
            np.isfinite(window).all()
            and np.isfinite(target_value)
            and np.isfinite(har[origin]).all()
        ):
            continue
        rows.append(window)
        targets.append(float(target_value))
        dates.append(np.datetime64(data.index[origin].date()))
        har_rows.append(har[origin])
        current.append(float(features["realized_vol_22"].iloc[origin]))
        rolling.append(float(features["realized_vol_60"].iloc[origin]))
        ewma.append(float(features["ewma_vol"].iloc[origin]))
        origin_closes.append(float(data["Close"].iloc[origin]))
        future_closes.append(float(data["Close"].iloc[origin + settings.horizon]))
    if not rows:
        raise ValueError("history did not produce any complete volatility examples")
    return VolatilityExamples(
        sequences=np.asarray(rows, dtype=np.float32),
        target=np.asarray(targets, dtype=np.float64),
        dates=np.asarray(dates, dtype="datetime64[D]"),
        feature_names=feature_names,
        har_features=np.asarray(har_rows, dtype=np.float64),
        current_volatility=np.asarray(current, dtype=np.float64),
        rolling_mean_volatility=np.asarray(rolling, dtype=np.float64),
        ewma_volatility=np.asarray(ewma, dtype=np.float64),
        origin_close=np.asarray(origin_closes, dtype=np.float64),
        future_close=np.asarray(future_closes, dtype=np.float64),
    )


@dataclass(frozen=True)
class LSTMConfig:
    """Bounded offline LSTM settings for a fair baseline comparison.

    PyTorch is imported only when :func:`lstm_predictions` is requested, so
    this optional research model can never become a production API
    dependency.  The target is log volatility and the scaler is fitted only
    on the supplied training rows.
    """

    hidden_size: int = 32
    dropout: float = 0.20
    maximum_epochs: int = 25
    patience: int = 5
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    device: str | None = None

    def __post_init__(self) -> None:
        if self.hidden_size < 4 or self.maximum_epochs < 1 or self.patience < 1:
            raise ValueError("LSTM size, epoch, and patience settings must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("LSTM dropout must be in [0, 1)")
        if self.batch_size < 1 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("LSTM optimizer and batch settings are invalid")


def lstm_predictions(
    examples: VolatilityExamples,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    *,
    config: LSTMConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a compact LSTM on log volatility and predict every example.

    ``validation_indices`` is kept outside the optimizer's training rows and
    is used only for early stopping.  The function is intentionally offline;
    importing it does not import PyTorch and no weights are persisted.
    """

    settings = config or LSTMConfig()
    train = np.asarray(train_indices, dtype=np.int64)
    validation = np.asarray(validation_indices, dtype=np.int64)
    if train.ndim != 1 or validation.ndim != 1 or len(train) < 8 or len(validation) < 2:
        raise ValueError("LSTM requires non-empty train and validation partitions")
    if set(train) & set(validation):
        raise ValueError("LSTM train and validation partitions overlap")

    try:
        import torch
        from torch import nn
    except ImportError as err:  # pragma: no cover - depends on optional environment
        raise RuntimeError(
            "PyTorch is required for --include-lstm; install an offline CPU or CUDA build."
        ) from err

    torch.manual_seed(settings.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(settings.seed)
    device = torch.device(settings.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    feature_count = int(examples.sequences.shape[-1])
    scaler = StandardScaler().fit(
        examples.sequences[train].reshape(-1, feature_count).astype(np.float64)
    )
    scaled = (
        scaler.transform(examples.sequences.reshape(-1, feature_count).astype(np.float64))
        .reshape(examples.sequences.shape)
        .astype(np.float32)
    )
    log_target = np.log(_positive(examples.target)).astype(np.float32)

    class VolatilityLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.LSTM(
                feature_count,
                settings.hidden_size,
                batch_first=True,
            )
            self.dropout = nn.Dropout(settings.dropout)
            self.head = nn.Linear(settings.hidden_size, 1)

        def forward(self, values: Any) -> Any:
            encoded, _ = self.encoder(values)
            return self.head(self.dropout(encoded[:, -1, :])).squeeze(-1)

    model = VolatilityLSTM().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    loss_fn = nn.SmoothL1Loss()
    train_x = torch.as_tensor(scaled[train], dtype=torch.float32, device=device)
    train_y = torch.as_tensor(log_target[train], dtype=torch.float32, device=device)
    val_x = torch.as_tensor(scaled[validation], dtype=torch.float32, device=device)
    val_y = torch.as_tensor(log_target[validation], dtype=torch.float32, device=device)
    started = time.perf_counter()
    best_state: dict[str, Any] | None = None
    best_loss = math.inf
    stale_epochs = 0
    completed_epochs = 0

    try:
        for epoch in range(settings.maximum_epochs):
            model.train()
            for start in range(0, len(train), settings.batch_size):
                stop = min(start + settings.batch_size, len(train))
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(train_x[start:stop]), train_y[start:stop])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                validation_loss = float(loss_fn(model(val_x), val_y).item())
            completed_epochs = epoch + 1
            if validation_loss < best_loss - 1e-5:
                best_loss = validation_loss
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= settings.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        all_x = torch.as_tensor(scaled, dtype=torch.float32, device=device)
        predictions: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(all_x), settings.batch_size):
                stop = min(start + settings.batch_size, len(all_x))
                predictions.append(model(all_x[start:stop]).detach().cpu().numpy())
        prediction = np.exp(np.clip(np.concatenate(predictions), -20.0, 5.0))
        if not np.isfinite(prediction).all() or (prediction <= 0).any():
            raise ValueError("LSTM produced non-finite or non-positive volatility")
        metadata = {
            "family": "lstm",
            "hidden_size": settings.hidden_size,
            "dropout": settings.dropout,
            "epochs": completed_epochs,
            "best_validation_log_loss": None if not np.isfinite(best_loss) else best_loss,
            "device": str(device),
            "training_seconds": time.perf_counter() - started,
            "scaler": "train_only_standard",
        }
        return np.asarray(prediction, dtype=np.float64), metadata
    finally:
        del model, optimizer, train_x, train_y, val_x, val_y
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _positive(values: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(values, dtype=np.float64), _EPS)


def fit_har_baseline(examples: VolatilityExamples, train_indices: np.ndarray) -> np.ndarray:
    """Fit log-HAR on training rows and predict all rows without target reads."""

    train = np.asarray(train_indices, dtype=np.int64)
    if train.ndim != 1 or len(train) < 10:
        raise ValueError("HAR baseline requires at least ten training rows")
    x_train = np.column_stack(
        (np.ones(len(train)), np.log(_positive(examples.har_features[train])))
    )
    y_train = np.log(_positive(examples.target[train]))
    coefficients, *_ = np.linalg.lstsq(x_train, y_train, rcond=None)
    x_all = np.column_stack(
        (np.ones(len(examples.target)), np.log(_positive(examples.har_features)))
    )
    return np.exp(np.clip(x_all @ coefficients, math.log(_EPS), math.log(10.0)))


def baseline_predictions(
    examples: VolatilityExamples,
    train_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return persistence, rolling mean, EWMA, and train-fitted HAR forecasts."""

    return {
        "persistence": _positive(examples.current_volatility),
        "rolling_mean": _positive(examples.rolling_mean_volatility),
        "ewma": _positive(examples.ewma_volatility),
        "har_rv": _positive(fit_har_baseline(examples, train_indices)),
    }


def _fit_scaled_regressor(
    examples: VolatilityExamples,
    train_indices: np.ndarray,
    estimator: Any,
) -> tuple[Any, StandardScaler]:
    train = np.asarray(train_indices, dtype=np.int64)
    x_train = examples.sequences[train].reshape(len(train), -1).astype(np.float64)
    scaler = StandardScaler().fit(x_train)
    estimator.fit(scaler.transform(x_train), np.log(_positive(examples.target[train])))
    return estimator, scaler


def learned_predictions(
    examples: VolatilityExamples,
    train_indices: np.ndarray,
    *,
    include_boosting: bool = True,
) -> dict[str, np.ndarray]:
    """Fit simple learned models using only the supplied training indices."""

    all_x = examples.sequences.reshape(len(examples.sequences), -1).astype(np.float64)
    predictions: dict[str, np.ndarray] = {}
    ridge, scaler = _fit_scaled_regressor(examples, train_indices, Ridge(alpha=1.0))
    predictions["ridge"] = _positive(
        np.exp(np.clip(ridge.predict(scaler.transform(all_x)), -20.0, 5.0))
    )
    elastic_net, elastic_scaler = _fit_scaled_regressor(
        examples,
        train_indices,
        ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000, tol=1e-4),
    )
    predictions["elastic_net"] = _positive(
        np.exp(np.clip(elastic_net.predict(elastic_scaler.transform(all_x)), -20.0, 5.0))
    )
    if include_boosting:
        boosting, boosting_scaler = _fit_scaled_regressor(
            examples,
            train_indices,
            GradientBoostingRegressor(
                learning_rate=0.05,
                n_estimators=200,
                max_depth=3,
                max_features="sqrt",
                random_state=42,
            ),
        )
        predictions["gradient_boosting"] = _positive(
            np.exp(np.clip(boosting.predict(boosting_scaler.transform(all_x)), -20.0, 5.0))
        )
    return predictions


def volatility_metrics(actual: np.ndarray, forecast: np.ndarray) -> dict[str, float]:
    """Calculate point errors plus QLIKE on variance, with lower being better."""

    observed = _positive(actual)
    predicted = _positive(forecast)
    if observed.shape != predicted.shape or observed.ndim != 1 or not len(observed):
        raise ValueError("metric arrays must be matched non-empty vectors")
    error = predicted - observed
    actual_variance = observed**2
    forecast_variance = predicted**2
    ratio = actual_variance / _positive(forecast_variance)
    return {
        "mae": float(np.mean(np.abs(error))),
        "mse": float(np.mean(error**2)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "qlike": float(np.mean(ratio - np.log(ratio) - 1.0)),
        "r2": float(1.0 - np.sum(error**2) / np.sum((observed - np.mean(observed)) ** 2))
        if np.sum((observed - np.mean(observed)) ** 2) > _EPS
        else 0.0,
    }


def evaluate_conformal_volatility_intervals(
    actual_validation: np.ndarray,
    forecast_validation: np.ndarray,
    actual_test: np.ndarray,
    forecast_test: np.ndarray,
    *,
    nominal_coverage: float = 0.90,
) -> dict[str, Any]:
    """Calibrate split-conformal intervals on validation residuals and evaluate on test."""

    val_act = _positive(actual_validation)
    val_pred = _positive(forecast_validation)
    test_act = _positive(actual_test)
    test_pred = _positive(forecast_test)
    if len(val_act) < 4 or len(test_act) < 4:
        raise ValueError("at least 4 validation and test observations required")

    log_residuals = np.abs(np.log(val_act) - np.log(val_pred))
    rank = min(int(np.ceil((len(log_residuals) + 1) * nominal_coverage)), len(log_residuals))
    radius = float(np.sort(log_residuals)[rank - 1])

    lower = test_pred * math.exp(-radius)
    upper = test_pred * math.exp(radius)

    inside = (test_act >= lower) & (test_act <= upper)
    empirical_coverage = float(np.mean(inside))
    average_width = float(np.mean(upper - lower))

    tertiles = np.quantile(test_act, [1.0 / 3.0, 2.0 / 3.0])
    regime_low = test_act <= tertiles[0]
    regime_normal = (test_act > tertiles[0]) & (test_act <= tertiles[1])
    regime_high = test_act > tertiles[1]

    def _regime_cov(mask: np.ndarray) -> float | None:
        count = int(np.sum(mask))
        return float(np.mean(inside[mask])) if count > 0 else None

    return {
        "nominal_coverage": float(nominal_coverage),
        "empirical_coverage": empirical_coverage,
        "conformal_log_radius": radius,
        "average_width": average_width,
        "regime_coverage": {
            "low_vol": _regime_cov(regime_low),
            "normal_vol": _regime_cov(regime_normal),
            "high_vol": _regime_cov(regime_high),
        },
    }


def evaluate_price_diffusion_cone(
    origin_close: np.ndarray,
    future_close: np.ndarray,
    forecast_annualized_vol: np.ndarray,
    horizon: int,
    *,
    nominal_coverage: float = 0.90,
    annualization: float = DEFAULT_ANNUALIZATION,
) -> dict[str, Any]:
    """Evaluate empirical coverage of the theoretical diffusion cone (e.g. p05-p95)."""

    p_orig = np.asarray(origin_close, dtype=np.float64)
    p_future = np.asarray(future_close, dtype=np.float64)
    vol = _positive(forecast_annualized_vol)
    if len(p_orig) != len(p_future) or len(p_orig) != len(vol) or len(p_orig) < 4:
        raise ValueError("matched arrays of at least 4 observations required")

    # Quantile z-scores for standard nominal coverages
    z = float(norm.ppf(0.5 + float(nominal_coverage) / 2.0))

    dt = horizon / annualization
    horizon_sigma = vol * math.sqrt(dt)
    lower = p_orig * np.exp(-z * horizon_sigma)
    upper = p_orig * np.exp(+z * horizon_sigma)

    inside = (p_future >= lower) & (p_future <= upper)
    empirical_coverage = float(np.mean(inside))
    avg_width_pct = float(np.mean((upper - lower) / p_orig))

    tertiles = np.quantile(vol, [1.0 / 3.0, 2.0 / 3.0])
    regime_low = vol <= tertiles[0]
    regime_normal = (vol > tertiles[0]) & (vol <= tertiles[1])
    regime_high = vol > tertiles[1]

    def _regime_cov(mask: np.ndarray) -> float | None:
        count = int(np.sum(mask))
        return float(np.mean(inside[mask])) if count > 0 else None

    return {
        "nominal_coverage": float(nominal_coverage),
        "empirical_coverage": empirical_coverage,
        "average_width_pct": avg_width_pct,
        "z_score": float(z),
        "regime_coverage": {
            "low_vol": _regime_cov(regime_low),
            "normal_vol": _regime_cov(regime_normal),
            "high_vol": _regime_cov(regime_high),
        },
    }


def evaluate_benchmark(
    examples: VolatilityExamples,
    split: ChronologicalSplit,
    *,
    include_boosting: bool = True,
    include_lstm: bool = False,
    lstm_config: LSTMConfig | None = None,
    nominal_coverage: float = 0.90,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Evaluate all baselines and simple ML models on validation and test."""

    forecasts = baseline_predictions(examples, split.train)
    forecasts.update(learned_predictions(examples, split.train, include_boosting=include_boosting))
    if include_lstm:
        lstm_forecast, _ = lstm_predictions(
            examples,
            split.train,
            split.validation,
            config=lstm_config,
        )
        forecasts["lstm"] = lstm_forecast
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for name, prediction in forecasts.items():
        output[name] = {}
        for partition, indices in (("validation", split.validation), ("test", split.test)):
            metrics = volatility_metrics(examples.target[indices], prediction[indices])
            metrics["rows"] = int(len(indices))
            output[name][partition] = metrics

        # Conformal prediction intervals on realized volatility
        try:
            val_act = examples.target[split.validation]
            val_pred = prediction[split.validation]
            test_act = examples.target[split.test]
            test_pred = prediction[split.test]
            output[name]["test"]["volatility_interval"] = evaluate_conformal_volatility_intervals(
                val_act, val_pred, test_act, test_pred, nominal_coverage=nominal_coverage
            )
        except Exception:
            pass

        # Price diffusion cone calibration
        if examples.origin_close is not None and examples.future_close is not None:
            try:
                output[name]["test"]["price_cone"] = evaluate_price_diffusion_cone(
                    examples.origin_close[split.test],
                    examples.future_close[split.test],
                    prediction[split.test],
                    horizon=split.embargo_sessions,
                    nominal_coverage=nominal_coverage,
                )
            except Exception:
                pass

    return output


def select_validation_model(metrics: dict[str, dict[str, dict[str, Any]]]) -> str:
    """Select by validation QLIKE only; test scores never influence selection."""

    if not metrics:
        raise ValueError("benchmark metrics are empty")
    return min(
        metrics,
        key=lambda name: (float(metrics[name]["validation"]["qlike"]), name),
    )


def experiment_metadata(
    examples: VolatilityExamples,
    config: VolatilityConfig,
    split: ChronologicalSplit,
    *,
    model: str,
    metrics: dict[str, dict[str, Any]],
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Build a small JSON-serialisable record for an experiment run."""

    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "target_version": TARGET_VERSION,
        "run_id": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "git_commit": git_commit,
        "model": model,
        "configuration": asdict(config),
        "feature_names": list(examples.feature_names),
        "rows": int(len(examples.target)),
        "training_rows": int(len(split.train)),
        "validation_rows": int(len(split.validation)),
        "test_rows": int(len(split.test)),
        "date_start": str(examples.dates[0]),
        "date_end": str(examples.dates[-1]),
        "metrics": metrics,
    }
    digest_payload = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["run_hash"] = hashlib.sha256(digest_payload).hexdigest()
    return payload


__all__ = [
    "PIPELINE_VERSION",
    "TARGET_VERSION",
    "ChronologicalSplit",
    "LSTMConfig",
    "VolatilityConfig",
    "VolatilityExamples",
    "baseline_predictions",
    "build_examples",
    "build_feature_frame",
    "chronological_split",
    "evaluate_benchmark",
    "evaluate_conformal_volatility_intervals",
    "evaluate_price_diffusion_cone",
    "experiment_metadata",
    "learned_predictions",
    "lstm_predictions",
    "realised_volatility",
    "select_validation_model",
    "validate_ohlcv",
    "volatility_metrics",
]
