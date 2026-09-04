"""Compact CUDA LSTM benchmark for the five-ticker seven-day product.

This module is deliberately separate from the production API.  It trains one
pooled model on stationary OHLCV features, evaluates it on an untouched final
15% time block, and only then refits a deployment candidate on all resolved
labels.  The test result and the refitted candidate are labelled separately.
"""

from __future__ import annotations

import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from .news_archive import (
    NEWS_FEATURE_NAMES,
    build_causal_news_features,
    load_news_archive,
    validate_news_archive,
)

DEFAULT_TICKERS = ("AAPL", "GOOGL", "MSFT", "NVDA", "TSLA")
TRI_EXCHANGE_TICKERS = (
    # NASDAQ (10)
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA",
    "AMD",
    "COST",
    "QCOM",
    # NYSE (10)
    "JPM",
    "XOM",
    "WMT",
    "JNJ",
    "CAT",
    "KO",
    "NEE",
    "DIS",
    "BAC",
    "GE",
    # LSE (10)
    "SHEL.L",
    "AZN.L",
    "HSBA.L",
    "BP.L",
    "ULVR.L",
    "GSK.L",
    "RIO.L",
    "BATS.L",
    "BARC.L",
    "DGE.L",
)
MODEL_VERSION = "pooled-price-lstm-v3-tri-exchange"
FEATURE_NAMES = (
    "return_1d",
    "return_2d",
    "return_5d",
    "return_10d",
    "return_20d",
    "range_1d",
    "overnight_return",
    "close_location",
    "volume_change",
    "drawdown_20d",
    "return_mean_5d",
    "return_vol_5d",
    "close_vs_sma_5d",
    "return_mean_10d",
    "return_vol_10d",
    "close_vs_sma_10d",
    "return_mean_20d",
    "return_vol_20d",
    "close_vs_sma_20d",
    "return_mean_60d",
    "return_vol_60d",
    "close_vs_sma_60d",
    "volume_z_20d",
    "weekday_sin",
    "weekday_cos",
)


@dataclass(frozen=True)
class PriceTrainingConfig:
    lookback: int = 60
    horizon: int = 7
    hidden_size: int = 64
    layers: int = 2
    dropout: float = 0.15
    batch_size: int = 128
    maximum_epochs: int = 60
    patience: int = 12
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    use_attention: bool = True
    direction_weight: float = 0.05
    feature_mode: str = "price_only"

    def __post_init__(self) -> None:
        if self.lookback < 20 or self.horizon < 1:
            raise ValueError("lookback and horizon are invalid")
        if self.hidden_size < 8 or self.layers < 1 or self.batch_size < 1:
            raise ValueError("model dimensions must be positive")
        if not 0 <= self.dropout < 1 or self.maximum_epochs < 1 or self.patience < 1:
            raise ValueError("training limits are invalid")
        if self.direction_weight < 0:
            raise ValueError("direction_weight must be non-negative")


@dataclass(frozen=True)
class GlobalPriceDataset:
    sequences: np.ndarray
    targets: np.ndarray
    ticker_indices: np.ndarray
    origin_positions: np.ndarray
    target_end_positions: np.ndarray
    origin_dates: np.ndarray
    split_train: np.ndarray
    split_validation: np.ndarray
    split_test: np.ndarray
    current_sequences: dict[str, np.ndarray]
    current_prices: dict[str, float]
    data_as_of: dict[str, str]
    ticker_names: tuple[str, ...]
    feature_names: tuple[str, ...] = FEATURE_NAMES
    feature_mode: str = "price_only"
    news_coverage: dict[str, Any] | None = None


def _normalise_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data.columns = [str(column).title() for column in data.columns]
    required = ("Open", "High", "Low", "Close", "Volume")
    if not set(required).issubset(data.columns):
        raise ValueError(f"OHLCV frame is missing {sorted(set(required) - set(data.columns))}")
    data = data.loc[:, required].apply(pd.to_numeric, errors="coerce")
    data.index = pd.to_datetime(data.index, errors="coerce").tz_localize(None)
    data = data.loc[~data.index.isna()]
    data = data.loc[~data.index.duplicated(keep="last")].sort_index()
    if len(data) < 500 or not np.isfinite(data.to_numpy(dtype=np.float64)).all():
        raise ValueError("OHLCV frame is too short or contains non-finite values")
    if (data[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("OHLC prices must be positive")
    data["High"] = np.maximum(data["High"], data[["Open", "Close", "Low"]].max(axis=1))
    data["Low"] = np.minimum(data["Low"], data[["Open", "Close", "High"]].min(axis=1))
    return data


def build_price_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return stationary, causal features computed through each session only."""
    data = _normalise_ohlcv(frame)
    close = data["Close"]
    log_close = np.log(close)
    returns = log_close.diff()
    log_volume = np.log1p(data["Volume"])
    denominator = (data["High"] - data["Low"]).replace(0.0, np.nan)
    columns: dict[str, pd.Series] = {
        "return_1d": returns,
        "return_2d": log_close.diff(2),
        "return_5d": log_close.diff(5),
        "return_10d": log_close.diff(10),
        "return_20d": log_close.diff(20),
        "range_1d": np.log(data["High"] / data["Low"]),
        "overnight_return": np.log(data["Open"] / close.shift(1)),
        "close_location": (close - data["Low"]) / denominator,
        "volume_change": log_volume.diff(),
        "drawdown_20d": close / close.rolling(20).max() - 1.0,
    }
    for window in (5, 10, 20, 60):
        columns[f"return_mean_{window}d"] = returns.rolling(window).mean()
        columns[f"return_vol_{window}d"] = returns.rolling(window).std(ddof=0)
        columns[f"close_vs_sma_{window}d"] = close / close.rolling(window).mean() - 1.0
    volume_mean = log_volume.rolling(20).mean()
    volume_std = log_volume.rolling(20).std(ddof=0).replace(0.0, np.nan)
    columns["volume_z_20d"] = (log_volume - volume_mean) / volume_std
    weekday = pd.Series(data.index.dayofweek, index=data.index, dtype=float)
    columns["weekday_sin"] = np.sin(2.0 * np.pi * weekday / 5.0)
    columns["weekday_cos"] = np.cos(2.0 * np.pi * weekday / 5.0)
    features = pd.DataFrame(columns, index=data.index).replace([np.inf, -np.inf], np.nan)
    return features.loc[:, FEATURE_NAMES].dropna()


def build_global_price_dataset(
    frames: dict[str, pd.DataFrame],
    config: PriceTrainingConfig | None = None,
    *,
    feature_mode: Literal["price_only", "price_plus_news"] = "price_only",
    news_archives: dict[str, list[dict[str, Any]]] | Path | str | None = None,
) -> GlobalPriceDataset:
    """Pool per-ticker sequence datasets with chronological 70/15/15 splits."""
    settings = config or PriceTrainingConfig(feature_mode=feature_mode)
    resolved_mode = feature_mode or settings.feature_mode
    tickers = tuple(sorted(symbol.strip().upper() for symbol in frames))
    if not tickers:
        raise ValueError("At least one ticker frame is required")

    effective_feature_names = FEATURE_NAMES
    news_coverage_summary: dict[str, Any] = {}
    parsed_archives: dict[str, list[dict[str, Any]]] = {}

    if resolved_mode == "price_plus_news":
        effective_feature_names = FEATURE_NAMES + NEWS_FEATURE_NAMES
        if news_archives is None:
            raise ValueError(
                "Historical news events missing; synthetic news or silent zero-fill is prohibited in news feature mode."
            )
        if isinstance(news_archives, (str, Path)):
            news_dir = Path(news_archives)
            if not news_dir.is_dir():
                raise FileNotFoundError(f"News archive directory does not exist: {news_dir}")
            for ticker in tickers:
                archive_file = news_dir / f"{ticker}.jsonl"
                if not archive_file.is_file():
                    raise ValueError(
                        f"Historical news archive missing for {ticker} at {archive_file}; synthetic news is prohibited."
                    )
                parsed_archives[ticker] = load_news_archive(archive_file)
        elif isinstance(news_archives, dict):
            for ticker in tickers:
                if ticker not in news_archives or not news_archives[ticker]:
                    raise ValueError(
                        f"Historical news events missing for {ticker}; synthetic news is prohibited."
                    )
                parsed_archives[ticker] = list(news_archives[ticker])

    sequences: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    ticker_indices: list[int] = []
    origins: list[int] = []
    target_ends: list[int] = []
    dates: list[str] = []
    train_indices: list[int] = []
    validation_indices: list[int] = []
    test_indices: list[int] = []
    current_sequences: dict[str, np.ndarray] = {}
    current_prices: dict[str, float] = {}
    data_as_of: dict[str, str] = {}

    for ticker_index, ticker in enumerate(tickers):
        data = _normalise_ohlcv(frames[ticker])
        feature_frame = build_price_features(data)

        if resolved_mode == "price_plus_news":
            events = parsed_archives[ticker]
            diag = validate_news_archive(events, data.index)
            news_coverage_summary[ticker] = diag
            if not diag["is_valid"]:
                raise ValueError(
                    f"News archive for {ticker} is invalid or has insufficient events: {diag.get('reason')}"
                )
            news_frame = build_causal_news_features(data.index, ticker=ticker, news_events=events)
            combined_features = pd.concat(
                [feature_frame, news_frame.loc[feature_frame.index]], axis=1
            )
            feature_frame = combined_features.loc[:, list(effective_feature_names)].dropna()

        feature_positions = pd.Series(np.arange(len(data), dtype=np.int64), index=data.index)
        feature_values = feature_frame.to_numpy(dtype=np.float32)
        close = np.log(data["Close"])
        train_boundary = int(len(data) * 0.70)
        test_boundary = int(len(data) * 0.85)
        current_sequences[ticker] = feature_values[-settings.lookback :].copy()
        if len(current_sequences[ticker]) != settings.lookback:
            raise ValueError(f"{ticker} does not have enough valid feature rows")
        current_prices[ticker] = float(data["Close"].iloc[-1])
        data_as_of[ticker] = data.index[-1].date().isoformat()

        for feature_row in range(settings.lookback - 1, len(feature_frame)):
            origin_date = feature_frame.index[feature_row]
            origin = int(feature_positions.loc[origin_date])
            target_end = origin + settings.horizon
            if target_end >= len(data):
                continue
            target = np.array(
                [
                    float(close.iloc[origin + step] - close.iloc[origin])
                    for step in range(1, settings.horizon + 1)
                ],
                dtype=np.float32,
            )
            sequence = feature_values[feature_row - settings.lookback + 1 : feature_row + 1]
            row = len(sequences)
            sequences.append(sequence)
            targets.append(target)
            ticker_indices.append(ticker_index)
            origins.append(origin)
            target_ends.append(target_end)
            dates.append(origin_date.date().isoformat())
            if target_end < train_boundary:
                train_indices.append(row)
            elif origin >= train_boundary and target_end < test_boundary:
                validation_indices.append(row)
            elif origin >= test_boundary:
                test_indices.append(row)

    split_arrays = tuple(
        np.asarray(values, dtype=np.int64)
        for values in (train_indices, validation_indices, test_indices)
    )
    if min(map(len, split_arrays)) < 30:
        raise ValueError("Chronological partitions are too small")
    sequence_array = np.asarray(sequences, dtype=np.float32)
    target_array = np.asarray(targets, dtype=np.float32)
    if not np.isfinite(sequence_array).all() or not np.isfinite(target_array).all():
        raise ValueError("Price dataset contains non-finite values")

    return GlobalPriceDataset(
        sequences=sequence_array,
        targets=target_array,
        ticker_indices=np.asarray(ticker_indices, dtype=np.int64),
        origin_positions=np.asarray(origins, dtype=np.int64),
        target_end_positions=np.asarray(target_ends, dtype=np.int64),
        origin_dates=np.asarray(dates),
        split_train=split_arrays[0],
        split_validation=split_arrays[1],
        split_test=split_arrays[2],
        current_sequences=current_sequences,
        current_prices=current_prices,
        data_as_of=data_as_of,
        ticker_names=tickers,
        feature_names=tuple(effective_feature_names),
        feature_mode=resolved_mode,
        news_coverage=news_coverage_summary if news_coverage_summary else None,
    )


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    errors = (np.exp(predicted) - np.exp(actual)) * 100.0
    baseline_errors = (1.0 - np.exp(actual)) * 100.0
    mae = float(np.mean(np.abs(errors)))
    baseline_mae = float(np.mean(np.abs(baseline_errors)))
    return {
        "mae_percent": mae,
        "rmse_percent": float(np.sqrt(np.mean(np.square(errors)))),
        "direction_accuracy": float(np.mean(np.sign(predicted) == np.sign(actual))),
        "persistence_mae_percent": baseline_mae,
        "relative_mae_vs_persistence": mae / baseline_mae if baseline_mae > 0 else None,
        "per_horizon": [
            {
                "day": day + 1,
                "mae_percent": float(np.mean(np.abs(errors[:, day]))),
                "rmse_percent": float(np.sqrt(np.mean(np.square(errors[:, day])))),
                "direction_accuracy": float(
                    np.mean(np.sign(predicted[:, day]) == np.sign(actual[:, day]))
                ),
                "relative_mae_vs_persistence": float(
                    np.mean(np.abs(errors[:, day])) / np.mean(np.abs(baseline_errors[:, day]))
                ),
            }
            for day in range(actual.shape[1])
        ],
    }


def _build_model(
    torch: Any,
    nn: Any,
    feature_count: int,
    ticker_count: int,
    settings: PriceTrainingConfig,
    embed_dim: int = 8,
) -> Any:
    class PooledPriceLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ticker_embedding = nn.Embedding(ticker_count, embed_dim)
            self.encoder = nn.LSTM(
                feature_count + embed_dim,
                settings.hidden_size,
                num_layers=settings.layers,
                dropout=settings.dropout if settings.layers > 1 else 0.0,
                batch_first=True,
            )
            self.norm = nn.LayerNorm(settings.hidden_size)
            if settings.use_attention:
                self.attn_dense = nn.Linear(settings.hidden_size, 16)
                self.attn_out = nn.Linear(16, 1)
            else:
                self.attn_dense = None
                self.attn_out = None

            self.head = nn.Sequential(
                nn.Linear(settings.hidden_size, 32),
                nn.GELU(),
                nn.Dropout(settings.dropout),
                nn.Linear(32, settings.horizon),
            )
            # Direct skip / residual connection mapping recent input features to forecast horizons
            self.skip = nn.Linear(feature_count, settings.horizon, bias=False)
            nn.init.zeros_(self.skip.weight)

        def forward(self, values: Any, ticker_ids: Any) -> Any:
            embedded = self.ticker_embedding(ticker_ids)
            repeated = embedded.unsqueeze(1).expand(-1, values.shape[1], -1)
            encoded, _ = self.encoder(torch.cat((values, repeated), dim=-1))
            if self.attn_dense is not None and self.attn_out is not None:
                attn_scores = self.attn_out(torch.tanh(self.attn_dense(encoded)))
                attn_weights = torch.softmax(attn_scores, dim=1)
                context = torch.sum(encoded * attn_weights, dim=1)
                representation = self.norm(encoded[:, -1, :] + context)
            else:
                representation = self.norm(encoded[:, -1, :])
            return self.head(representation) + self.skip(values[:, -1, :])

    return PooledPriceLSTM()


def _fit(
    torch: Any,
    nn: Any,
    dataset: GlobalPriceDataset,
    settings: PriceTrainingConfig,
    train_indices: np.ndarray,
    validation_indices: np.ndarray | None,
    epochs: int,
    device_name: str = "cuda",
) -> tuple[Any, dict[str, np.ndarray], int, float]:
    feature_count = dataset.sequences.shape[-1]
    flattened = dataset.sequences[train_indices].reshape(-1, feature_count).astype(np.float64)
    feature_mean = flattened.mean(axis=0)
    feature_std = flattened.std(axis=0)
    feature_std[feature_std < 1e-8] = 1.0
    target_mean = dataset.targets[train_indices].mean(axis=0, dtype=np.float64)
    target_std = dataset.targets[train_indices].std(axis=0, dtype=np.float64)
    target_std[target_std < 1e-8] = 1.0
    scaled_x = ((dataset.sequences - feature_mean) / feature_std).astype(np.float32)
    scaled_y = ((dataset.targets - target_mean) / target_std).astype(np.float32)

    device = torch.device(device_name)
    model = _build_model(torch, nn, feature_count, len(dataset.ticker_names), settings).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs), eta_min=1e-5
    )
    loss_fn = nn.SmoothL1Loss(beta=0.5)
    best_state: dict[str, Any] | None = None
    best_loss = math.inf
    stale = 0
    completed = 0

    x = torch.as_tensor(scaled_x, device=device)
    y = torch.as_tensor(scaled_y, device=device)
    target_mean_tensor = torch.as_tensor(target_mean, device=device, dtype=torch.float32)
    target_std_tensor = torch.as_tensor(target_std, device=device, dtype=torch.float32)
    ticker_ids = torch.as_tensor(dataset.ticker_indices, dtype=torch.long, device=device)
    train_tensor = torch.as_tensor(train_indices, dtype=torch.long, device=device)
    validation_tensor = (
        torch.as_tensor(validation_indices, dtype=torch.long, device=device)
        if validation_indices is not None
        else None
    )

    for epoch in range(epochs):
        model.train()
        generator = torch.Generator(device="cpu").manual_seed(settings.seed + epoch)
        order = train_indices[torch.randperm(len(train_indices), generator=generator).numpy()]
        for start in range(0, len(order), settings.batch_size):
            batch = torch.as_tensor(
                order[start : start + settings.batch_size], dtype=torch.long, device=device
            )
            optimizer.zero_grad(set_to_none=True)
            pred = model(x[batch], ticker_ids[batch])
            loss_huber = loss_fn(pred, y[batch])
            if settings.direction_weight > 0:
                unscaled_pred = pred * target_std_tensor + target_mean_tensor
                unscaled_target = y[batch] * target_std_tensor + target_mean_tensor
                # Penalize predictions whose sign disagrees with the actual future return sign
                direction_penalty = torch.mean(
                    torch.relu(-torch.sign(unscaled_target) * (unscaled_pred / target_std_tensor))
                )
                loss = loss_huber + settings.direction_weight * direction_penalty
            else:
                loss = loss_huber
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        completed = epoch + 1
        if validation_tensor is None:
            continue
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                loss_fn(
                    model(x[validation_tensor], ticker_ids[validation_tensor]), y[validation_tensor]
                ).item()
            )
        print(
            f"epoch={completed:02d} lr={scheduler.get_last_lr()[0]:.6f} "
            f"validation_loss={validation_loss:.6f}",
            flush=True,
        )
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= settings.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    scalers = {
        "feature_mean": feature_mean.astype(np.float32),
        "feature_std": feature_std.astype(np.float32),
        "target_mean": target_mean.astype(np.float32),
        "target_std": target_std.astype(np.float32),
    }
    del optimizer, train_tensor
    return model, scalers, completed, best_loss


def _predict(
    torch: Any,
    model: Any,
    dataset: GlobalPriceDataset,
    scalers: dict[str, np.ndarray],
    indices: np.ndarray,
    device_name: str = "cuda",
) -> np.ndarray:
    device = torch.device(device_name)
    scaled = (
        (dataset.sequences[indices] - scalers["feature_mean"]) / scalers["feature_std"]
    ).astype(np.float32)
    ticker_ids = dataset.ticker_indices[indices]
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), 512):
            stop = min(start + 512, len(indices))
            raw = model(
                torch.as_tensor(scaled[start:stop], device=device),
                torch.as_tensor(ticker_ids[start:stop], dtype=torch.long, device=device),
            )
            predictions.append(raw.cpu().numpy())
    standard = np.concatenate(predictions)
    return standard * scalers["target_std"] + scalers["target_mean"]


def train_cuda_price_model(
    dataset: GlobalPriceDataset,
    output_dir: str | Path,
    config: PriceTrainingConfig | None = None,
) -> dict[str, Any]:
    """Train, test, refit, save, and return a CUDA price-model report."""
    settings = config or PriceTrainingConfig()
    try:
        import torch
        from torch import nn
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("PyTorch is required for GPU price training") from error
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to label a CPU run as GPU training")
    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    torch.cuda.manual_seed_all(settings.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    started = time.perf_counter()
    selection_model, selection_scalers, selected_epochs, best_validation_loss = _fit(
        torch,
        nn,
        dataset,
        settings,
        dataset.split_train,
        dataset.split_validation,
        settings.maximum_epochs,
        device_name="cuda",
    )
    validation_prediction = _predict(
        torch,
        selection_model,
        dataset,
        selection_scalers,
        dataset.split_validation,
        device_name="cuda",
    )
    test_prediction = _predict(
        torch, selection_model, dataset, selection_scalers, dataset.split_test, device_name="cuda"
    )
    validation_metrics = _metrics(dataset.targets[dataset.split_validation], validation_prediction)
    pooled_test_metrics = _metrics(dataset.targets[dataset.split_test], test_prediction)
    per_ticker: dict[str, Any] = {}
    for ticker_index, ticker in enumerate(dataset.ticker_names):
        mask = dataset.ticker_indices[dataset.split_test] == ticker_index
        per_ticker[ticker] = _metrics(
            dataset.targets[dataset.split_test][mask], test_prediction[mask]
        )

    final_model, final_scalers, _, _ = _fit(
        torch,
        nn,
        dataset,
        settings,
        np.arange(len(dataset.sequences), dtype=np.int64),
        None,
        max(1, selected_epochs),
        device_name="cuda",
    )
    forecasts: dict[str, Any] = {}
    target_lower = np.quantile(dataset.targets, 0.01, axis=0)
    target_upper = np.quantile(dataset.targets, 0.99, axis=0)
    final_model.eval()
    with torch.no_grad():
        for ticker_index, ticker in enumerate(dataset.ticker_names):
            values = (
                (dataset.current_sequences[ticker] - final_scalers["feature_mean"])
                / final_scalers["feature_std"]
            ).astype(np.float32)
            raw = (
                final_model(
                    torch.as_tensor(values[None, ...], device="cuda"),
                    torch.as_tensor([ticker_index], dtype=torch.long, device="cuda"),
                )
                .cpu()
                .numpy()[0]
            )
            returns = np.clip(
                raw * final_scalers["target_std"] + final_scalers["target_mean"],
                target_lower,
                target_upper,
            )
            prices = dataset.current_prices[ticker] * np.exp(returns)
            forecasts[ticker] = {
                "data_as_of": dataset.data_as_of[ticker],
                "current_price": dataset.current_prices[ticker],
                "forecast_prices": [float(value) for value in prices],
            }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "model.pt"
    torch.save(
        {
            "model_version": MODEL_VERSION,
            "state_dict": {
                key: value.detach().cpu() for key, value in final_model.state_dict().items()
            },
            "config": asdict(settings),
            "feature_names": list(dataset.feature_names),
            "feature_mode": dataset.feature_mode,
            "ticker_names": list(dataset.ticker_names),
            "scalers": {key: value.tolist() for key, value in final_scalers.items()},
            "embed_dim": 8,
        },
        checkpoint_path,
    )
    news_used = dataset.feature_mode == "price_plus_news"
    report = {
        "model_version": MODEL_VERSION,
        "status": "development_candidate",
        "feature_mode": dataset.feature_mode,
        "feature_count": len(dataset.feature_names),
        "device": str(torch.cuda.get_device_name(0)),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "config": asdict(settings),
        "rows": {
            "train": int(len(dataset.split_train)),
            "validation": int(len(dataset.split_validation)),
            "test": int(len(dataset.split_test)),
        },
        "selection": {
            "epochs": selected_epochs,
            "best_validation_scaled_huber": best_validation_loss,
            "metrics": validation_metrics,
        },
        "untouched_test": {
            "metric_source": "chronological_15_percent_test_after_validation_selection",
            "pooled": pooled_test_metrics,
            "per_ticker": per_ticker,
        },
        "final_refit": {
            "uses_all_resolved_labels_after_test_reporting": True,
            "forecasts": forecasts,
        },
        "training_seconds": time.perf_counter() - started,
        "checkpoint": checkpoint_path.name,
        "news_features_used": news_used,
        "news_reason": (
            "Timestamped Alpaca, Yahoo, and SEC EDGAR causal news features integrated (35 features)."
            if news_used
            else "Price-only baseline (25 features); news features omitted."
        ),
        "news_coverage": dataset.news_coverage if news_used else None,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    del selection_model, final_model
    torch.cuda.empty_cache()
    return report
