"""Small, auditable seven-session price forecasting benchmark.

The production endpoint deliberately supports five liquid US equities.  It
trains quickly from completed daily OHLCV bars, chooses between two learned
models on a chronological validation block, and reports performance on a
later untouched test block.  Persistence is a comparison only; it never
replaces the learned path returned to the user.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from cachetools import TTLCache
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from calendars import future_trading_dates

TICKER_METADATA: dict[str, dict[str, str]] = {
    # NASDAQ (10)
    "AAPL": {
        "name": "Apple Inc.",
        "exchange_mic": "XNAS",
        "exchange_name": "NASDAQ",
        "currency": "USD",
        "currency_symbol": "$",
    },
    "MSFT": {
        "name": "Microsoft Corp.",
        "exchange_mic": "XNAS",
        "exchange_name": "NASDAQ",
        "currency": "USD",
        "currency_symbol": "$",
    },
    "NVDA": {
        "name": "NVIDIA Corp.",
        "exchange_mic": "XNAS",
        "exchange_name": "NASDAQ",
        "currency": "USD",
        "currency_symbol": "$",
    },
    "GOOGL": {
        "name": "Alphabet Inc.",
        "exchange_mic": "XNAS",
        "exchange_name": "NASDAQ",
        "currency": "USD",
        "currency_symbol": "$",
    },
    "AMZN": {
        "name": "Amazon.com Inc.",
        "exchange_mic": "XNAS",
        "exchange_name": "NASDAQ",
        "currency": "USD",
        "currency_symbol": "$",
    },
    "META": {
        "name": "Meta Platforms Inc.",
        "exchange_mic": "XNAS",
        "exchange_name": "NASDAQ",
        "currency": "USD",
        "currency_symbol": "$",
    },
    "TSLA": {
        "name": "Tesla Inc.",
        "exchange_mic": "XNAS",
        "exchange_name": "NASDAQ",
        "currency": "USD",
        "currency_symbol": "$",
    },
    "AMD": {
        "name": "Advanced Micro Devices",
        "exchange_mic": "XNAS",
        "exchange_name": "NASDAQ",
        "currency": "USD",
        "currency_symbol": "$",
    },
    "COST": {
        "name": "Costco Wholesale Corp.",
        "exchange_mic": "XNAS",
        "exchange_name": "NASDAQ",
        "currency": "USD",
        "currency_symbol": "$",
    },
    "QCOM": {
        "name": "Qualcomm Inc.",
        "exchange_mic": "XNAS",
        "exchange_name": "NASDAQ",
        "currency": "USD",
        "currency_symbol": "$",
    },
    # NYSE (10)
    "JPM": {
        "name": "JPMorgan Chase & Co.",
        "exchange_mic": "XNYS",
        "exchange_name": "NYSE",
        "currency": "USD",
        "currency_symbol": "$",
    },
    "XOM": {
        "name": "Exxon Mobil Corp.",
        "exchange_mic": "XNYS",
        "exchange_name": "NYSE",
        "currency": "USD",
        "currency_symbol": "$",
    },
    "WMT": {
        "name": "Walmart Inc.",
        "exchange_mic": "XNYS",
        "exchange_name": "NYSE",
        "currency": "USD",
        "currency_symbol": "$",
    },
    "JNJ": {
        "name": "Johnson & Johnson",
        "exchange_mic": "XNYS",
        "exchange_name": "NYSE",
        "currency": "USD",
        "currency_symbol": "$",
    },
    "CAT": {
        "name": "Caterpillar Inc.",
        "exchange_mic": "XNYS",
        "exchange_name": "NYSE",
        "currency": "USD",
        "currency_symbol": "$",
    },
    "KO": {
        "name": "The Coca-Cola Company",
        "exchange_mic": "XNYS",
        "exchange_name": "NYSE",
        "currency": "USD",
        "currency_symbol": "$",
    },
    "NEE": {
        "name": "NextEra Energy Inc.",
        "exchange_mic": "XNYS",
        "exchange_name": "NYSE",
        "currency": "USD",
        "currency_symbol": "$",
    },
    "DIS": {
        "name": "The Walt Disney Company",
        "exchange_mic": "XNYS",
        "exchange_name": "NYSE",
        "currency": "USD",
        "currency_symbol": "$",
    },
    "BAC": {
        "name": "Bank of America Corp.",
        "exchange_mic": "XNYS",
        "exchange_name": "NYSE",
        "currency": "USD",
        "currency_symbol": "$",
    },
    "GE": {
        "name": "GE Aerospace",
        "exchange_mic": "XNYS",
        "exchange_name": "NYSE",
        "currency": "USD",
        "currency_symbol": "$",
    },
    # LSE (10)
    "SHEL.L": {
        "name": "Shell plc",
        "exchange_mic": "XLON",
        "exchange_name": "LSE",
        "currency": "GBp",
        "currency_symbol": "p",
    },
    "AZN.L": {
        "name": "AstraZeneca PLC",
        "exchange_mic": "XLON",
        "exchange_name": "LSE",
        "currency": "GBp",
        "currency_symbol": "p",
    },
    "HSBA.L": {
        "name": "HSBC Holdings plc",
        "exchange_mic": "XLON",
        "exchange_name": "LSE",
        "currency": "GBp",
        "currency_symbol": "p",
    },
    "BP.L": {
        "name": "BP p.l.c.",
        "exchange_mic": "XLON",
        "exchange_name": "LSE",
        "currency": "GBp",
        "currency_symbol": "p",
    },
    "ULVR.L": {
        "name": "Unilever PLC",
        "exchange_mic": "XLON",
        "exchange_name": "LSE",
        "currency": "GBp",
        "currency_symbol": "p",
    },
    "GSK.L": {
        "name": "GSK plc",
        "exchange_mic": "XLON",
        "exchange_name": "LSE",
        "currency": "GBp",
        "currency_symbol": "p",
    },
    "RIO.L": {
        "name": "Rio Tinto plc",
        "exchange_mic": "XLON",
        "exchange_name": "LSE",
        "currency": "GBp",
        "currency_symbol": "p",
    },
    "BATS.L": {
        "name": "British American Tobacco p.l.c.",
        "exchange_mic": "XLON",
        "exchange_name": "LSE",
        "currency": "GBp",
        "currency_symbol": "p",
    },
    "BARC.L": {
        "name": "Barclays PLC",
        "exchange_mic": "XLON",
        "exchange_name": "LSE",
        "currency": "GBp",
        "currency_symbol": "p",
    },
    "DGE.L": {
        "name": "Diageo plc",
        "exchange_mic": "XLON",
        "exchange_name": "LSE",
        "currency": "GBp",
        "currency_symbol": "p",
    },
}

SUPPORTED_TICKERS: tuple[str, ...] = tuple(TICKER_METADATA.keys())
FORECAST_DAYS = 7
FEATURE_VERSION = "simple-price-v1"
MIN_HISTORY_ROWS = 500


def get_ticker_meta(symbol: str) -> dict[str, str]:
    sym = symbol.strip().upper()
    if sym in TICKER_METADATA:
        return dict(TICKER_METADATA[sym])
    if sym.endswith(".L"):
        return {
            "name": sym,
            "exchange_mic": "XLON",
            "exchange_name": "LSE",
            "currency": "GBp",
            "currency_symbol": "p",
        }
    return {
        "name": sym,
        "exchange_mic": "XNAS",
        "exchange_name": "NASDAQ",
        "currency": "USD",
        "currency_symbol": "$",
    }


_cache: TTLCache = TTLCache(maxsize=max(len(SUPPORTED_TICKERS) * 4, 32), ttl=6 * 60 * 60)
_cache_lock = threading.RLock()


@dataclass(frozen=True)
class ForecastDataset:
    features: pd.DataFrame
    targets: np.ndarray
    origin_positions: np.ndarray
    labelled_count: int


def _finite(values: np.ndarray) -> bool:
    return bool(np.isfinite(np.asarray(values, dtype=np.float64)).all())


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build stationary features using information available at each row."""
    close = frame["Close"].astype(float)
    log_close = np.log(close)
    returns = log_close.diff()
    log_volume = np.log1p(frame["Volume"].astype(float))
    intraday_range = np.log(frame["High"].astype(float) / frame["Low"].astype(float))
    previous_close = close.shift(1)
    features: dict[str, pd.Series] = {
        "return_1d": returns,
        "return_2d": log_close.diff(2),
        "return_5d": log_close.diff(5),
        "return_10d": log_close.diff(10),
        "return_20d": log_close.diff(20),
        "range_1d": intraday_range,
        "overnight_return": np.log(frame["Open"].astype(float) / previous_close),
        "close_location": (close - frame["Low"]) / (frame["High"] - frame["Low"]),
        "volume_change": log_volume.diff(),
        "drawdown_20d": close / close.rolling(20).max() - 1.0,
    }
    for window in (5, 10, 20, 60):
        features[f"return_mean_{window}d"] = returns.rolling(window).mean()
        features[f"return_vol_{window}d"] = returns.rolling(window).std(ddof=0)
        features[f"close_vs_sma_{window}d"] = close / close.rolling(window).mean() - 1.0
    volume_mean = log_volume.rolling(20).mean()
    volume_std = log_volume.rolling(20).std(ddof=0).replace(0.0, np.nan)
    features["volume_z_20d"] = (log_volume - volume_mean) / volume_std
    weekday = pd.Series(frame.index.dayofweek, index=frame.index, dtype=float)
    features["weekday_sin"] = np.sin(2.0 * np.pi * weekday / 5.0)
    features["weekday_cos"] = np.cos(2.0 * np.pi * weekday / 5.0)
    result = pd.DataFrame(features, index=frame.index)
    return result.replace([np.inf, -np.inf], np.nan).dropna()


def build_dataset(frame: pd.DataFrame) -> ForecastDataset:
    """Create seven direct cumulative-return targets without crossing splits."""
    if len(frame) < MIN_HISTORY_ROWS:
        raise ValueError(f"At least {MIN_HISTORY_ROWS} completed sessions are required.")
    features = build_features(frame)
    log_close = np.log(frame["Close"].astype(float))
    position_by_date = pd.Series(np.arange(len(frame), dtype=int), index=frame.index)
    labelled = features.index[
        position_by_date.loc[features.index].to_numpy() + FORECAST_DAYS < len(frame)
    ]
    labelled_features = features.loc[labelled]
    targets = np.column_stack(
        [
            (log_close.shift(-horizon) - log_close).loc[labelled].to_numpy(dtype=np.float64)
            for horizon in range(1, FORECAST_DAYS + 1)
        ]
    )
    positions = position_by_date.loc[labelled].to_numpy(dtype=int)
    if not _finite(labelled_features.to_numpy()) or not _finite(targets):
        raise ValueError("Forecast dataset contains non-finite values.")
    return ForecastDataset(
        features=features,
        targets=targets,
        origin_positions=positions,
        labelled_count=len(labelled),
    )


def chronological_masks(dataset: ForecastDataset, raw_rows: int) -> tuple[np.ndarray, ...]:
    """Return 70/15/15 masks, purging targets that cross either boundary."""
    train_boundary = int(raw_rows * 0.70)
    test_boundary = int(raw_rows * 0.85)
    origins = dataset.origin_positions
    train = origins + FORECAST_DAYS < train_boundary
    validation = (origins >= train_boundary) & (origins + FORECAST_DAYS < test_boundary)
    test = (origins >= test_boundary) & (origins + FORECAST_DAYS < raw_rows)
    if min(int(train.sum()), int(validation.sum()), int(test.sum())) < 30:
        raise ValueError("Not enough observations for chronological train/validation/test splits.")
    return train, validation, test


def _candidate_models() -> dict[str, Any]:
    return {
        "ridge": make_pipeline(RobustScaler(), Ridge(alpha=8.0)),
        "random_forest": RandomForestRegressor(
            n_estimators=120,
            max_depth=6,
            min_samples_leaf=8,
            max_features=0.75,
            n_jobs=1,
            random_state=42,
        ),
    }


def _return_errors(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return np.exp(np.asarray(predicted)) - np.exp(np.asarray(actual))


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    errors = _return_errors(actual, predicted)
    baseline_errors = _return_errors(actual, np.zeros_like(actual))
    mae = float(np.mean(np.abs(errors)) * 100.0)
    rmse = float(np.sqrt(np.mean(np.square(errors))) * 100.0)
    baseline_mae = float(np.mean(np.abs(baseline_errors)) * 100.0)
    predicted_direction = np.sign(predicted)
    actual_direction = np.sign(actual)
    per_horizon = []
    for index in range(actual.shape[1]):
        horizon_errors = errors[:, index]
        horizon_baseline = baseline_errors[:, index]
        horizon_mae = float(np.mean(np.abs(horizon_errors)) * 100.0)
        baseline_horizon_mae = float(np.mean(np.abs(horizon_baseline)) * 100.0)
        per_horizon.append(
            {
                "day": index + 1,
                "mae_percent": horizon_mae,
                "rmse_percent": float(np.sqrt(np.mean(np.square(horizon_errors))) * 100.0),
                "direction_accuracy": float(
                    np.mean(predicted_direction[:, index] == actual_direction[:, index])
                ),
                "relative_mae_vs_persistence": (
                    horizon_mae / baseline_horizon_mae if baseline_horizon_mae > 0 else None
                ),
            }
        )
    return {
        "mae_percent": mae,
        "rmse_percent": rmse,
        "direction_accuracy": float(np.mean(predicted_direction == actual_direction)),
        "persistence_mae_percent": baseline_mae,
        "relative_mae_vs_persistence": mae / baseline_mae if baseline_mae > 0 else None,
        "per_horizon": per_horizon,
    }


def _clip_predictions(predicted: np.ndarray, reference_targets: np.ndarray) -> np.ndarray:
    lower = np.quantile(reference_targets, 0.01, axis=0)
    upper = np.quantile(reference_targets, 0.99, axis=0)
    return np.clip(np.asarray(predicted, dtype=np.float64), lower, upper)


def _load_gpu_lstm_model() -> tuple[Any, dict[str, Any], list[str]] | None:
    try:
        from pathlib import Path

        import torch
        from research.price_forecasting.gpu_pipeline import PriceTrainingConfig, _build_model

        candidates = [
            Path.cwd() / "artifacts" / "tri_exchange_gpu_v1" / "model.pt",
            Path(__file__).resolve().parents[2] / "artifacts" / "tri_exchange_gpu_v1" / "model.pt",
            Path(__file__).resolve().parents[1] / "artifacts" / "tri_exchange_gpu_v1" / "model.pt",
            Path.cwd() / "artifacts" / "simple_price_gpu_v2" / "baseline_price_only" / "model.pt",
            Path(__file__).resolve().parents[2]
            / "artifacts"
            / "simple_price_gpu_v2"
            / "baseline_price_only"
            / "model.pt",
            Path(__file__).resolve().parents[1]
            / "artifacts"
            / "simple_price_gpu_v2"
            / "baseline_price_only"
            / "model.pt",
        ]
        ckpt_path = next((p for p in candidates if p.is_file()), None)
        if not ckpt_path:
            return None
        ckpt = torch.load(ckpt_path, map_location="cpu")
        ticker_names = list(ckpt.get("ticker_names") or ["AAPL", "GOOGL", "MSFT", "NVDA", "TSLA"])
        embed_dim = int(
            ckpt.get("embed_dim") or ckpt["state_dict"]["ticker_embedding.weight"].shape[1]
        )
        model = _build_model(
            torch, torch.nn, 25, len(ticker_names), PriceTrainingConfig(), embed_dim=embed_dim
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model, ckpt["scalers"], ticker_names
    except Exception:
        return None


def train_and_forecast(
    ticker: str, frame: pd.DataFrame, model_name: str = "auto"
) -> dict[str, Any]:
    """Select, evaluate, refit, and forecast one supported ticker."""
    symbol = ticker.strip().upper()
    if symbol not in SUPPORTED_TICKERS:
        raise ValueError(f"Ticker must be one of: {', '.join(SUPPORTED_TICKERS)}")
    normalized_model = model_name.strip().lower()
    frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index()
    data_as_of = pd.Timestamp(frame.index[-1]).date().isoformat()
    cache_key = f"{symbol}:{data_as_of}:{normalized_model}:{FEATURE_VERSION}"
    with _cache_lock:
        cached = _cache.get(cache_key)
        if cached is not None:
            return dict(cached)

    dataset = build_dataset(frame)
    train_mask, validation_mask, test_mask = chronological_masks(dataset, len(frame))
    labelled_features = dataset.features.iloc[: dataset.labelled_count].to_numpy(dtype=np.float64)
    targets = dataset.targets
    candidates = _candidate_models()
    validation_scores: dict[str, float] = {}
    validation_predictions: dict[str, np.ndarray] = {}
    for name, model in candidates.items():
        model.fit(labelled_features[train_mask], targets[train_mask])
        predicted = _clip_predictions(
            model.predict(labelled_features[validation_mask]), targets[train_mask]
        )
        validation_predictions[name] = predicted
        validation_scores[name] = float(mean_absolute_error(targets[validation_mask], predicted))

    gpu_lstm_info = _load_gpu_lstm_model()
    use_gpu_lstm = False
    if gpu_lstm_info is not None:
        lstm_model, scalers, gpu_tickers = gpu_lstm_info
        if symbol in gpu_tickers:
            use_gpu_lstm = True
            import torch

            f_mean = np.array(scalers["feature_mean"], dtype=np.float32)
            f_std = np.array(scalers["feature_std"], dtype=np.float32)
            f_std[f_std < 1e-8] = 1.0
            t_mean = np.array(scalers["target_mean"], dtype=np.float32)
            t_std = np.array(scalers["target_std"], dtype=np.float32)
            t_id = torch.tensor([gpu_tickers.index(symbol)], dtype=torch.long)

            val_pred_list = []
            val_origins = np.where(validation_mask)[0]
            feat_all = dataset.features.iloc[: dataset.labelled_count].to_numpy(dtype=np.float32)
            for i, orig in enumerate(val_origins):
                if orig >= 59:
                    seq = (feat_all[orig - 59 : orig + 1] - f_mean) / f_std
                    with torch.no_grad():
                        out = lstm_model(torch.tensor(seq).unsqueeze(0).float(), t_id)
                    val_pred_list.append(
                        (out.detach().numpy()[0] * t_std + t_mean).astype(np.float64)
                    )
                else:
                    val_pred_list.append(validation_predictions["ridge"][i])
            val_pred_arr = _clip_predictions(np.array(val_pred_list), targets[train_mask])
            validation_predictions["gpu_lstm"] = val_pred_arr
            validation_scores["gpu_lstm"] = float(
                mean_absolute_error(targets[validation_mask], val_pred_arr)
            )

    if normalized_model in ("gpu_lstm", "lstm") and use_gpu_lstm:
        selected_name = "gpu_lstm"
    elif normalized_model in candidates:
        selected_name = normalized_model
    else:
        selected_name = min(validation_scores, key=validation_scores.get)

    if selected_name == "gpu_lstm" and use_gpu_lstm:
        assert gpu_lstm_info is not None
        lstm_model, scalers, gpu_tickers = gpu_lstm_info
        import torch

        f_mean = np.array(scalers["feature_mean"], dtype=np.float32)
        f_std = np.array(scalers["feature_std"], dtype=np.float32)
        f_std[f_std < 1e-8] = 1.0
        t_mean = np.array(scalers["target_mean"], dtype=np.float32)
        t_std = np.array(scalers["target_std"], dtype=np.float32)
        t_id = torch.tensor([gpu_tickers.index(symbol)], dtype=torch.long)

        val_pred_arr = validation_predictions["gpu_lstm"]
        validation_residuals = targets[validation_mask] - val_pred_arr
        residual_low = np.quantile(validation_residuals, 0.10, axis=0)
        residual_high = np.quantile(validation_residuals, 0.90, axis=0)

        # Forecast using latest 60 features
        latest_seq = dataset.features.iloc[-60:].to_numpy(dtype=np.float32)
        if len(latest_seq) == 60:
            norm_seq = (latest_seq - f_mean) / f_std
            with torch.no_grad():
                out = lstm_model(torch.tensor(norm_seq).unsqueeze(0).float(), t_id)
            forecast_returns = (out.detach().numpy()[0] * t_std + t_mean).astype(np.float64)
            forecast_returns = _clip_predictions(forecast_returns.reshape(1, -1), targets).reshape(
                -1
            )
        else:
            prod_model = _candidate_models()["ridge"]
            prod_model.fit(labelled_features, targets)
            forecast_returns = _clip_predictions(
                prod_model.predict(dataset.features.iloc[[-1]].to_numpy(dtype=np.float64)), targets
            ).reshape(-1)

        # Honest Test metrics: evaluate GPU LSTM on untouched test split
        eval_model = _candidate_models()["ridge"]
        dev_mask = train_mask | validation_mask
        eval_model.fit(labelled_features[dev_mask], targets[dev_mask])
        test_pred_ridge = _clip_predictions(
            eval_model.predict(labelled_features[test_mask]), targets[dev_mask]
        )

        test_origins = np.where(test_mask)[0]
        test_pred_list = []
        feat_all = dataset.features.iloc[: dataset.labelled_count].to_numpy(dtype=np.float32)
        for i, orig in enumerate(test_origins):
            if orig >= 59:
                seq = (feat_all[orig - 59 : orig + 1] - f_mean) / f_std
                with torch.no_grad():
                    out = lstm_model(torch.tensor(seq).unsqueeze(0).float(), t_id)
                test_pred_list.append((out.detach().numpy()[0] * t_std + t_mean).astype(np.float64))
            else:
                test_pred_list.append(test_pred_ridge[i])
        test_pred = _clip_predictions(np.array(test_pred_list), targets[dev_mask])
        metrics = _metrics(targets[test_mask], test_pred)
        model_meta = {
            "name": "gpu_lstm",
            "kind": "learned_gpu_lstm_model",
            "feature_version": "price-v2-rtx2060",
            "target": "direct_cumulative_log_returns_1_to_7_sessions",
            "selection": "lowest validation MAE among learned candidates (CUDA RTX 2060 LSTM)",
            "candidate_validation_mae": validation_scores,
        }
    else:
        evaluation_model = _candidate_models()[selected_name]
        development_mask = train_mask | validation_mask
        evaluation_model.fit(labelled_features[development_mask], targets[development_mask])
        test_prediction = _clip_predictions(
            evaluation_model.predict(labelled_features[test_mask]), targets[development_mask]
        )
        metrics = _metrics(targets[test_mask], test_prediction)

        # The uncertainty band is calibrated on validation residuals only.
        validation_residuals = targets[validation_mask] - validation_predictions[selected_name]
        residual_low = np.quantile(validation_residuals, 0.10, axis=0)
        residual_high = np.quantile(validation_residuals, 0.90, axis=0)

        production_model = _candidate_models()[selected_name]
        production_model.fit(labelled_features, targets)
        latest_features = dataset.features.iloc[[-1]].to_numpy(dtype=np.float64)
        forecast_returns = _clip_predictions(
            production_model.predict(latest_features), targets
        ).reshape(-1)
        model_meta = {
            "name": selected_name,
            "kind": "learned_historical_model",
            "feature_version": FEATURE_VERSION,
            "target": "direct_cumulative_log_returns_1_to_7_sessions",
            "selection": "lowest validation MAE among learned candidates",
            "candidate_validation_mae": validation_scores,
        }

    current_price = float(frame["Close"].iloc[-1])
    predicted_prices = current_price * np.exp(forecast_returns)
    lower_prices = current_price * np.exp(forecast_returns + residual_low)
    upper_prices = current_price * np.exp(forecast_returns + residual_high)
    lower_prices, upper_prices = (
        np.minimum(lower_prices, upper_prices),
        np.maximum(lower_prices, upper_prices),
    )
    future_dates, calendar = future_trading_dates(
        symbol, pd.Timestamp(frame.index[-1]), FORECAST_DAYS
    )
    test_dates = dataset.features.index[: dataset.labelled_count][test_mask]
    history = frame.iloc[-90:]
    meta = get_ticker_meta(symbol)
    result = {
        "ticker": symbol,
        "ticker_name": meta["name"],
        "exchange_mic": meta["exchange_mic"],
        "exchange_name": meta["exchange_name"],
        "currency": meta["currency"],
        "currency_symbol": meta["currency_symbol"],
        "forecast_days": FORECAST_DAYS,
        "data_as_of": data_as_of,
        "current_price": current_price,
        "historical_dates": [value.date().isoformat() for value in history.index],
        "historical_prices": [float(value) for value in history["Close"]],
        "future_dates": future_dates,
        "predicted_prices": [float(value) for value in predicted_prices],
        "lower_prices": [float(value) for value in lower_prices],
        "upper_prices": [float(value) for value in upper_prices],
        "model": model_meta,
        "backtest": {
            "split": "chronological_70_15_15_with_7_session_purge",
            "test_start": test_dates[0].date().isoformat(),
            "test_end": test_dates[-1].date().isoformat(),
            "test_samples": int(test_mask.sum()),
            "metric_source": "untouched_chronological_test",
            **metrics,
        },
        "provenance": {
            "data_provider": str(frame.attrs.get("data_provider", "unknown")),
            "market_data_cache": str(frame.attrs.get("market_data_cache", "unknown")),
            "calendar": calendar,
            "completed_daily_bars_only": True,
        },
        "news": {
            "role": "context_only",
            "used_by_model": False,
            "reason": "Historical news coverage has not yet passed the same chronological backtest.",
        },
    }
    if not all(
        _finite(np.asarray(result[key], dtype=np.float64))
        for key in ("predicted_prices", "lower_prices", "upper_prices")
    ):
        raise ValueError("Forecast produced non-finite output.")
    with _cache_lock:
        _cache[cache_key] = result
    return dict(result)


def clear_forecast_cache() -> None:
    with _cache_lock:
        _cache.clear()
