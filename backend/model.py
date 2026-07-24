# model.py — Builds, trains, evaluates, and runs the multi-feature LSTM model
#
# Phase 3: Walk-Forward Validation (5-fold expanding window), per-fold diagnostics,
#          cross_validation.json, validation_results.json, final model trained on 100% of data.

import hashlib
import json
import logging
import shutil
import subprocess
import sys
import threading
import time
import tracemalloc
import weakref
from pathlib import Path

import joblib  # type: ignore[import-untyped]
import numpy as np
import tensorflow as tf  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit  # type: ignore[import-untyped]
from sklearn.preprocessing import MinMaxScaler  # type: ignore[import-untyped]
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore[import-untyped]
from tensorflow.keras.layers import (  # type: ignore[import-untyped]
    LSTM,
    Attention,
    Bidirectional,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    Layer,
    LayerNormalization,
)
from tensorflow.keras.models import Model, Sequential, load_model  # type: ignore[import-untyped]

from config import (
    APP_VERSION,
    BATCH_SIZE,
    EPOCHS,
    FEATURES,
    LSTM_UNITS,
    MAX_FORECAST_DAYS,
    MODEL_DIR,
    MODEL_MAX_AGE_DAYS,
    SCHEMA_VERSION,
    VALIDATION_CONFIG,
    WINDOW_SIZE,
)
from data_pipeline import create_direction_sequences, create_sequences

logger = logging.getLogger(__name__)


# ── Git / environment helpers ────────────────────────────────────────
def _get_git_commit() -> str:
    """Return the short HEAD git commit hash, or 'unknown' on failure."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _get_model_profile(model, model_path: Path) -> dict:
    """Collect parameter count, saved file size, and peak memory usage."""

    param_count = int(model.count_params())
    trainable = int(sum(tf.keras.backend.count_params(w) for w in model.trainable_weights))
    size_bytes = model_path.stat().st_size if model_path.exists() else 0
    try:
        current, peak = tracemalloc.get_traced_memory()
        peak_mb = round(peak / 1024 / 1024, 2)
    except Exception:
        peak_mb = None
    return {
        "parameter_count": param_count,
        "trainable_parameters": trainable,
        "model_size_bytes": size_bytes,
        "peak_memory_mb": peak_mb,
    }


def _compute_residual_diagnostics(all_residuals: list[float]) -> dict:
    """Compute standard forecasting residual diagnostics from pooled fold residuals.

    Implemented in pure NumPy (no statsmodels) to avoid an extra dependency.
    Durbin-Watson:  dw = sum(diff(r)^2) / sum(r^2)
    Ljung-Box (lag 10): Q = n*(n+2) * sum(rho_k^2 / (n-k) for k=1..10)
    """
    import scipy.stats as scipy_stats  # already a transitive dep via sklearn

    r = np.array(all_residuals, dtype=float)
    n = len(r)
    if n < 8:
        return {"error": "insufficient_residuals", "n": n}

    # Durbin-Watson
    denom = float(np.sum(r**2))
    dw = float(np.sum(np.diff(r) ** 2) / denom) if denom != 0.0 else 0.0

    skew = float(scipy_stats.skew(r))
    kurt = float(scipy_stats.kurtosis(r))  # excess kurtosis
    sw_stat, sw_p = scipy_stats.shapiro(r[:5000])  # shapiro max-5000 limit

    # Ljung-Box at lag 10 (numpy implementation)
    lags = 10
    r_demeaned = r - np.mean(r)
    c0 = float(np.dot(r_demeaned, r_demeaned))
    lb_stat: float
    lb_p: float
    if c0 == 0.0 or n <= lags:
        lb_stat = 0.0
        lb_p = 1.0
    else:
        acf_vals = np.array(
            [np.dot(r_demeaned[k:], r_demeaned[: n - k]) / c0 for k in range(1, lags + 1)]
        )
        lb_stat = float(
            n * (n + 2) * np.sum(acf_vals**2 / np.array([n - k for k in range(1, lags + 1)]))
        )
        import scipy.stats as _chi2

        lb_p = float(1.0 - _chi2.chi2.cdf(lb_stat, df=lags))

    return {
        "n_residuals": n,
        "durbin_watson": round(dw, 4),
        "mean": round(float(np.mean(r)), 4),
        "std": round(float(np.std(r)), 4),
        "skewness": round(skew, 4),
        "kurtosis": round(kurt, 4),
        "shapiro_wilk_stat": round(float(sw_stat), 4),
        "shapiro_wilk_p": round(float(sw_p), 4),
        "ljung_box_stat": round(lb_stat, 4),
        "ljung_box_p": round(lb_p, 4),
        "is_normal": bool(sw_p > 0.05),
        "has_autocorrelation": bool(lb_p < 0.05),
    }


# ── Per-ticker lock ───────────────────────────────────────────────────
_training_locks: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
_locks_lock = threading.Lock()


def _get_ticker_lock(ticker: str) -> threading.Lock:
    with _locks_lock:
        lock = _training_locks.get(ticker)
        if lock is None:
            lock = threading.Lock()
            _training_locks[ticker] = lock
        return lock


# ── Unscale helper ───────────────────────────────────────────────────
def _unscale_close(scaled_values: np.ndarray, scaler) -> np.ndarray:
    """Unscale predictions using the scaler's 'Close' feature parameters."""
    close_idx = FEATURES.index("Close")
    close_scale = scaler.scale_[close_idx]
    close_min = scaler.min_[close_idx]
    return (scaled_values - close_min) / close_scale


# ── Build ────────────────────────────────────────────────────────────


@tf.keras.utils.register_keras_serializable()
class TemporalAttention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1), a


def build_lstm_model(
    forecast_days: int = MAX_FORECAST_DAYS, num_features: int = len(FEATURES)
) -> Sequential:
    model = Sequential(
        [
            Input(shape=(WINDOW_SIZE, num_features)),
            LSTM(LSTM_UNITS, return_sequences=True),
            Dropout(0.25),
            LSTM(LSTM_UNITS // 2, return_sequences=False),
            Dropout(0.25),
            Dense(32, activation="relu"),
            Dense(forecast_days),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def build_attention_lstm_model(
    forecast_days: int = MAX_FORECAST_DAYS, num_features: int = len(FEATURES)
) -> Model:
    inputs = Input(shape=(WINDOW_SIZE, num_features))
    lstm_out = LSTM(LSTM_UNITS, return_sequences=True)(inputs)
    lstm_out = Dropout(0.25)(lstm_out)
    attention_out, attention_weights = Attention()(
        [lstm_out, lstm_out], return_attention_scores=True
    )
    pooled = GlobalAveragePooling1D()(attention_out)
    predictions = Dense(forecast_days, activation="sigmoid")(pooled)
    model = Model(inputs=inputs, outputs=[predictions, attention_weights])
    model.compile(optimizer="adam", loss=["binary_crossentropy", None])
    return model


def _build_bilstm_attention_backbone(num_features: int) -> Model:
    inputs = Input(shape=(WINDOW_SIZE, num_features))
    x = LayerNormalization()(inputs)
    x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True))(x)
    x = Dropout(0.25)(x)
    x = Bidirectional(LSTM(LSTM_UNITS // 2, return_sequences=True))(x)

    context_vector, attention_weights = TemporalAttention()(x)

    x = Dense(LSTM_UNITS, activation="relu")(context_vector)
    x = Dropout(0.25)(x)

    return Model(inputs=inputs, outputs=[x, attention_weights], name="backbone")


def build_bilstm_attention_regression(
    forecast_days: int = MAX_FORECAST_DAYS, num_features: int = len(FEATURES)
) -> Model:
    backbone = _build_bilstm_attention_backbone(num_features)
    inputs = backbone.input
    x, attention_weights = backbone.output

    predictions = Dense(forecast_days)(x)
    model = Model(inputs=inputs, outputs=[predictions, attention_weights])
    model.compile(optimizer="adam", loss=["mean_squared_error", None])
    return model


def build_bilstm_attention_direction(
    forecast_days: int = MAX_FORECAST_DAYS, num_features: int = len(FEATURES)
) -> Model:
    backbone = _build_bilstm_attention_backbone(num_features)
    inputs = backbone.input
    x, attention_weights = backbone.output

    predictions = Dense(forecast_days, activation="sigmoid")(x)
    model = Model(inputs=inputs, outputs=[predictions, attention_weights])
    model.compile(optimizer="adam", loss=["binary_crossentropy", None])
    return model


def _build_model_for_type(
    model_type: str, forecast_days: int, num_features: int
) -> Model | Sequential:
    """Factory: instantiate the correct model architecture for model_type."""
    if model_type == "bilstm_attention_regression":
        return build_bilstm_attention_regression(
            forecast_days=forecast_days, num_features=num_features
        )
    elif model_type == "bilstm_attention_direction":
        return build_bilstm_attention_direction(
            forecast_days=forecast_days, num_features=num_features
        )
    elif model_type == "attention":
        return build_attention_lstm_model(forecast_days=forecast_days, num_features=num_features)
    else:
        return build_lstm_model(forecast_days=forecast_days, num_features=num_features)


def _is_direction_model(model_type: str) -> bool:
    return "direction" in model_type or model_type == "attention"


# ── Load metrics & metadata ──────────────────────────────────────────
def load_metadata(ticker: str, model_type: str = "lstm") -> dict:
    meta_path = Path(MODEL_DIR) / ticker / model_type / "metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to load metadata for %s/%s: %s", ticker, model_type, e)
    return {}


def load_metrics(ticker: str, model_type: str = "attention") -> dict:
    metrics_path = Path(MODEL_DIR) / ticker / model_type / "metrics.json"
    if metrics_path.exists():
        try:
            with open(metrics_path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to load metrics for %s/%s: %s", ticker, model_type, e)
    return {}


def load_cross_validation(ticker: str, model_type: str = "lstm") -> dict:
    """Load cross_validation.json for a ticker/model_type."""
    cv_path = Path(MODEL_DIR) / ticker / model_type / "cross_validation.json"
    if cv_path.exists():
        try:
            with open(cv_path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to load cross_validation for %s/%s: %s", ticker, model_type, e)
    return {}


def load_validation_results(ticker: str, model_type: str = "lstm") -> list:
    """Load validation_results.json (per-fold residuals) for a ticker/model_type."""
    vr_path = Path(MODEL_DIR) / ticker / model_type / "validation_results.json"
    if vr_path.exists():
        try:
            with open(vr_path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to load validation_results for %s/%s: %s", ticker, model_type, e)
    return []


# ── Walk-Forward Validation helpers ─────────────────────────────────


def _compute_fold_metrics_regression(y_true_prices: np.ndarray, y_pred_prices: np.ndarray) -> dict:
    """Compute regression metrics from unscaled price arrays."""
    rmse = float(np.sqrt(mean_squared_error(y_true_prices, y_pred_prices)))
    mae = float(mean_absolute_error(y_true_prices, y_pred_prices))
    nonzero = y_true_prices != 0
    mape = (
        float(
            np.mean(
                np.abs((y_true_prices[nonzero] - y_pred_prices[nonzero]) / y_true_prices[nonzero])
            )
            * 100
        )
        if np.any(nonzero)
        else None
    )
    r2 = float(r2_score(y_true_prices, y_pred_prices))

    # Directional accuracy on 1-step-ahead
    if len(y_true_prices) > 1:
        da = float(np.mean(np.sign(np.diff(y_true_prices)) == np.sign(np.diff(y_pred_prices))))
    else:
        da = None

    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "mape": round(mape, 4) if mape is not None else None,
        "r2": round(r2, 6),
        "direction_accuracy": round(da, 4) if da is not None else None,
    }


def _compute_fold_metrics_direction(y_true_binary: np.ndarray, y_pred_probs: np.ndarray) -> dict:
    """Compute classification metrics from binary targets and predicted probabilities."""
    y_pred_binary = (y_pred_probs > 0.5).astype(int)
    acc = float(accuracy_score(y_true_binary, y_pred_binary))
    prec = float(precision_score(y_true_binary, y_pred_binary, zero_division=0))
    rec = float(recall_score(y_true_binary, y_pred_binary, zero_division=0))
    f1 = float(f1_score(y_true_binary, y_pred_binary, zero_division=0))
    naive_baseline = float(np.mean(y_true_binary == int(np.bincount(y_true_binary).argmax())))
    return {
        "direction_accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "naive_baseline": round(naive_baseline, 4),
        "rmse": None,
        "mae": None,
        "mape": None,
        "r2": None,
    }


def _train_single_fold(
    X_train_fold: np.ndarray,
    y_train_fold: np.ndarray,
    X_val_fold: np.ndarray,
    y_val_fold: np.ndarray,
    model_type: str,
    forecast_days: int,
    num_features: int,
) -> tuple:
    """Train one fold model. Returns (model, history, training_seconds)."""
    model = _build_model_for_type(model_type, forecast_days, num_features)
    early_stop = EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)

    t0 = time.time()
    history = model.fit(
        X_train_fold,
        y_train_fold,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val_fold, y_val_fold),
        callbacks=[early_stop],
        verbose=0,
    )
    training_seconds = round(time.time() - t0, 2)
    return model, history, training_seconds


def _build_fold_residuals(
    fold_idx: int,
    val_dates: list,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scaler,
    model_type: str,
    fold_meta: dict,
) -> dict:
    """
    Build per-fold diagnostics entry with residuals, dates, actuals and predictions.
    For regression: unscale to price domain. For direction: work in probability space.
    """
    is_dir = _is_direction_model(model_type)
    residual_rows = []
    actuals_list: list[int | float] = []
    predictions_list: list[int | float] = []

    if is_dir:
        # y_true: (N, forecast_days) binary, y_pred: (N, forecast_days) probabilities
        # Report on the 1-step-ahead prediction for each sequence date
        for i, date in enumerate(val_dates):
            prob = float(y_pred[i, 0])
            actual = int(y_true[i, 0])
            predicted = int(prob > 0.5)
            residual = actual - predicted
            residual_rows.append(
                {
                    "date": date,
                    "actual": actual,
                    "predicted_prob": round(prob, 4),
                    "predicted_direction": predicted,
                    "residual": residual,
                    "absolute_error": abs(residual),
                    "percentage_error": None,
                }
            )
        actuals_list = [int(y_true[i, 0]) for i in range(len(val_dates))]
        predictions_list = [round(float(y_pred[i, 0]), 4) for i in range(len(val_dates))]
    else:
        # Regression: unscale first-step predictions to price domain
        pred_first = y_pred[:, 0]
        true_first = y_true[:, 0]
        pred_prices = _unscale_close(pred_first, scaler)
        true_prices = _unscale_close(true_first, scaler)
        for i, date in enumerate(val_dates):
            act_price = round(float(true_prices[i]), 4)
            pred_price = round(float(pred_prices[i]), 4)
            resid = round(act_price - pred_price, 4)
            abs_err = round(abs(resid), 4)
            pct_err = round(abs(resid / act_price) * 100, 4) if act_price != 0 else None
            residual_rows.append(
                {
                    "date": date,
                    "actual": act_price,
                    "predicted": pred_price,
                    "residual": resid,
                    "absolute_error": abs_err,
                    "percentage_error": pct_err,
                }
            )
        actuals_list = [round(float(true_prices[i]), 4) for i in range(len(val_dates))]
        predictions_list = [round(float(pred_prices[i]), 4) for i in range(len(val_dates))]

    return {
        "fold": fold_idx,
        **fold_meta,
        "actuals": actuals_list,
        "predictions": predictions_list,
        "residuals": residual_rows,
    }


def _run_walk_forward_validation(
    feature_values: np.ndarray,
    close_values: np.ndarray,
    log_returns: np.ndarray,
    dates,
    model_type: str,
    forecast_days: int,
) -> tuple[list, dict]:
    """
    Run TimeSeriesSplit 5-fold expanding walk-forward validation.

    Returns (fold_results_list, cv_summary_dict).
    """
    n_folds = VALIDATION_CONFIG.folds
    tss = TimeSeriesSplit(n_splits=n_folds)
    num_features = feature_values.shape[1]
    is_dir = _is_direction_model(model_type)

    # For direction models, we work with log_returns aligned to feature_values[1:]
    if is_dir:
        aligned_features = feature_values[1:]
        aligned_dates = dates[1:]
    else:
        aligned_features = feature_values
        aligned_dates = dates

    n_rows = len(aligned_features)
    indices = np.arange(n_rows)

    fold_results = []
    all_fold_metrics = []

    logger.info("Starting Walk-Forward Validation: %d folds, model_type=%s", n_folds, model_type)

    for fold_idx, (train_idx, val_idx) in enumerate(tss.split(indices), start=1):
        fold_start_t = time.time()

        # ── Per-fold scaler (fit on train partition only) ────────────
        train_features_raw = aligned_features[train_idx]
        val_features_raw = aligned_features[val_idx]
        fold_scaler = MinMaxScaler()
        fold_scaler.fit(train_features_raw)
        train_features_scaled = fold_scaler.transform(train_features_raw)
        val_features_scaled = fold_scaler.transform(val_features_raw)

        train_dates_fold = aligned_dates[train_idx]
        val_dates_fold = aligned_dates[val_idx]

        # ── Build sequences ───────────────────────────────────────────
        try:
            if is_dir:
                train_log_returns = log_returns[train_idx]
                val_log_returns = log_returns[val_idx]
                X_train_f, y_train_f, _ = create_direction_sequences(
                    train_features_scaled, train_log_returns, train_dates_fold, forecast_days
                )
                X_val_f, y_val_f, val_seq_dates = create_direction_sequences(
                    val_features_scaled, val_log_returns, val_dates_fold, forecast_days
                )
            else:
                close_idx = FEATURES.index("Close")
                X_train_f, y_train_f, _ = create_sequences(
                    train_features_scaled, train_dates_fold, close_idx, forecast_days
                )
                X_val_f, y_val_f, val_seq_dates = create_sequences(
                    val_features_scaled, val_dates_fold, close_idx, forecast_days
                )
        except ValueError as exc:
            logger.warning("Fold %d skipped: not enough data. (%s)", fold_idx, exc)
            continue

        if len(X_train_f) == 0 or len(X_val_f) == 0:
            logger.warning("Fold %d skipped: empty sequences after windowing.", fold_idx)
            continue

        # ── Train fold model ─────────────────────────────────────────
        fold_model, fold_history, fold_train_secs = _train_single_fold(
            X_train_f,
            y_train_f,
            X_val_f,
            y_val_f,
            model_type,
            forecast_days,
            num_features,
        )

        # ── Predict on validation fold ────────────────────────────────
        preds = fold_model.predict(X_val_f, verbose=0)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        # ── Compute fold metrics ──────────────────────────────────────
        if is_dir:
            metrics = _compute_fold_metrics_direction(y_val_f[:, 0], preds[:, 0])
        else:
            pred_prices = _unscale_close(preds[:, 0], fold_scaler)
            true_prices = _unscale_close(y_val_f[:, 0], fold_scaler)
            metrics = _compute_fold_metrics_regression(true_prices, pred_prices)

        fold_total_secs = round(time.time() - fold_start_t, 2)

        train_start_date = (
            str(train_dates_fold[0].date())
            if hasattr(train_dates_fold[0], "date")
            else str(train_dates_fold[0])
        )
        train_end_date = (
            str(train_dates_fold[-1].date())
            if hasattr(train_dates_fold[-1], "date")
            else str(train_dates_fold[-1])
        )
        val_start_date = (
            str(val_dates_fold[0].date())
            if hasattr(val_dates_fold[0], "date")
            else str(val_dates_fold[0])
        )
        val_end_date = (
            str(val_dates_fold[-1].date())
            if hasattr(val_dates_fold[-1], "date")
            else str(val_dates_fold[-1])
        )

        fold_meta = {
            "train_start": train_start_date,
            "train_end": train_end_date,
            "validation_start": val_start_date,
            "validation_end": val_end_date,
            "train_samples": len(X_train_f),
            "validation_samples": len(X_val_f),
            "training_seconds": fold_train_secs,
            "fold_total_seconds": fold_total_secs,
            **metrics,
        }

        # ── Build residuals ───────────────────────────────────────────
        fold_result = _build_fold_residuals(
            fold_idx=fold_idx,
            val_dates=val_seq_dates,
            y_true=y_val_f,
            y_pred=preds,
            scaler=fold_scaler,
            model_type=model_type,
            fold_meta=fold_meta,
        )
        fold_results.append(fold_result)
        all_fold_metrics.append(metrics)

        logger.info(
            "Fold %d/%d completed | train=%d val=%d | %s",
            fold_idx,
            n_folds,
            len(X_train_f),
            len(X_val_f),
            " | ".join(f"{k}={v}" for k, v in metrics.items() if v is not None),
        )

    # ── Aggregate cross-validation summary ───────────────────────────
    cv_summary = _aggregate_cv_metrics(all_fold_metrics, n_folds)

    # ── Pool residuals for diagnostics (regression only) ─────────────
    if not is_dir and fold_results:
        all_residuals: list[float] = []
        for fr in fold_results:
            for row in fr.get("residuals", []):
                v = row.get("residual")
                if v is not None:
                    all_residuals.append(float(v))
        if all_residuals:
            try:
                cv_summary["residual_diagnostics"] = _compute_residual_diagnostics(all_residuals)
            except Exception as _rd_exc:
                logger.warning("residual_diagnostics failed: %s", _rd_exc)

    return fold_results, cv_summary


def _aggregate_cv_metrics(all_fold_metrics: list, n_folds: int) -> dict:
    """Average and std of each metric across folds that have non-None values."""
    if not all_fold_metrics:
        return {"folds_completed": 0}

    metric_keys = list(all_fold_metrics[0].keys())
    summary: dict = {"folds": n_folds, "folds_completed": len(all_fold_metrics)}

    for key in metric_keys:
        vals = [m[key] for m in all_fold_metrics if m.get(key) is not None]
        if vals:
            summary[f"average_{key}"] = round(float(np.mean(vals)), 6)
            summary[f"std_{key}"] = round(float(np.std(vals)), 6)
        else:
            summary[f"average_{key}"] = None
            summary[f"std_{key}"] = None

    return summary


# ── Train ────────────────────────────────────────────────────────────
def train_model(
    X_train,
    y_train,
    X_test,
    y_test,
    ticker: str,
    scaler=None,
    model_type: str = "lstm",
    feature_df=None,
):
    """
    Phase 3 training pipeline:
    1. Walk-forward validation (5-fold expanding) to generate realistic performance metrics.
    2. Train final production model on 100% of available data.
    3. Atomically save everything: model, scaler, metadata, history, cross_validation, validation_results.

    Parameters
    ----------
    X_train, y_train    : labelled train sequences (kept for API compatibility; used for final training).
    X_test, y_test      : held-out test sequences (used for post-training eval).
    feature_df          : optional raw feature DataFrame; required for walk-forward splits.
    """
    forecast_days = y_train.shape[1]
    num_features = X_train.shape[2]

    tracemalloc.start()

    # ── Walk-Forward Validation ───────────────────────────────────────
    fold_results: list = []
    cv_summary: dict = {"note": "walk_forward_skipped_no_feature_df"}

    if feature_df is not None:
        from config import FEATURES as _FEATURES

        feature_values = feature_df[_FEATURES].values
        close_values = feature_df["Close"].values
        # compute aligned log returns
        log_returns = np.log(close_values[1:] / close_values[:-1])
        dates = feature_df.index
        try:
            fold_results, cv_summary = _run_walk_forward_validation(
                feature_values, close_values, log_returns, dates, model_type, forecast_days
            )
        except Exception:
            logger.warning(
                "Walk-forward validation failed; will still train final model.", exc_info=True
            )

    # ── Final Model: train on 100% of data ───────────────────────────
    # Combine train + test partitions for the final production model
    X_all = np.concatenate([X_train, X_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)

    final_model = _build_model_for_type(model_type, forecast_days, num_features)
    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    logger.info(
        "Training final production model for %s (%s):\n"
        "  Schema: v%d | Window: %d | Features: %d | Total Samples: %d",
        ticker,
        model_type,
        SCHEMA_VERSION,
        WINDOW_SIZE,
        num_features,
        len(X_all),
    )

    # Minimal val_split for early stopping on the final model (use last 10%)
    val_split_n = max(1, int(len(X_all) * 0.1))
    X_final_train, X_final_val = X_all[:-val_split_n], X_all[-val_split_n:]
    y_final_train, y_final_val = y_all[:-val_split_n], y_all[-val_split_n:]

    start_time = time.time()
    history = final_model.fit(
        X_final_train,
        y_final_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_final_val, y_final_val),
        callbacks=[early_stop],
        verbose=0,
    )
    training_duration_seconds = round(time.time() - start_time, 2)

    epochs_trained = len(history.history["loss"])
    best_epoch = (
        int(np.argmin(history.history["val_loss"]) + 1)
        if "val_loss" in history.history
        else epochs_trained
    )
    early_stopped = epochs_trained < EPOCHS

    # ── Dataset fingerprint ───────────────────────────────────────────
    hasher = hashlib.sha256()
    hasher.update(X_train.tobytes())
    hasher.update(y_train.tobytes())
    config_str = (
        f"FEATURES={FEATURES}|WINDOW_SIZE={WINDOW_SIZE}|"
        f"FORECAST_DAYS={forecast_days}|SCHEMA_VERSION={SCHEMA_VERSION}|MODEL_TYPE={model_type}"
    )
    hasher.update(config_str.encode("utf-8"))
    fingerprint = hasher.hexdigest()

    # ── Feature statistics ────────────────────────────────────────────
    feature_stats: dict = {}
    if scaler is not None:
        try:
            X_train_unscaled = scaler.inverse_transform(X_train.reshape(-1, num_features))
            feature_stats = {
                "mean": np.mean(X_train_unscaled, axis=0).tolist(),
                "std": np.std(X_train_unscaled, axis=0).tolist(),
                "min": np.min(X_train_unscaled, axis=0).tolist(),
                "max": np.max(X_train_unscaled, axis=0).tolist(),
                "feature_names": FEATURES,
            }
        except Exception:
            pass

    # ── Compute test-set classification or regression metrics ─────────
    test_metrics: dict = {}
    if _is_direction_model(model_type) and len(X_test) > 0:
        preds_test = final_model.predict(X_test, verbose=0)
        if isinstance(preds_test, (list, tuple)):
            preds_test = preds_test[0]
        pred_first = (preds_test[:, 0] > 0.5).astype(int)
        true_first = y_test[:, 0].astype(int)
        test_metrics = {
            "precision": float(precision_score(true_first, pred_first, zero_division=0)),
            "recall": float(recall_score(true_first, pred_first, zero_division=0)),
            "f1": float(f1_score(true_first, pred_first, zero_division=0)),
            "naive_baseline": float(
                accuracy_score(
                    true_first, np.full_like(true_first, int(np.bincount(true_first).argmax()))
                )
            )
            if len(true_first) > 0
            else 0.0,
        }

    # ── Fail-Safe Atomic Save ─────────────────────────────────────────
    base_dir = Path(MODEL_DIR) / ticker / model_type
    tmp_dir = Path(MODEL_DIR) / ticker / f"{model_type}_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        final_model.save(str(tmp_dir / "model.keras"))
        model_saved_path = tmp_dir / "model.keras"
        if scaler is not None:
            joblib.dump(scaler, str(tmp_dir / "scaler.joblib"))

        meta = {
            "schema_version": SCHEMA_VERSION,
            "model_type": model_type,
            "window_size": WINDOW_SIZE,
            "feature_count": num_features,
            "features": FEATURES,
            "dataset_fingerprint": fingerprint,
            "feature_stats": feature_stats,
            "epochs_trained": epochs_trained,
            "early_stopped": early_stopped,
            "best_epoch": best_epoch,
            "train_loss": history.history["loss"][-1] if "loss" in history.history else None,
            "val_loss": history.history["val_loss"][-1] if "val_loss" in history.history else None,
            "training_duration_seconds": training_duration_seconds,
            "scaler": "MinMaxScaler",
            "validation_method": VALIDATION_CONFIG.method,
            "validation_folds": VALIDATION_CONFIG.folds,
            "app_version": APP_VERSION,
            "tensorflow_version": tf.__version__,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "hyperparameters": {
                "optimizer": "adam",
                "learning_rate": 0.001,
                "batch_size": BATCH_SIZE,
                "epochs_max": EPOCHS,
                "epochs_trained": len(history.history["loss"]) if history else 0,
                "patience": 10,
                "dropout": 0.25,
                "lstm_units": LSTM_UNITS,
                "window_size": WINDOW_SIZE,
                "forecast_days": MAX_FORECAST_DAYS,
            },
            "environment": {
                "python_version": sys.version.split()[0],
                "tensorflow_version": tf.__version__,
                "numpy_version": np.__version__,
                "sklearn_version": __import__("sklearn").__version__,
                "git_commit": _get_git_commit(),
            },
        }
        meta["model_profile"] = _get_model_profile(final_model, model_saved_path)
        tracemalloc.stop()
        with open(tmp_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        history_data = {
            "best_val_loss": min(history.history["val_loss"])
            if "val_loss" in history.history
            else None,
            "best_epoch": best_epoch,
            "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
        }
        with open(tmp_dir / "history.json", "w") as f:
            json.dump(history_data, f, indent=2)

        if test_metrics:
            with open(tmp_dir / "metrics.json", "w") as f:
                json.dump(test_metrics, f, indent=2)

        # Walk-forward outputs
        with open(tmp_dir / "cross_validation.json", "w") as f:
            json.dump(cv_summary, f, indent=2)

        with open(tmp_dir / "validation_results.json", "w") as f:
            json.dump(fold_results, f, indent=2)

        # Atomic directory swap
        if base_dir.exists():
            shutil.rmtree(base_dir)
        tmp_dir.rename(base_dir)

        logger.info("Successfully trained and saved model → %s", base_dir)

    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        logger.error("Failed to save trained model artifacts to %s", tmp_dir, exc_info=True)
        raise

    update_manifest(ticker, model_type, {"status": "trained", "schema_version": SCHEMA_VERSION})
    return final_model, scaler


def update_manifest(ticker: str, model_type: str, metadata: dict | None = None) -> None:
    """Automatically update saved_models/manifest.json."""
    manifest_path = Path(MODEL_DIR) / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except Exception:
            manifest = {}

    key = f"{ticker}_{model_type}"
    manifest[key] = {
        "ticker": ticker,
        "model_type": model_type,
        "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "version": APP_VERSION,
        "schema_version": SCHEMA_VERSION,
        "metadata": metadata or {},
    }

    try:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
    except Exception:
        logger.warning("Failed to update manifest.json", exc_info=True)


def get_manifest() -> dict:
    """Return saved_models/manifest.json contents."""
    manifest_path = Path(MODEL_DIR) / "manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


# ── Staleness & Schema Validation ────────────────────────────────────
def _is_stale(path: Path) -> bool:
    if not path.exists():
        return True
    age_days = (time.time() - path.stat().st_mtime) / 86400
    return age_days > MODEL_MAX_AGE_DAYS


def is_schema_valid(ticker: str, model_type: str) -> bool:
    """Check if saved metadata matches current SCHEMA_VERSION and FEATURES list."""
    meta = load_metadata(ticker, model_type)
    if not meta:
        return False
    return (
        meta.get("schema_version") == SCHEMA_VERSION
        and meta.get("features") == FEATURES
        and meta.get("window_size") == WINDOW_SIZE
    )


# ── Load or train ────────────────────────────────────────────────────
def load_or_train(
    ticker: str,
    X_train,
    y_train,
    X_test,
    y_test,
    scaler=None,
    model_type: str = "lstm",
    feature_df=None,
):
    """Load cached model & scaler or retrain with fail-safe fallback."""
    base_dir = Path(MODEL_DIR) / ticker / model_type
    model_path = base_dir / "model.keras"
    scaler_path = base_dir / "scaler.joblib"
    lock = _get_ticker_lock(ticker)

    with lock:
        if (
            model_path.exists()
            and scaler_path.exists()
            and not _is_stale(model_path)
            and is_schema_valid(ticker, model_type)
        ):
            try:
                model = load_model(
                    str(model_path), custom_objects={"TemporalAttention": TemporalAttention}
                )
                loaded_scaler = joblib.load(str(scaler_path))
                logger.info("Loaded valid cached model (%s/%s)", ticker, model_type)
                return model, loaded_scaler
            except Exception:
                logger.warning(
                    "Failed to load cached model for %s/%s. Retraining...",
                    ticker,
                    model_type,
                    exc_info=True,
                )

        # Retrain with fail-safe behavior
        try:
            return train_model(
                X_train,
                y_train,
                X_test,
                y_test,
                ticker,
                scaler=scaler,
                model_type=model_type,
                feature_df=feature_df,
            )
        except Exception:
            if model_path.exists() and scaler_path.exists():
                logger.warning(
                    "Retraining failed. Falling back to existing cached model for %s/%s",
                    ticker,
                    model_type,
                )
                model = load_model(
                    str(model_path), custom_objects={"TemporalAttention": TemporalAttention}
                )
                loaded_scaler = joblib.load(str(scaler_path))
                return model, loaded_scaler
            raise


# ── Evaluate ─────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, scaler):
    """
    RMSE, MAE, MAPE, R², and directional accuracy on the test set in original price scale.
    """
    empty = {
        "rmse": None,
        "mae": None,
        "mape": None,
        "r2": None,
        "directional_accuracy": None,
    }
    if len(X_test) == 0 or len(y_test) == 0:
        return empty

    preds = model.predict(X_test, verbose=0)
    if isinstance(preds, (list, tuple)):
        preds = preds[0]

    pred_first = preds[:, 0]
    true_first = y_test[:, 0]

    pred_prices = _unscale_close(pred_first, scaler)
    true_prices = _unscale_close(true_first, scaler)

    rmse = float(np.sqrt(mean_squared_error(true_prices, pred_prices)))
    mae = float(mean_absolute_error(true_prices, pred_prices))

    nonzero = true_prices != 0
    mape = (
        float(
            np.mean(np.abs((true_prices[nonzero] - pred_prices[nonzero]) / true_prices[nonzero]))
            * 100
        )
        if np.any(nonzero)
        else None
    )

    r2 = float(r2_score(true_prices, pred_prices))

    da = None
    if preds.shape[1] > 1:
        pred_last = preds[:, -1]
        true_last = y_test[:, -1]

        pred_last_unscaled = _unscale_close(pred_last, scaler)
        true_last_unscaled = _unscale_close(true_last, scaler)

        dirs_true = np.sign(true_last_unscaled - true_prices)
        dirs_pred = np.sign(pred_last_unscaled - pred_prices)
        da = float(np.mean(dirs_true == dirs_pred))
    elif len(true_prices) > 1:
        dirs_true = np.sign(np.diff(true_prices))
        dirs_pred = np.sign(np.diff(pred_prices))
        da = float(np.mean(dirs_true == dirs_pred))

    return {
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "mape": round(mape, 2) if mape is not None else None,
        "r2": round(r2, 4),
        "directional_accuracy": round(da, 4) if da is not None else None,
    }


# ── Predict ──────────────────────────────────────────────────────────
def predict_future(model, feature_df, scaler, days: int = 7):
    """
    Direct multi-step price prediction using the last multi-feature sequence.
    """
    feature_values = feature_df[FEATURES].values
    last_window = feature_values[-WINDOW_SIZE:]
    scaled_window = scaler.transform(last_window)
    input_seq = scaled_window.reshape(1, WINDOW_SIZE, len(FEATURES))

    preds = model.predict(input_seq, verbose=0)
    preds_scaled = preds[0][0] if isinstance(preds, (list, tuple)) else preds[0]
    preds_scaled = preds_scaled[:days]

    return _unscale_close(preds_scaled, scaler).tolist()


def predict_direction(model, feature_df, scaler, days: int = 7):
    """
    Directional multi-step prediction using the Attention model.
    """
    feature_values = feature_df[FEATURES].values
    last_window = feature_values[-WINDOW_SIZE:]
    scaled_window = scaler.transform(last_window)
    input_seq = scaled_window.reshape(1, WINDOW_SIZE, len(FEATURES))

    probabilities, attention_weights = model.predict(input_seq, verbose=0)
    probabilities = probabilities[0][:days]

    probs_list = [float(p) for p in probabilities]
    directions = ["Up" if p > 0.5 else "Down" for p in probs_list]

    return directions, probs_list, attention_weights.flatten().tolist()
