# model.py — Builds, trains, evaluates, and runs the multi-feature LSTM model
#
# Phase 3: Walk-Forward Validation (5-fold expanding window), per-fold diagnostics,
#          cross_validation.json, validation_results.json, final model trained on 100% of data.

import hashlib
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import tracemalloc
import uuid
import weakref
from contextlib import contextmanager
from pathlib import Path

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
from sklearn.preprocessing import MinMaxScaler  # type: ignore[import-untyped]
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore[import-untyped]
from tensorflow.keras.layers import (  # type: ignore[import-untyped]
    GRU,
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
    settings,
)
from evaluation.metrics import evaluate_forecast_horizons, evaluate_probability_forecast
from evaluation.splits import generate_walk_forward_splits, purged_tail_split

logger = logging.getLogger(__name__)


class ArtifactValidationError(RuntimeError):
    """A cached artifact failed integrity or compatibility validation."""


class TrainingCapacityError(RuntimeError):
    """The bounded training pool could not accept more work."""


_training_slots = threading.BoundedSemaphore(settings.training_concurrency)


def set_reproducibility(seed: int | None = None) -> None:
    """Apply deterministic seeds before every model construction/training run."""
    chosen = VALIDATION_CONFIG.seed if seed is None else seed
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1" if VALIDATION_CONFIG.deterministic else "0")
    random.seed(chosen)
    np.random.seed(chosen)
    tf.keras.utils.set_random_seed(chosen)
    if VALIDATION_CONFIG.deterministic:
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            logger.info("TensorFlow deterministic operations are unavailable")


def generate_validation_splits(
    n_rows: int, config=VALIDATION_CONFIG
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate deterministic expanding or fixed-window rolling splits."""
    return generate_walk_forward_splits(
        n_rows,
        folds=config.folds,
        min_train_size=config.min_train_size,
        validation_size=config.horizon,
        gap=config.gap,
        method=config.method,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_scaler_json(scaler: MinMaxScaler, path: Path) -> None:
    payload = {
        "format": "sklearn_minmax_v1",
        "feature_range": list(scaler.feature_range),
        "n_features_in": int(scaler.n_features_in_),
        "data_min": scaler.data_min_.tolist(),
        "data_max": scaler.data_max_.tolist(),
        "data_range": scaler.data_range_.tolist(),
        "scale": scaler.scale_.tolist(),
        "min": scaler.min_.tolist(),
        "n_samples_seen": int(scaler.n_samples_seen_),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_scaler_json(path: Path) -> MinMaxScaler:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != "sklearn_minmax_v1":
            raise ArtifactValidationError("Unsupported scaler format.")
        count = int(payload["n_features_in"])
        names = ("data_min", "data_max", "data_range", "scale", "min")
        arrays = {name: np.asarray(payload[name], dtype=float) for name in names}
        feature_range = tuple(float(value) for value in payload["feature_range"])
        samples_seen = int(payload["n_samples_seen"])
    except ArtifactValidationError:
        raise
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError("Scaler metadata is malformed.") from exc
    if (
        count != len(FEATURES)
        or len(feature_range) != 2
        or not np.isfinite(feature_range).all()
        or feature_range[0] >= feature_range[1]
        or samples_seen < 1
        or any(value.shape != (count,) for value in arrays.values())
    ):
        raise ArtifactValidationError("Scaler feature shape or range is incompatible.")
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ArtifactValidationError("Scaler contains non-finite values.")
    scaler = MinMaxScaler(feature_range=feature_range)
    scaler.n_features_in_ = count
    scaler.n_samples_seen_ = samples_seen
    for name, value in arrays.items():
        setattr(scaler, f"{name}_", value)
    return scaler


def _artifact_root(ticker: str, model_type: str) -> Path:
    if not re.fullmatch(r"[A-Z0-9.\-]{1,12}", ticker):
        raise ArtifactValidationError("Invalid ticker artifact identity.")
    if model_type not in {
        "lstm",
        "gru",
        "attention",
        "bilstm_attention_regression",
        "bilstm_attention_direction",
    }:
        raise ArtifactValidationError("Invalid model artifact identity.")
    return Path(MODEL_DIR) / ticker / model_type


def _active_artifact_dir(ticker: str, model_type: str) -> Path | None:
    root = _artifact_root(ticker, model_type)
    pointer = root / "current.json"
    if not pointer.exists():
        return None
    try:
        version = json.loads(pointer.read_text(encoding="utf-8"))["version"]
    except Exception as exc:
        raise ArtifactValidationError("Artifact pointer is corrupt.") from exc
    if not isinstance(version, str) or not version.replace("-", "").isalnum():
        raise ArtifactValidationError("Artifact pointer contains an invalid version.")
    candidate = root / "versions" / version
    if not candidate.is_dir():
        raise ArtifactValidationError("Artifact version is incomplete.")
    return candidate


def _validate_integrity(directory: Path) -> None:
    integrity_path = directory / "integrity.json"
    try:
        expected = json.loads(integrity_path.read_text(encoding="utf-8"))["sha256"]
    except Exception as exc:
        raise ArtifactValidationError("Artifact integrity manifest is missing or corrupt.") from exc
    for name in ("model.keras", "scaler.json", "metadata.json"):
        path = directory / name
        if not path.is_file() or expected.get(name) != _sha256(path):
            raise ArtifactValidationError(f"Artifact integrity check failed for {name}.")


@contextmanager
def _process_file_lock(path: Path, timeout: float):
    """Cross-process lock using atomic exclusive creation with stale-lock recovery."""
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()}\n".encode())
            os.close(fd)
            break
        # Windows can report a sharing violation (PermissionError) while another
        # process owns the lock file, instead of the POSIX-style FileExistsError.
        # Both states mean the lock is contended and must follow the same timeout path.
        except (FileExistsError, PermissionError) as err:
            if time.monotonic() >= deadline:
                raise TrainingCapacityError("Timed out waiting for model artifact lock.") from err
            try:
                if time.time() - path.stat().st_mtime > settings.artifact_lock_timeout_seconds * 2:
                    path.unlink(missing_ok=True)
                    continue
            except OSError:
                pass
            time.sleep(0.1)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


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
        lb_p = float(1.0 - scipy_stats.chi2.cdf(lb_stat, df=lags))

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
    if close_scale == 0:
        return np.full_like(scaled_values, scaler.data_min_[close_idx])
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


def build_gru_model(
    forecast_days: int = MAX_FORECAST_DAYS, num_features: int = len(FEATURES)
) -> Sequential:
    """GRU candidate with the same capacity budget as the baseline LSTM."""

    model = Sequential(
        [
            Input(shape=(WINDOW_SIZE, num_features)),
            GRU(LSTM_UNITS, return_sequences=True),
            Dropout(0.25),
            GRU(LSTM_UNITS // 2, return_sequences=False),
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
    set_reproducibility()
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
    elif model_type == "gru":
        return build_gru_model(forecast_days=forecast_days, num_features=num_features)
    else:
        return build_lstm_model(forecast_days=forecast_days, num_features=num_features)


def _is_direction_model(model_type: str) -> bool:
    return "direction" in model_type or model_type == "attention"


# ── Load metrics & metadata ──────────────────────────────────────────
def load_metadata(ticker: str, model_type: str = "lstm") -> dict:
    directory = _active_artifact_dir(ticker, model_type)
    meta_path = directory / "metadata.json" if directory else Path("__missing__")
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to load metadata for %s/%s: %s", ticker, model_type, e)
    return {}


def load_metrics(ticker: str, model_type: str = "attention") -> dict:
    directory = _active_artifact_dir(ticker, model_type)
    metrics_path = directory / "metrics.json" if directory else Path("__missing__")
    if metrics_path.exists():
        try:
            with open(metrics_path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to load metrics for %s/%s: %s", ticker, model_type, e)
    return {}


def load_cross_validation(ticker: str, model_type: str = "lstm") -> dict:
    """Load cross_validation.json for a ticker/model_type."""
    directory = _active_artifact_dir(ticker, model_type)
    cv_path = directory / "cross_validation.json" if directory else Path("__missing__")
    if cv_path.exists():
        try:
            with open(cv_path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to load cross_validation for %s/%s: %s", ticker, model_type, e)
    return {}


def load_validation_results(ticker: str, model_type: str = "lstm") -> list:
    """Load validation_results.json (per-fold residuals) for a ticker/model_type."""
    directory = _active_artifact_dir(ticker, model_type)
    vr_path = directory / "validation_results.json" if directory else Path("__missing__")
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


def _compute_fold_metrics_direction(
    y_true_binary: np.ndarray,
    y_pred_probs: np.ndarray,
    training_targets: np.ndarray,
) -> dict:
    """Compute all-horizon metrics with a training-fold majority baseline."""
    y_true_binary = y_true_binary.ravel()
    y_pred_probs = y_pred_probs.ravel()
    probability_metrics = evaluate_probability_forecast(
        y_true_binary, y_pred_probs, training_targets=training_targets
    )
    training_targets = training_targets.ravel().astype(int)
    y_pred_binary = (y_pred_probs >= 0.5).astype(int)
    acc = float(accuracy_score(y_true_binary, y_pred_binary))
    prec = float(precision_score(y_true_binary, y_pred_binary, zero_division=0))
    rec = float(recall_score(y_true_binary, y_pred_binary, zero_division=0))
    f1 = float(f1_score(y_true_binary, y_pred_binary, zero_division=0))
    training_majority = int(np.bincount(training_targets, minlength=2).argmax())
    naive_baseline = float(np.mean(y_true_binary == training_majority))
    return {
        "direction_accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "naive_baseline": round(naive_baseline, 4),
        "balanced_accuracy": round(probability_metrics["balanced_accuracy"], 4),
        "brier_score": round(probability_metrics["brier_score"], 6),
        "log_loss": round(probability_metrics["log_loss"], 6),
        "rmse": None,
        "mae": None,
        "mape": None,
        "r2": None,
    }


def _train_single_fold(
    X_train_fold: np.ndarray,
    y_train_fold: np.ndarray,
    model_type: str,
    forecast_days: int,
    num_features: int,
) -> tuple:
    """Train one fold model using only the training fold for fitting and early stopping."""
    model = _build_model_for_type(model_type, forecast_days, num_features)
    early_stop = EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)
    fitting, inner_validation = purged_tail_split(
        len(X_train_fold), validation_fraction=0.1, purge=forecast_days - 1
    )
    X_fit = X_train_fold[fitting]
    y_fit = y_train_fold[fitting]
    X_inner_validation = X_train_fold[inner_validation]
    y_inner_validation = y_train_fold[inner_validation]

    t0 = time.time()
    history = model.fit(
        X_fit,
        y_fit,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_inner_validation, y_inner_validation),
        callbacks=[early_stop],
        shuffle=False,
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
    splits = generate_validation_splits(n_rows, VALIDATION_CONFIG)

    fold_results = []
    all_fold_metrics = []

    logger.info("Starting Walk-Forward Validation: %d folds, model_type=%s", n_folds, model_type)

    for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):
        fold_start_t = time.time()

        # ── Per-fold scaler (fit on train partition only) ────────────
        train_features_raw = aligned_features[train_idx]
        fold_scaler = MinMaxScaler()
        fold_scaler.fit(train_features_raw)
        all_features_scaled = fold_scaler.transform(aligned_features)

        train_dates_fold = aligned_dates[train_idx]
        val_dates_fold = aligned_dates[val_idx]

        def build_for_targets(target_starts: list[int], scaled_features=all_features_scaled):
            X_rows, y_rows, date_rows, origin_rows = [], [], [], []
            close_idx = FEATURES.index("Close")
            for start in target_starts:
                if start < WINDOW_SIZE or start + forecast_days > n_rows:
                    continue
                X_rows.append(scaled_features[start - WINDOW_SIZE : start])
                if is_dir:
                    y_rows.append((log_returns[start : start + forecast_days] > 0).astype(int))
                else:
                    y_rows.append(scaled_features[start : start + forecast_days, close_idx])
                date_rows.append(str(aligned_dates[start - 1].date()))
                origin_rows.append(float(aligned_features[start - 1, close_idx]))
            return np.asarray(X_rows), np.asarray(y_rows), date_rows, np.asarray(origin_rows)

        train_last = int(train_idx[-1])
        val_first, val_last = int(val_idx[0]), int(val_idx[-1])
        train_targets = list(range(WINDOW_SIZE, train_last - forecast_days + 2))
        val_targets = list(range(val_first, val_last - forecast_days + 2))
        X_train_f, y_train_f, _, _ = build_for_targets(train_targets)
        X_val_f, y_val_f, val_seq_dates, val_origins = build_for_targets(val_targets)

        if len(X_train_f) == 0 or len(X_val_f) == 0:
            logger.warning("Fold %d skipped: empty sequences after windowing.", fold_idx)
            continue

        # ── Train fold model ─────────────────────────────────────────
        fold_model, fold_history, fold_train_secs = _train_single_fold(
            X_train_f,
            y_train_f,
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
            metrics = _compute_fold_metrics_direction(y_val_f, preds, y_train_f)
            horizon_metrics = None
        else:
            pred_prices = _unscale_close(preds, fold_scaler)
            true_prices = _unscale_close(y_val_f, fold_scaler)
            horizon_metrics = evaluate_forecast_horizons(
                true_prices,
                pred_prices,
                val_origins,
                horizons=range(1, forecast_days + 1),
                scale_series=aligned_features[train_idx, FEATURES.index("Close")],
            )
            metrics = {
                key: horizon_metrics["pooled"].get(key)
                for key in (
                    "rmse",
                    "mae",
                    "mape",
                    "r2",
                    "direction_accuracy",
                    "mase",
                    "rmsse",
                    "relative_mae",
                    "relative_rmse",
                )
            }

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
            "train_index_start": int(train_idx[0]),
            "train_index_end": int(train_idx[-1]),
            "validation_index_start": int(val_idx[0]),
            "validation_index_end": int(val_idx[-1]),
            "gap": VALIDATION_CONFIG.gap,
            "train_samples": len(X_train_f),
            "validation_samples": len(X_val_f),
            "training_seconds": fold_train_secs,
            "fold_total_seconds": fold_total_secs,
            "early_stopping_source": "training_fold_tail",
            "evaluation_source": "untouched_walk_forward_fold",
            "metric_scope": "forecast_origin_horizon_pairs",
            "horizon_metrics": horizon_metrics,
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
    cv_summary.update(
        {
            "metric_source": "walk_forward_out_of_fold" if all_fold_metrics else "unavailable",
            "metric_scope": (
                "forecast_origin_horizon_pairs" if all_fold_metrics else "unavailable"
            ),
            "validation_method": VALIDATION_CONFIG.method,
            "validation_horizon": VALIDATION_CONFIG.horizon,
            "validation_gap": VALIDATION_CONFIG.gap,
            "validation_min_train_size": VALIDATION_CONFIG.min_train_size,
        }
    )

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
    feature_metadata: dict | None = None,
):
    """
    Production training pipeline:
    1. Configured walk-forward validation to generate untouched out-of-fold metrics.
    2. Train final production model on 100% of available data.
    3. Atomically save everything: model, scaler, metadata, history, cross_validation, validation_results.

    Parameters
    ----------
    X_train, y_train    : labelled train sequences (kept for API compatibility; used for final training).
    X_test, y_test      : remaining chronological sequences, combined only for final serving fit.
    feature_df          : optional raw feature DataFrame; required for walk-forward splits.
    """
    if scaler is None:
        raise ValueError("A fitted scaler is required to train and persist a model artifact.")
    if X_train.ndim != 3 or X_test.ndim != 3 or y_train.ndim != 2 or y_test.ndim != 2:
        raise ValueError("Training arrays must be 3D inputs and 2D targets.")
    if not len(X_train) or not len(X_test) or not len(y_train) or not len(y_test):
        raise ValueError("Training and serving partitions must both contain samples.")
    if X_train.shape[2] != len(FEATURES) or X_test.shape[2] != len(FEATURES):
        raise ValueError("Training feature count is incompatible with the feature schema.")
    if X_train.shape[1] != WINDOW_SIZE or X_test.shape[1] != WINDOW_SIZE:
        raise ValueError("Training sequence window is incompatible with the feature schema.")
    if getattr(scaler, "n_features_in_", None) != len(FEATURES):
        raise ValueError("Training scaler is incompatible with the feature schema.")
    if y_train.shape[1] != y_test.shape[1]:
        raise ValueError("Training and serving target horizons differ.")
    forecast_days = int(y_train.shape[1])
    if not 1 <= forecast_days <= MAX_FORECAST_DAYS:
        raise ValueError(f"Training horizon must be between 1 and {MAX_FORECAST_DAYS}.")
    if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
        raise ValueError("Training features contain non-finite values.")
    if _is_direction_model(model_type) and not set(
        np.unique(np.concatenate([y_train, y_test]))
    ).issubset({0, 1}):
        raise ValueError("Direction targets must be binary.")
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

    logger.info(
        "Selecting and refitting final production model for %s (%s):\n"
        "  Schema: v%d | Window: %d | Features: %d | Total Samples: %d",
        ticker,
        model_type,
        SCHEMA_VERSION,
        WINDOW_SIZE,
        num_features,
        len(X_all),
    )
    selection_train, selection_validation = purged_tail_split(
        len(X_all), validation_fraction=0.1, purge=forecast_days - 1
    )
    selection_model = _build_model_for_type(model_type, forecast_days, num_features)
    selection_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    start_time = time.time()
    selection_history = selection_model.fit(
        X_all[selection_train],
        y_all[selection_train],
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_all[selection_validation], y_all[selection_validation]),
        callbacks=[selection_stop],
        shuffle=False,
        verbose=0,
    )
    selection_epochs = len(selection_history.history["loss"])
    best_epoch = int(np.argmin(selection_history.history["val_loss"]) + 1)
    final_model = _build_model_for_type(model_type, forecast_days, num_features)
    history = final_model.fit(
        X_all,
        y_all,
        epochs=best_epoch,
        batch_size=BATCH_SIZE,
        shuffle=False,
        verbose=0,
    )
    training_duration_seconds = round(time.time() - start_time, 2)
    epochs_trained = len(history.history["loss"])
    early_stopped = selection_epochs < EPOCHS

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

    # Published metrics are exclusively out-of-fold; the production model has seen all samples.
    published_metrics: dict = {
        "metric_source": "walk_forward_out_of_fold",
        "metric_scope": cv_summary.get("metric_scope", "forecast_origin_horizon_pairs"),
    }
    for key in (
        "rmse",
        "mae",
        "mape",
        "r2",
        "direction_accuracy",
        "mase",
        "rmsse",
        "relative_mae",
        "relative_rmse",
        "precision",
        "recall",
        "balanced_accuracy",
        "brier_score",
        "log_loss",
        "f1",
        "naive_baseline",
    ):
        value = cv_summary.get(f"average_{key}")
        if value is not None:
            public_key = "directional_accuracy" if key == "direction_accuracy" else key
            published_metrics[public_key] = value
    if cv_summary.get("folds_completed", 0) == 0:
        published_metrics = {
            "metric_source": "unavailable",
            "detail": "No leakage-free walk-forward folds completed.",
        }

    # ── Fail-Safe Atomic Save ─────────────────────────────────────────
    base_dir = _artifact_root(ticker, model_type)
    version_id = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex}"
    tmp_dir = base_dir / "versions" / version_id
    tmp_dir.mkdir(parents=True, exist_ok=False)

    activated = False
    try:
        final_model.save(str(tmp_dir / "model.keras"))
        model_saved_path = tmp_dir / "model.keras"
        if scaler is not None:
            _save_scaler_json(scaler, tmp_dir / "scaler.json")

        meta = {
            "schema_version": SCHEMA_VERSION,
            "model_type": model_type,
            "window_size": WINDOW_SIZE,
            "feature_count": num_features,
            "output_width": forecast_days,
            "features": FEATURES,
            "dataset_fingerprint": fingerprint,
            "feature_stats": feature_stats,
            "epochs_trained": epochs_trained,
            "early_stopped": early_stopped,
            "best_epoch": best_epoch,
            "train_loss": history.history["loss"][-1] if "loss" in history.history else None,
            "val_loss": (
                selection_history.history["val_loss"][-1]
                if "val_loss" in selection_history.history
                else None
            ),
            "training_duration_seconds": training_duration_seconds,
            "scaler": "MinMaxScaler",
            "validation_method": VALIDATION_CONFIG.method,
            "validation_folds": VALIDATION_CONFIG.folds,
            "validation_horizon": VALIDATION_CONFIG.horizon,
            "validation_gap": VALIDATION_CONFIG.gap,
            "validation_min_train_size": VALIDATION_CONFIG.min_train_size,
            "seed": VALIDATION_CONFIG.seed,
            "deterministic": VALIDATION_CONFIG.deterministic,
            "metric_source": "walk_forward_out_of_fold",
            "metric_scope": "forecast_origin_horizon_pairs",
            "data_snapshot": feature_metadata or {},
            "app_version": APP_VERSION,
            "tensorflow_version": tf.__version__,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "hyperparameters": {
                "optimizer": "adam",
                "learning_rate": 0.001,
                "batch_size": BATCH_SIZE,
                "epochs_max": EPOCHS,
                "epochs_trained": len(history.history["loss"]) if history else 0,
                "selection_epochs": selection_epochs,
                "selection_best_epoch": best_epoch,
                "production_refit_samples": len(X_all),
                "selection_purge": forecast_days - 1,
                "selection_validation_fraction": 0.1,
                "patience": 10,
                "dropout": 0.25,
                "lstm_units": LSTM_UNITS,
                "window_size": WINDOW_SIZE,
                "forecast_days": forecast_days,
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

        with open(tmp_dir / "metrics.json", "w") as f:
            json.dump(published_metrics, f, indent=2)

        # Walk-forward outputs
        with open(tmp_dir / "cross_validation.json", "w") as f:
            json.dump(cv_summary, f, indent=2)

        with open(tmp_dir / "validation_results.json", "w") as f:
            json.dump(fold_results, f, indent=2)

        hashes = {
            name: _sha256(tmp_dir / name)
            for name in ("model.keras", "scaler.json", "metadata.json")
        }
        (tmp_dir / "integrity.json").write_text(
            json.dumps({"algorithm": "sha256", "sha256": hashes}, indent=2),
            encoding="utf-8",
        )

        pointer_tmp = base_dir / f".current-{uuid.uuid4().hex}.json"
        pointer_tmp.write_text(json.dumps({"version": version_id}), encoding="utf-8")
        os.replace(pointer_tmp, base_dir / "current.json")
        activated = True

        with _process_file_lock(base_dir / ".access.lock", settings.artifact_lock_timeout_seconds):
            versions = sorted(
                (base_dir / "versions").iterdir(),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            for old_version in versions[settings.model_versions_to_keep :]:
                shutil.rmtree(old_version, ignore_errors=True)

        logger.info("Successfully trained and saved model → %s", base_dir)

    except Exception:
        if tmp_dir.exists() and not activated:
            shutil.rmtree(tmp_dir)
        logger.error("Failed to save trained model artifacts to %s", tmp_dir, exc_info=True)
        raise
    finally:
        if tracemalloc.is_tracing():
            tracemalloc.stop()

    update_manifest(ticker, model_type, {"status": "trained", "schema_version": SCHEMA_VERSION})
    return final_model, scaler


def update_manifest(ticker: str, model_type: str, metadata: dict | None = None) -> None:
    """Automatically update saved_models/manifest.json."""
    manifest_path = Path(MODEL_DIR) / "manifest.json"
    try:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with _process_file_lock(
            manifest_path.with_suffix(".lock"), settings.artifact_lock_timeout_seconds
        ):
            manifest = {}
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except Exception:
                    logger.warning("Replacing corrupt manifest.json")
            key = f"{ticker}_{model_type}"
            manifest[key] = {
                "ticker": ticker,
                "model_type": model_type,
                "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "version": APP_VERSION,
                "schema_version": SCHEMA_VERSION,
                "metadata": metadata or {},
            }
            temp_path = manifest_path.with_name(f".manifest-{uuid.uuid4().hex}.json")
            temp_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            os.replace(temp_path, manifest_path)
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


def enforce_storage_quota(exclude: tuple[str, str] | None = None) -> None:
    """Bound model count/storage and evict the least-recently-used inactive model."""
    root = Path(MODEL_DIR)
    root.mkdir(parents=True, exist_ok=True)
    free_mb = shutil.disk_usage(root).free / (1024 * 1024)
    protected = _artifact_root(*exclude) if exclude else None
    candidates: list[tuple[float, Path, int]] = []
    all_roots: list[tuple[Path, int]] = []
    for pointer in root.glob("*/*/current.json"):
        model_root = pointer.parent
        size = sum(path.stat().st_size for path in model_root.rglob("*") if path.is_file())
        all_roots.append((model_root, size))
        if model_root != protected:
            candidates.append((pointer.stat().st_mtime, model_root, size))

    total_bytes = sum(size for _, size in all_roots)
    projected_count = len(all_roots) + (1 if protected and not protected.exists() else 0)
    max_bytes = settings.model_max_storage_mb * 1024 * 1024
    candidates.sort(key=lambda item: item[0])
    while candidates and (
        projected_count > settings.model_max_count
        or total_bytes >= max_bytes
        or free_mb < settings.model_min_free_mb
    ):
        _, victim, size = candidates.pop(0)
        try:
            with (
                _process_file_lock(victim / ".train.lock", timeout=0),
                _process_file_lock(victim / ".access.lock", timeout=0),
            ):
                shutil.rmtree(victim)
        except (OSError, TrainingCapacityError):
            logger.info("Skipping busy artifact during quota eviction: %s", victim)
            continue
        total_bytes -= size
        projected_count -= 1
        free_mb = shutil.disk_usage(root).free / (1024 * 1024)

    if (
        projected_count > settings.model_max_count
        or total_bytes >= max_bytes
        or free_mb < settings.model_min_free_mb
    ):
        raise TrainingCapacityError("Model storage quota is exhausted.")


# ── Staleness & Schema Validation ────────────────────────────────────
def _is_stale(path: Path) -> bool:
    if not path.exists():
        return True
    age_days = (time.time() - path.stat().st_mtime) / 86400
    return age_days > MODEL_MAX_AGE_DAYS


def is_schema_valid(ticker: str, model_type: str, expected_output_width: int | None = None) -> bool:
    """Check if saved metadata matches current SCHEMA_VERSION and FEATURES list."""
    meta = load_metadata(ticker, model_type)
    if not meta:
        return False
    valid = (
        meta.get("schema_version") == SCHEMA_VERSION
        and meta.get("features") == FEATURES
        and meta.get("window_size") == WINDOW_SIZE
        and meta.get("feature_count") == len(FEATURES)
    )
    if expected_output_width is not None:
        valid = valid and meta.get("output_width") == expected_output_width
    return valid


def _load_valid_artifact(
    ticker: str, model_type: str, expected_output_width: int, allow_stale: bool = False
):
    root = _artifact_root(ticker, model_type)
    # Readers use a stable root-level lock.  A writer only changes ``current.json``
    # after its version is complete, while pruning/eviction takes this same lock;
    # no reader can therefore observe a directory disappearing mid-deserialisation.
    with _process_file_lock(root / ".access.lock", settings.artifact_lock_timeout_seconds):
        directory = _active_artifact_dir(ticker, model_type)
        if directory is None:
            raise ArtifactValidationError("No versioned artifact is active.")
        _validate_integrity(directory)
        model_path = directory / "model.keras"
        scaler_path = directory / "scaler.json"
        if not allow_stale and _is_stale(model_path):
            raise ArtifactValidationError("Artifact is stale.")
        if not is_schema_valid(ticker, model_type, expected_output_width):
            raise ArtifactValidationError("Artifact metadata is incompatible.")
        loaded_scaler = _load_scaler_json(scaler_path)
        # ``safe_mode`` rejects Lambda/Python payloads embedded in a .keras archive and
        # ``compile=False`` avoids deserialising an unneeded optimizer state.  This is
        # deliberately after the integrity validation above: an artifact from outside
        # the trusted model store is never loaded merely because it has the right name.
        try:
            model = load_model(
                str(model_path),
                custom_objects={"TemporalAttention": TemporalAttention},
                safe_mode=True,
                compile=False,
            )
            output = model.outputs[0] if isinstance(model.outputs, list) else model.output
            actual_width = int(output.shape[-1])
        except Exception as exc:
            raise ArtifactValidationError("Artifact model cannot be safely deserialised.") from exc
        if actual_width != expected_output_width:
            raise ArtifactValidationError(
                f"Model output width mismatch: expected {expected_output_width}, got {actual_width}."
            )
        return model, loaded_scaler


def load_fresh_artifact(ticker: str, model_type: str, expected_output_width: int):
    """Load a serving artifact without admitting any training work."""
    lock = _get_ticker_lock(ticker)
    with lock:
        return _load_valid_artifact(ticker, model_type, expected_output_width)


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
    feature_metadata: dict | None = None,
    telemetry=None,
    allow_stale_fallback: bool = True,
):
    """Load or train an artifact, with an optional stale-artifact failure fallback."""
    expected_output_width = int(y_train.shape[1])
    lock = _get_ticker_lock(ticker)

    def artifact_state(error: ArtifactValidationError) -> str:
        message = str(error).lower()
        if "no versioned artifact" in message:
            return "missing"
        if "stale" in message:
            return "stale"
        return "incompatible"

    def load_artifact(allow_stale: bool = False):
        started = time.perf_counter()
        try:
            return _load_valid_artifact(
                ticker, model_type, expected_output_width, allow_stale=allow_stale
            )
        finally:
            if telemetry is not None:
                telemetry.add_timing("artifact_load_validation", time.perf_counter() - started)

    def set_artifact(state: str, action: str) -> None:
        if telemetry is not None:
            telemetry.set_artifact(state, action)

    with lock:
        try:
            loaded = load_artifact()
            set_artifact("fresh", "loaded")
            logger.info("Loaded valid cached model (%s/%s)", ticker, model_type)
            return loaded
        except ArtifactValidationError as err:
            initial_state = artifact_state(err)
            set_artifact(initial_state, "not_applicable")
            logger.info("No compatible fresh artifact for %s/%s", ticker, model_type)

        artifact_lock = _artifact_root(ticker, model_type) / ".train.lock"
        with _process_file_lock(artifact_lock, settings.training_wait_seconds):
            try:
                loaded = load_artifact()
                set_artifact(initial_state, "loaded")
                return loaded
            except ArtifactValidationError:
                pass
            if not _training_slots.acquire(timeout=settings.training_wait_seconds):
                raise TrainingCapacityError("Training concurrency limit reached.")
            try:
                if telemetry is not None:
                    telemetry.set_stage("training")
                training_started: float | None = None
                training_recorded = False
                with _process_file_lock(
                    Path(MODEL_DIR) / ".quota.lock", settings.artifact_lock_timeout_seconds
                ):
                    enforce_storage_quota(exclude=(ticker, model_type))
                training_started = time.perf_counter()
                trained = train_model(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    ticker,
                    scaler=scaler,
                    model_type=model_type,
                    feature_df=feature_df,
                    feature_metadata=feature_metadata,
                )
                if telemetry is not None:
                    telemetry.add_timing("training", time.perf_counter() - training_started)
                    training_recorded = True
                    telemetry.set_artifact(initial_state, "retrained")
                with _process_file_lock(
                    Path(MODEL_DIR) / ".quota.lock", settings.artifact_lock_timeout_seconds
                ):
                    enforce_storage_quota(exclude=(ticker, model_type))
                return trained
            except Exception:
                if telemetry is not None and training_started is not None and not training_recorded:
                    telemetry.add_timing("training", time.perf_counter() - training_started)
                if not allow_stale_fallback:
                    raise
                try:
                    loaded = load_artifact(allow_stale=True)
                    set_artifact(initial_state, "loaded")
                    return loaded
                except ArtifactValidationError:
                    raise
            finally:
                _training_slots.release()


# ── Evaluate ─────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, scaler):
    """
    Diagnostic metric utility in original price scale.

    API responses do not publish this result for the all-data production model; they load only
    walk-forward out-of-fold metrics persisted by ``train_model``.
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
    if not 1 <= days <= MAX_FORECAST_DAYS:
        raise ValueError(f"Forecast days must be between 1 and {MAX_FORECAST_DAYS}.")
    if len(feature_df) < WINDOW_SIZE:
        raise ValueError("Not enough feature rows for prediction.")
    feature_values = feature_df[FEATURES].values
    last_window = feature_values[-WINDOW_SIZE:]
    scaled_window = scaler.transform(last_window)
    input_seq = scaled_window.reshape(1, WINDOW_SIZE, len(FEATURES))

    raw_predictions = model.predict(input_seq, verbose=0)
    predictions = (
        raw_predictions[0] if isinstance(raw_predictions, (list, tuple)) else raw_predictions
    )
    preds_scaled = np.asarray(predictions, dtype=float).reshape(-1)
    if len(preds_scaled) < days:
        raise ArtifactValidationError(
            f"Price model returned {len(preds_scaled)} outputs for a {days}-day forecast."
        )
    if not np.isfinite(preds_scaled[:days]).all():
        raise ValueError("Price model returned non-finite predictions.")
    prices = _unscale_close(preds_scaled[:days], scaler)
    if not np.isfinite(prices).all() or np.any(prices <= 0):
        raise ValueError("Price model returned invalid non-positive prices.")
    return prices.tolist()


def predict_direction(model, feature_df, scaler, days: int = 7):
    """
    Directional multi-step prediction using the Attention model.
    """
    if not 1 <= days <= MAX_FORECAST_DAYS:
        raise ValueError(f"Forecast days must be between 1 and {MAX_FORECAST_DAYS}.")
    if len(feature_df) < WINDOW_SIZE:
        raise ValueError("Not enough feature rows for prediction.")
    feature_values = feature_df[FEATURES].values
    last_window = feature_values[-WINDOW_SIZE:]
    scaled_window = scaler.transform(last_window)
    input_seq = scaled_window.reshape(1, WINDOW_SIZE, len(FEATURES))

    output = model.predict(input_seq, verbose=0)
    if not isinstance(output, (list, tuple)) or len(output) != 2:
        raise ArtifactValidationError(
            "Direction model must return probabilities and attention weights."
        )
    probabilities, attention_weights = output
    probabilities = np.asarray(probabilities, dtype=float)
    attention_weights = np.asarray(attention_weights, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[0] != 1 or probabilities.shape[1] < days:
        raise ArtifactValidationError(
            "Direction model probability output has an incompatible horizon."
        )
    if attention_weights.shape[0] != 1 or attention_weights.size != WINDOW_SIZE:
        raise ArtifactValidationError("Direction model attention output has an incompatible shape.")
    probabilities = probabilities[0, :days]
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Direction model returned invalid probabilities.")

    probs_list = [float(p) for p in probabilities]
    directions = ["Up" if p > 0.5 else "Down" for p in probs_list]

    return directions, probs_list, attention_weights.reshape(-1).tolist()
