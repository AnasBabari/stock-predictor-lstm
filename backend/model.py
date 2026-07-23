# model.py — Builds, trains, evaluates, and runs the LSTM model
#
# Fixes applied:
#   3.2  Direct multi-step output (Dense(N) instead of recursive loop)
#   3.3  Explicit validation_data=(X_test, y_test)
#   3.6  Additional metrics: MAPE, R², directional accuracy
#   3.7  verbose=0 during training
#   2.4  Per-ticker lock to prevent concurrent training races
#   6.7  pathlib for cross-platform path construction

import json
import logging
import threading
import time
import weakref
from pathlib import Path

import joblib  # type: ignore[import-untyped]
import numpy as np
from sklearn.metrics import (  # type: ignore[import-untyped]
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    accuracy_score,
)
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore[import-untyped]
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Attention, GlobalAveragePooling1D  # type: ignore[import-untyped]
from tensorflow.keras.models import Sequential, load_model, Model  # type: ignore[import-untyped]

from config import (
    BATCH_SIZE,
    EPOCHS,
    LSTM_UNITS,
    MAX_FORECAST_DAYS,
    MODEL_DIR,
    MODEL_MAX_AGE_DAYS,
    WINDOW_SIZE,
)

logger = logging.getLogger(__name__)

# ── Per-ticker lock (2.4, Bug 3) ───────────────────────────────────────
_training_locks: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
_locks_lock = threading.Lock()


def _get_ticker_lock(ticker: str) -> threading.Lock:
    with _locks_lock:
        lock = _training_locks.get(ticker)
        if lock is None:
            lock = threading.Lock()
            _training_locks[ticker] = lock
        return lock


# ── Build ────────────────────────────────────────────────────────────
def build_model(forecast_days: int = MAX_FORECAST_DAYS) -> Sequential:
    """Two-layer LSTM with dropout and direct multi-step output (3.2)."""
    model = Sequential(
        [
            Input(shape=(WINDOW_SIZE, 1)),
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


def build_attention_lstm_model(forecast_days: int = MAX_FORECAST_DAYS) -> Model:
    """LSTM with Attention mechanism for binary classification."""
    inputs = Input(shape=(WINDOW_SIZE, 1))
    
    # LSTM returns sequences so Attention can act on each time step
    lstm_out = LSTM(LSTM_UNITS, return_sequences=True)(inputs)
    lstm_out = Dropout(0.25)(lstm_out)
    
    # Self-attention over the LSTM sequence outputs
    attention_out, attention_weights = Attention()([lstm_out, lstm_out], return_attention_scores=True)
    
    # Pool sequence to a single vector per batch item
    pooled = GlobalAveragePooling1D()(attention_out)
    
    # Final dense layer for binary classification
    predictions = Dense(forecast_days, activation="sigmoid")(pooled)
    
    # Model with two outputs: predictions and attention weights
    model = Model(inputs=inputs, outputs=[predictions, attention_weights])
    model.compile(optimizer="adam", loss=["binary_crossentropy", None])
    
    return model


# ── Load metrics helper ──────────────────────────────────────────────
def load_metrics(ticker: str, model_type: str = "attention") -> dict:
    metrics_path = Path(MODEL_DIR) / f"{ticker}_{model_type}_metrics.json"
    if metrics_path.exists():
        try:
            with open(metrics_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to load metrics for %s: %s", ticker, e)
    return {}


# ── Train ────────────────────────────────────────────────────────────
def train_model(X_train, y_train, X_test, y_test, ticker: str, scaler=None, model_type: str = "lstm"):
    """Train with explicit validation data (3.3), silent output (3.7), and scaler persistence."""
    forecast_days = y_train.shape[1]
    
    if model_type == "attention":
        model = build_attention_lstm_model(forecast_days=forecast_days)
    else:
        model = build_model(forecast_days=forecast_days)
        
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )
    logger.info(
        "Training model for %s (%d samples, %d-step output)…",
        ticker,
        len(X_train),
        forecast_days,
    )
    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=0,
    )
    model_path = Path(MODEL_DIR) / f"{ticker}_{model_type}_model.keras"
    scaler_path = Path(MODEL_DIR) / f"{ticker}_{model_type}_scaler.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    if scaler is not None:
        joblib.dump(scaler, str(scaler_path))
    logger.info("Model and scaler saved → %s", model_path)
    
    if model_type == "attention":
        preds = model.predict(X_test, verbose=0)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
            
        pred_first = (preds[:, 0] > 0.5).astype(int)
        true_first = y_test[:, 0].astype(int)
        
        precision = float(precision_score(true_first, pred_first, zero_division=0))
        recall = float(recall_score(true_first, pred_first, zero_division=0))
        
        majority_class = int(np.bincount(true_first).argmax()) if len(true_first) > 0 else 0
        naive_preds = np.full_like(true_first, majority_class)
        naive_baseline = float(accuracy_score(true_first, naive_preds)) if len(true_first) > 0 else 0.0
        
        metrics = {
            "precision": precision,
            "recall": recall,
            "naive_baseline": naive_baseline
        }
        
        metrics_path = Path(MODEL_DIR) / f"{ticker}_{model_type}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)
            
    return model, scaler


# ── Staleness check ──────────────────────────────────────────────────
def _is_stale(path: Path) -> bool:
    """Return True if the saved model is older than MODEL_MAX_AGE_DAYS."""
    if not path.exists():
        return True
    age_days = (time.time() - path.stat().st_mtime) / 86400
    return age_days > MODEL_MAX_AGE_DAYS


# ── Load or train ────────────────────────────────────────────────────
def load_or_train(ticker: str, X_train, y_train, X_test, y_test, scaler=None, model_type: str = "lstm"):
    """Load a cached model & scaler or train a new pair, with per-ticker locking (2.4)."""
    model_path = Path(MODEL_DIR) / f"{ticker}_{model_type}_model.keras"
    scaler_path = Path(MODEL_DIR) / f"{ticker}_{model_type}_scaler.joblib"
    lock = _get_ticker_lock(ticker)

    with lock:
        if model_path.exists() and scaler_path.exists() and not _is_stale(model_path):
            try:
                model = load_model(str(model_path))
                loaded_scaler = joblib.load(str(scaler_path))
                
                # Verify output shape matches
                expected_out = y_train.shape[1]
                if isinstance(model.output_shape, list):
                    actual_out = model.output_shape[0][-1]
                else:
                    actual_out = model.output_shape[-1]
                
                if actual_out != expected_out:
                    logger.info(
                        "Model output shape mismatch for %s (got %s, need %s) — retraining.",
                        ticker,
                        actual_out,
                        expected_out,
                    )
                    return train_model(X_train, y_train, X_test, y_test, ticker, scaler=scaler, model_type=model_type)
                logger.info("Loaded cached model & scaler for %s", ticker)
                return model, loaded_scaler
            except Exception:
                logger.warning(
                    "Failed to load model/scaler for %s, retraining…",
                    ticker,
                    exc_info=True,
                )
                
        return train_model(X_train, y_train, X_test, y_test, ticker, scaler=scaler, model_type=model_type)


# ── Evaluate (3.6) ──────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, scaler):
    """
    RMSE, MAE, MAPE, R², and directional accuracy on the held-out test
    set, in original price scale.  Evaluates next-day predictions
    (first output step) for comparability.
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

    preds = model.predict(X_test, verbose=0)  # (n, forecast_days)
    if isinstance(preds, (list, tuple)):
        preds = preds[0]
        
    pred_first = preds[:, 0]
    true_first = y_test[:, 0]

    pred_prices = scaler.inverse_transform(pred_first.reshape(-1, 1)).flatten()
    true_prices = scaler.inverse_transform(true_first.reshape(-1, 1)).flatten()

    rmse = float(np.sqrt(mean_squared_error(true_prices, pred_prices)))
    mae = float(mean_absolute_error(true_prices, pred_prices))

    # MAPE (skip zeros)
    nonzero = true_prices != 0
    mape = (
        float(
            np.mean(np.abs((true_prices[nonzero] - pred_prices[nonzero]) / true_prices[nonzero]))
            * 100
        )
        if np.any(nonzero)
        else None
    )

    # R²
    r2 = float(r2_score(true_prices, pred_prices))

    # Directional accuracy
    da = None
    if preds.shape[1] > 1:
        # Compare first vs last step in each forecast window
        pred_last = preds[:, -1]
        true_last = y_test[:, -1]

        pred_last_unscaled = scaler.inverse_transform(pred_last.reshape(-1, 1)).flatten()
        true_last_unscaled = scaler.inverse_transform(true_last.reshape(-1, 1)).flatten()

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


# ── Predict (3.2 — single forward pass) ─────────────────────────────
def predict_future(model, closing_prices, scaler, days: int = 7):
    """
    Direct multi-step prediction in a single forward pass.
    No recursive error accumulation.
    """
    last_window = closing_prices[-WINDOW_SIZE:]
    scaled = scaler.transform(last_window)
    input_seq = scaled.reshape(1, WINDOW_SIZE, 1)

    preds = model.predict(input_seq, verbose=0)
    if isinstance(preds, (list, tuple)):
        preds_scaled = preds[0][0]
    else:
        preds_scaled = preds[0]  # (MAX_FORECAST_DAYS,)
    preds_scaled = preds_scaled[:days]

    return scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten().tolist()
