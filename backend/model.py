# model.py - Builds, trains, and runs the LSTM model

#type:ignore
import os
import time
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error
from config import LSTM_UNITS, EPOCHS, BATCH_SIZE, MODEL_DIR, WINDOW_SIZE, MODEL_MAX_AGE_DAYS


def build_model():
    """Two-layer LSTM with dropout regularisation and a dense hidden layer."""
    model = Sequential([
        LSTM(LSTM_UNITS, return_sequences=True, input_shape=(WINDOW_SIZE, 1)),
        Dropout(0.25),
        LSTM(LSTM_UNITS // 2, return_sequences=False),
        Dropout(0.25),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def train_model(X_train, y_train, ticker: str):
    model = build_model()
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )
    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1,
    )
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(f"{MODEL_DIR}/{ticker}_model.keras")
    return model


def _is_stale(path: str) -> bool:
    """Return True if the saved model is older than MODEL_MAX_AGE_DAYS."""
    if not os.path.exists(path):
        return True
    age_days = (time.time() - os.path.getmtime(path)) / 86400
    return age_days > MODEL_MAX_AGE_DAYS


def load_or_train(ticker: str, X_train, y_train):
    path = f"{MODEL_DIR}/{ticker}_model.keras"
    if os.path.exists(path) and not _is_stale(path):
        try:
            return load_model(path)
        except Exception:
            pass  # architecture mismatch after upgrade → retrain
    return train_model(X_train, y_train, ticker)


def evaluate_model(model, X_test, y_test, scaler):
    """Return RMSE and MAE on the held-out test set, in original price scale."""
    if len(X_test) == 0 or len(y_test) == 0:
        return {"rmse": None, "mae": None}

    preds = model.predict(X_test, verbose=0).flatten()
    pred_prices = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    true_prices = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    rmse = float(np.sqrt(mean_squared_error(true_prices, pred_prices)))
    mae = float(mean_absolute_error(true_prices, pred_prices))
    return {"rmse": round(rmse, 2), "mae": round(mae, 2)}


def predict_future(model, closing_prices, scaler, days: int = 7):
    last_window = closing_prices[-WINDOW_SIZE:]
    scaled = scaler.transform(last_window)
    input_seq = scaled.reshape(1, WINDOW_SIZE, 1)

    predictions = []
    for _ in range(days):
        pred = model.predict(input_seq, verbose=0)[0][0]
        predictions.append(pred)
        input_seq = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1)

    return scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    ).flatten().tolist()