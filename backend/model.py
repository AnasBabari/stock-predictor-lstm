# model.py - Builds, trains, and runs the  LSTM model

import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from config import LSTM_UNITS, EPOCHS, BATCH_SIZE, MODEL_DIR


def build_model():
    model = Sequential([
        LSTM(LSTM_UNITS, return_sequences=False, input_shape=(60, 1)),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def train_model(X_train, y_train, ticker: str):
    model = build_model()
    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=1
    )
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(f"{MODEL_DIR}/{ticker}_model.keras")
    return model


def load_or_train(ticker: str, X_train, y_train):
    
    path = f"{MODEL_DIR}/{ticker}_model.keras"
    if os.path.exists(path):
        return load_model(path)
    return train_model(X_train, y_train, ticker)


def predict_future(model, closing_prices, scaler, days: int = 5):
    
    last_60 = closing_prices[-60:]
    scaled = scaler.transform(last_60)
    input_seq = scaled.reshape(1, 60, 1)

    predictions = []
    for _ in range(days):
        pred = model.predict(input_seq, verbose=0)[0][0]
        predictions.append(pred)
        input_seq = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1)

    return scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    ).flatten().tolist()