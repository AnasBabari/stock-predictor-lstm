#Fetches and preprocesses stock data

import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from config import HISTORICAL_YEARS, WINDOW_SIZE, TRAIN_SPLIT


def fetch_data(ticker: str):
    
    data = yf.download(ticker, period=f"{HISTORICAL_YEARS}y")
    closing_prices = data["Close"].values.reshape(-1, 1)
    return closing_prices


def preprocess(closing_prices):
    
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(closing_prices)

    X, y = [], []
    for i in range(WINDOW_SIZE, len(scaled)):
        X.append(scaled[i - WINDOW_SIZE:i, 0])
        y.append(scaled[i, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))  

    split = int(len(X) * TRAIN_SPLIT)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    return X_train, X_test, y_train, y_test, scaler


def get_pipeline(ticker: str):
    
    closing_prices = fetch_data(ticker)
    return preprocess(closing_prices), closing_prices