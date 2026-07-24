"""Pure pandas/numpy implementations of technical indicators (zero external dependencies)."""

import numpy as np
import pandas as pd


def compute_sma(df: pd.DataFrame, window: int = 20, column: str = "Close") -> pd.Series:
    """Simple Moving Average."""
    return df[column].rolling(window=window).mean()


def compute_ema(df: pd.DataFrame, window: int = 20, column: str = "Close") -> pd.Series:
    """Exponential Moving Average."""
    return df[column].ewm(span=window, adjust=False).mean()


def compute_rsi(df: pd.DataFrame, window: int = 14, column: str = "Close") -> pd.Series:
    """Relative Strength Index (RSI)."""
    delta = df[column].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / (loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def compute_macd(
    df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, column: str = "Close"
) -> pd.DataFrame:
    """Moving Average Convergence Divergence (MACD) and Signal line."""
    ema_fast = df[column].ewm(span=fast, adjust=False).mean()
    ema_slow = df[column].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()

    return pd.DataFrame({"MACD": macd, "MACD_Signal": macd_signal}, index=df.index)


def compute_bollinger_bands(
    df: pd.DataFrame, window: int = 20, num_std: float = 2.0, column: str = "Close"
) -> pd.DataFrame:
    """Bollinger Bands (Upper and Lower)."""
    sma = compute_sma(df, window=window, column=column)
    rolling_std = df[column].rolling(window=window).std()

    upper = sma + (rolling_std * num_std)
    lower = sma - (rolling_std * num_std)

    return pd.DataFrame({"BB_Upper": upper, "BB_Lower": lower}, index=df.index)


def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range (ATR)."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr


def compute_obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume (OBV)."""
    close = df["Close"]
    volume = df["Volume"]

    direction = np.sign(close.diff().fillna(0))
    obv = (direction * volume).cumsum()
    return obv


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and append all technical indicators to the DataFrame."""
    result = df.copy()

    result["SMA_20"] = compute_sma(result, window=20)
    result["EMA_20"] = compute_ema(result, window=20)
    result["RSI_14"] = compute_rsi(result, window=14)

    macd_df = compute_macd(result)
    result["MACD"] = macd_df["MACD"]
    result["MACD_Signal"] = macd_df["MACD_Signal"]

    bb_df = compute_bollinger_bands(result)
    result["BB_Upper"] = bb_df["BB_Upper"]
    result["BB_Lower"] = bb_df["BB_Lower"]

    result["ATR_14"] = compute_atr(result)
    result["OBV"] = compute_obv(result)

    return result
