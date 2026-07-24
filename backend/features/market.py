"""Market context features (SPY, QQQ, VIX, ^TNX returns)."""

import numpy as np
import pandas as pd
import yfinance as yf  # type: ignore[import-untyped]

MARKET_TICKERS = {
    "SPY": "SPY_Return_1D",
    "QQQ": "QQQ_Return_1D",
    "^VIX": "VIX_Return_1D",
    "^TNX": "TNX_Return_1D",
}


def add_market_context(df: pd.DataFrame, period: str = "5y") -> pd.DataFrame:
    """
    Fetch benchmark index prices, join on the target DataFrame's index,
    and compute 1-day log returns for stationarity.
    """
    result = df.copy()

    for ticker, feature_name in MARKET_TICKERS.items():
        try:
            m_data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            if not m_data.empty and "Close" in m_data.columns:
                m_close = m_data["Close"]
                # Align on primary df index with left join & ffill for holidays
                aligned_close = m_close.reindex(df.index).ffill().bfill()
                # Compute 1-day log return
                returns = np.log(aligned_close / aligned_close.shift(1)).fillna(0.0)
                # Handle single-column Series squeeze if needed
                if isinstance(returns, pd.DataFrame):
                    returns = returns.iloc[:, 0]
                result[feature_name] = returns
            else:
                result[feature_name] = 0.0
        except Exception:
            result[feature_name] = 0.0

    return result
