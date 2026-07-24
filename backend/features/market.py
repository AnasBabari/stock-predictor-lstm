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
            m_data = yf.download(
                ticker,
                period=period,
                progress=False,
                auto_adjust=True,
            )

            if isinstance(m_data.columns, pd.MultiIndex):
                m_data.columns = m_data.columns.get_level_values(0)

            if m_data.empty:
                result[feature_name] = 0.0
                continue

            if "Close" not in m_data.columns:
                result[feature_name] = 0.0
                continue

            m_close = m_data["Close"]
            if isinstance(m_close, pd.DataFrame):
                m_close = m_close.iloc[:, 0]

            aligned_close = m_close.reindex(df.index).ffill().bfill()

            returns = np.log(aligned_close / aligned_close.shift(1)).fillna(0.0)

            if isinstance(returns, pd.DataFrame):
                returns = returns.iloc[:, 0]

            result[feature_name] = returns

        except Exception:
            result[feature_name] = 0.0

    return result
