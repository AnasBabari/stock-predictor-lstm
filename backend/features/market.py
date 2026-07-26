"""Market context features with explicit, versioned degradation handling."""

import logging

import numpy as np
import pandas as pd
import yfinance as yf  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)
MARKET_CONTEXT_SCHEMA_VERSION = 2


class MarketContextUnavailable(ValueError):
    """A retryable benchmark-data failure."""


MARKET_TICKERS = {
    "SPY": "SPY_Return_1D",
    "QQQ": "QQQ_Return_1D",
    "^VIX": "VIX_Return_1D",
    "^TNX": "TNX_Return_1D",
}


def add_market_context(df: pd.DataFrame, period: str = "5y") -> tuple[pd.DataFrame, dict]:
    """
    Fetch benchmark index prices, join on the target DataFrame's index,
    and compute 1-day log returns for stationarity.
    """
    result = df.copy()
    sources: dict[str, dict] = {}

    for ticker, feature_name in MARKET_TICKERS.items():
        try:
            m_data = yf.download(
                ticker,
                period=period,
                progress=False,
                auto_adjust=True,
                timeout=30,
            )

            if isinstance(m_data.columns, pd.MultiIndex):
                m_data.columns = m_data.columns.get_level_values(0)

            if m_data.empty or "Close" not in m_data.columns:
                raise MarketContextUnavailable(f"Benchmark {ticker} returned no closing prices.")

            m_close = m_data["Close"]
            if isinstance(m_close, pd.DataFrame):
                m_close = m_close.iloc[:, 0]

            # Reindex the complete source series before differencing so the first target row
            # uses a genuinely prior observation and closed-market days become a real 0 return.
            combined_index = m_close.index.union(df.index)
            aligned_close = m_close.reindex(combined_index).sort_index().ffill()
            returns = np.log(aligned_close / aligned_close.shift(1)).reindex(df.index)
            if returns.isna().any() or not np.isfinite(returns.to_numpy(dtype=float)).all():
                raise MarketContextUnavailable(
                    f"Benchmark {ticker} could not be aligned from prior observations."
                )

            if isinstance(returns, pd.DataFrame):
                returns = returns.iloc[:, 0]

            result[feature_name] = returns
            sources[feature_name] = {
                "ticker": ticker,
                "status": "live",
                "rows": int(len(m_data)),
                "alignment": "prior_observation_carry_forward_v2",
            }

        except MarketContextUnavailable:
            raise
        except Exception as exc:
            logger.warning("Market context unavailable for %s: %s", ticker, type(exc).__name__)
            raise MarketContextUnavailable(
                f"Benchmark {ticker} is temporarily unavailable."
            ) from exc

    return result, {
        "schema_version": MARKET_CONTEXT_SCHEMA_VERSION,
        "policy": "fail_closed",
        "imputation": "closed-market close carried forward before return calculation",
        "status": "complete",
        "sources": sources,
    }
