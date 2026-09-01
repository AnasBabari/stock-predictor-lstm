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


def add_market_context_from_frames(
    df: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict]:
    """Build market returns from an already verified snapshot without network access."""

    result = df.copy()
    sources: dict[str, dict] = {}
    for ticker, feature_name in MARKET_TICKERS.items():
        source = frames.get(ticker)
        if source is None or source.empty or "Close" not in source:
            raise MarketContextUnavailable(
                f"Verified snapshot is missing benchmark context {ticker}."
            )
        close = source["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        combined_index = close.index.union(df.index)
        aligned_close = close.reindex(combined_index).sort_index().ffill()
        returns = np.log(aligned_close / aligned_close.shift(1)).reindex(df.index)
        # Initial market return NaNs are acceptable only where another
        # pre-existing feature is already in its rolling warm-up period.
        required_index = df.dropna().index
        required_returns = returns.reindex(required_index)
        if (
            required_returns.isna().any()
            or not np.isfinite(required_returns.to_numpy(dtype=float)).all()
        ):
            raise MarketContextUnavailable(
                f"Snapshot benchmark {ticker} cannot be aligned from prior observations."
            )
        result[feature_name] = returns
        sources[feature_name] = {
            "ticker": ticker,
            "status": "snapshot",
            "rows": int(len(source)),
            "last_date": close.dropna().index[-1].strftime("%Y-%m-%d")
            if len(close.dropna())
            else None,
            "alignment": "prior_observation_carry_forward_v2",
        }
    return result, {
        "schema_version": MARKET_CONTEXT_SCHEMA_VERSION,
        "policy": "verified_snapshot",
        "imputation": "closed-market close carried forward before return calculation",
        "status": "complete",
        "sources": sources,
    }


def add_market_context(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Fetch benchmark index prices in one batched download, join on the target
    DataFrame's index, and compute 1-day log returns for stationarity.

    A single ``yf.download`` call for all benchmark symbols replaces the
    former four sequential downloads (4x fewer round-trips on the snapshot
    build critical path). Any missing/empty benchmark fails closed.
    """
    result = df.copy()
    sources: dict[str, dict] = {}

    start_date = df.index.min().strftime("%Y-%m-%d") if not df.empty else None
    symbols = list(MARKET_TICKERS)

    try:
        raw = yf.download(
            symbols,
            start=start_date,
            progress=False,
            auto_adjust=True,
            timeout=30,
            group_by="ticker",
        )
    except MarketContextUnavailable:
        raise
    except Exception as exc:
        logger.warning("Market context unavailable: %s", type(exc).__name__)
        raise MarketContextUnavailable("Benchmark market data is temporarily unavailable.") from exc

    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise MarketContextUnavailable("Benchmark market data is temporarily unavailable.")

    for ticker, feature_name in MARKET_TICKERS.items():
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if ticker not in raw.columns.get_level_values(0):
                    raise MarketContextUnavailable(
                        f"Benchmark {ticker} returned no closing prices."
                    )
                m_data = raw[ticker]
            else:
                # Single-symbol fallback shape (no group_by hierarchy).
                m_data = raw

            if m_data.empty or "Close" not in m_data.columns:
                raise MarketContextUnavailable(f"Benchmark {ticker} returned no closing prices.")

            m_close = m_data["Close"]
            if isinstance(m_close, pd.DataFrame):
                m_close = m_close.iloc[:, 0]
            m_close = m_close.dropna()

            # Reindex the complete source series before differencing so the first target row
            # uses a genuinely prior observation and closed-market days become a real 0 return.
            combined_index = m_close.index.union(df.index)
            aligned_close = m_close.reindex(combined_index).sort_index().ffill()
            returns = np.log(aligned_close / aligned_close.shift(1)).reindex(df.index)
            valid_returns = returns.loc[df.dropna().index]
            if (
                valid_returns.isna().any()
                or not np.isfinite(valid_returns.to_numpy(dtype=float)).all()
            ):
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
                "last_date": m_close.index[-1].strftime("%Y-%m-%d") if len(m_close) else None,
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
