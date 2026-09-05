"""Same-market, leave-one-out historical context for the fixed cached basket."""

from dataclasses import replace

import numpy as np
import pandas as pd

from .gpu_pipeline import _normalise_ohlcv, build_price_features

CONTEXT_NAMES = (
    "market_return_1d",
    "market_return_5d",
    "market_return_20d",
    "market_vol_20d",
    "market_dispersion",
    "excess_return_1d",
    "excess_return_5d",
    "excess_return_20d",
    "context_missing",
    "context_stale_sessions",
    "context_peer_coverage",
)


def build_market_context(frames, coverage=0.80):
    if not 0 < coverage <= 1:
        raise ValueError("Invalid coverage threshold")
    normalized = {s: _normalise_ohlcv(f) for s, f in frames.items()}
    result = {}
    for uk in (False, True):
        symbols = sorted(s for s in normalized if s.endswith(".L") == uk)
        if not symbols:
            continue
        if len(symbols) < 3:
            raise ValueError("At least three stocks per market required")
        close = pd.concat({s: normalized[s].Close for s in symbols}, axis=1).sort_index()
        volume = pd.concat({s: normalized[s].Volume for s in symbols}, axis=1).reindex(close.index)
        returns = np.log(close).diff().where(volume.gt(0)).replace([np.inf, -np.inf], np.nan)
        count, total = returns.count(axis=1), returns.sum(axis=1)
        dispersion = returns.std(axis=1, ddof=1)
        for symbol in symbols:
            own = returns[symbol]
            peers = count - own.notna().astype(int)
            enough = peers >= max(2, int(np.ceil(coverage * (len(symbols) - 1))))
            market = ((total - own.fillna(0)) / peers.replace(0, np.nan)).where(enough)
            features = pd.DataFrame(index=close.index)
            for window in (1, 5, 20):
                features[f"market_return_{window}d"] = market.rolling(window).sum()
            features["market_vol_20d"] = market.rolling(20).std(ddof=1)
            features["market_dispersion"] = dispersion.where(enough)
            for window in (1, 5, 20):
                features[f"excess_return_{window}d"] = (
                    np.log(close[symbol]).diff(window) - features[f"market_return_{window}d"]
                )
            missing = features.isna().any(axis=1)
            positions = np.arange(len(features))
            last_complete = np.maximum.accumulate(np.where(~missing, positions, -1))
            features = features.ffill().fillna(0)
            features["context_missing"] = missing.astype(float)
            features["context_stale_sessions"] = positions - last_complete
            features["context_peer_coverage"] = peers / (len(symbols) - 1)
            result[symbol] = features.loc[:, CONTEXT_NAMES]
    return result


def append_context(dataset, frames, context):
    lookback = dataset.sequences.shape[1]
    augmented = np.empty(
        (*dataset.sequences.shape[:2], dataset.sequences.shape[2] + len(CONTEXT_NAMES)),
        dtype=np.float32,
    )
    augmented[:, :, : dataset.sequences.shape[2]] = dataset.sequences
    current = {}
    for stock_index, symbol in enumerate(dataset.ticker_names):
        feature_dates = build_price_features(frames[symbol]).index
        aligned = context[symbol].reindex(feature_dates).to_numpy(dtype=np.float32)
        if not np.isfinite(aligned).all():
            raise ValueError("Context date join failed")
        rows = np.flatnonzero(dataset.ticker_indices == stock_index)
        positions = feature_dates.get_indexer(pd.to_datetime(dataset.origin_dates[rows]))
        if (positions < lookback - 1).any():
            raise ValueError("Context sequence alignment failed")
        windows = np.lib.stride_tricks.sliding_window_view(aligned, lookback, axis=0).transpose(
            0, 2, 1
        )
        augmented[rows, :, dataset.sequences.shape[2] :] = windows[positions - lookback + 1]
        current[symbol] = np.concatenate(
            [dataset.current_sequences[symbol], aligned[-lookback:]], axis=1
        )
    return replace(
        dataset,
        sequences=augmented,
        current_sequences=current,
        feature_names=dataset.feature_names + CONTEXT_NAMES,
        feature_mode="price_market_context",
    )
