"""Universal point-in-time multi-asset panel builder for historical 70/15/15 multimodal model training."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from backend.contracts.schemas_v11 import (
    MULTIMODAL_NEWS_FEATURE_COLUMNS_V11,
    REQUIRED_TARGET_HORIZONS_V11,
)
from research.volatility_forecasting.multimodal_features_v2 import (
    EnrichedFeatureExtractor,
)
from research.volatility_forecasting.news_aggregator_v2 import (
    EnrichedNewsArticle,
    MultiDimensionalNewsAggregator,
)


@dataclass(frozen=True)
class HistoricalPanelDataset:
    dates: list[str]
    security_ids: list[str]
    numeric_features: np.ndarray  # [N, 34]
    news_features: np.ndarray  # [N, 19]
    shuffled_news_negative_control: np.ndarray  # [N, 19]
    returns_targets: np.ndarray  # [N, 4] for h in (1, 3, 5, 7)
    rv_targets: np.ndarray  # [N, 4] for h in (1, 3, 5, 7)
    panel_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_samples": len(self.dates),
            "unique_dates": len(set(self.dates)),
            "unique_securities": len(set(self.security_ids)),
            "numeric_shape": list(self.numeric_features.shape),
            "news_shape": list(self.news_features.shape),
            "targets_shape": list(self.returns_targets.shape),
            "panel_sha256": self.panel_sha256,
        }


class HistoricalPITDatasetBuilderV11:
    """Builds point-in-time panel datasets across target equities and exogenous market/sector contexts."""

    @staticmethod
    def construct_panel_from_series(
        equities_ohlcv: dict[str, pd.DataFrame],
        sector_ohlcv: pd.DataFrame,
        market_ohlcv: pd.DataFrame,
        news_articles: list[EnrichedNewsArticle] | None = None,
        horizons: tuple[int, ...] = REQUIRED_TARGET_HORIZONS_V11,
        warmup_sessions: int = 65,
    ) -> HistoricalPanelDataset:
        all_dates: list[str] = []
        all_sec_ids: list[str] = []
        all_num_feats: list[np.ndarray] = []
        all_news_feats: list[np.ndarray] = []
        all_rets: list[np.ndarray] = []
        all_rv: list[np.ndarray] = []

        max_h = max(horizons)

        for sec_id, df in equities_ohlcv.items():
            df_sorted = df.copy().sort_index()
            c = df_sorted["Close"].to_numpy(dtype=float)
            dates = [str(d)[:10] for d in df_sorted.index]
            n = len(c)

            if n <= warmup_sessions + max_h:
                continue

            # Compute daily log returns for target calculations
            daily_rets = np.log(c[1:] / c[:-1])

            # Loop over valid historical origin sessions
            for t_idx in range(warmup_sessions, n - max_h):
                t_date = dates[t_idx]
                history_df = df_sorted.iloc[: t_idx + 1]

                # Align context data strictly up to t_date
                sec_sub = sector_ohlcv.loc[:t_date] if sector_ohlcv is not None else None
                mkt_sub = market_ohlcv.loc[:t_date] if market_ohlcv is not None else None

                # 1. Extract 34 Numeric Features
                feats = EnrichedFeatureExtractor.extract_from_series(
                    target_df=history_df, sector_df=sec_sub, market_df=mkt_sub
                )
                num_arr = feats.to_array()

                # 2. Extract 19 Causal News Features
                if news_articles:
                    news_sub = [
                        a
                        for a in news_articles
                        if (
                            a.available_at[:10] <= t_date
                            and (a.ticker_relevance > 0.5 or a.event_type == "macro")
                        )
                    ]
                    news_agg = MultiDimensionalNewsAggregator.aggregate_causal_window(
                        articles=news_sub, cutoff_iso=f"{t_date}T20:00:00Z"
                    )
                    news_arr = news_agg.to_array()
                else:
                    news_arr = np.zeros(len(MULTIMODAL_NEWS_FEATURE_COLUMNS_V11), dtype=float)

                # 3. Compute Targets for horizons in (1, 3, 5, 7)
                # Cumulative log return: log(P_{t+h} / P_t)
                # Realized variance: sum of squared daily log returns
                h_rets = []
                h_rv = []
                p0 = c[t_idx]
                for h in horizons:
                    p_h = c[t_idx + h]
                    cum_ret = float(np.log(p_h / p0))
                    # Daily returns from t_idx to t_idx + h
                    step_rets = daily_rets[t_idx : t_idx + h]
                    cum_rv = float(np.sum(step_rets**2))
                    h_rets.append(cum_ret)
                    h_rv.append(cum_rv)

                all_dates.append(t_date)
                all_sec_ids.append(sec_id)
                all_num_feats.append(num_arr)
                all_news_feats.append(news_arr)
                all_rets.append(np.array(h_rets, dtype=float))
                all_rv.append(np.array(h_rv, dtype=float))

        # Sort chronologically
        sort_order = np.argsort(all_dates)
        dates_sorted = [all_dates[i] for i in sort_order]
        sec_ids_sorted = [all_sec_ids[i] for i in sort_order]
        num_mat = np.array(all_num_feats, dtype=float)[sort_order]
        news_mat = np.array(all_news_feats, dtype=float)[sort_order]
        rets_mat = np.array(all_rets, dtype=float)[sort_order]
        rv_mat = np.array(all_rv, dtype=float)[sort_order]

        # 4. Generate Shuffled Negative Control for News (M3)
        rng = np.random.default_rng(1337)
        shuffled_news = rng.permutation(news_mat)

        # 5. Compute SHA-256 Digest of Panel
        raw_sig = f"{len(dates_sorted)}:{num_mat.shape}:{news_mat.shape}:{rets_mat.shape}"
        panel_digest = hashlib.sha256(raw_sig.encode()).hexdigest()

        return HistoricalPanelDataset(
            dates=dates_sorted,
            security_ids=sec_ids_sorted,
            numeric_features=num_mat,
            news_features=news_mat,
            shuffled_news_negative_control=shuffled_news,
            returns_targets=rets_mat,
            rv_targets=rv_mat,
            panel_sha256=panel_digest,
        )
