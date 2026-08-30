"""Universal point-in-time multi-asset panel builder with fail-closed membership masks and causal controls."""

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
    same_origin_shuffled_news: np.ndarray  # [N, 19] M3 Control A
    causal_delayed_news: np.ndarray  # [N, 19] M3 Control B (10-session delay)
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
    """Constructs point-in-time historical datasets with fail-closed membership masks and content hashing."""

    @staticmethod
    def is_active_member(
        date_str: str,
        intervals: list[tuple[str, str]],
    ) -> bool:
        """Checks if date_str falls within any active membership interval (start, end)."""
        return any(start <= date_str <= end for start, end in intervals)

    @classmethod
    def construct_panel_from_series(
        cls,
        equities_ohlcv: dict[str, pd.DataFrame],
        sector_ohlcv: pd.DataFrame,
        market_ohlcv: pd.DataFrame,
        news_articles: list[EnrichedNewsArticle] | None = None,
        membership_masks: dict[str, list[tuple[str, str]]]
        | None = None,  # ticker -> [(start, end), ...]
        horizons: tuple[int, ...] = REQUIRED_TARGET_HORIZONS_V11,
        warmup_sessions: int = 65,
    ) -> HistoricalPanelDataset:
        all_dates: list[str] = []
        all_sec_ids: list[str] = []
        all_num_feats: list[np.ndarray] = []
        all_news_feats: list[np.ndarray] = []
        all_delayed_news_feats: list[np.ndarray] = []
        all_rets: list[np.ndarray] = []
        all_rv: list[np.ndarray] = []

        max_h = max(horizons)

        for sec_id, df in equities_ohlcv.items():
            # If membership mask supplied, reject if security is not in the mask (fail-closed)
            if membership_masks is not None and sec_id not in membership_masks:
                continue

            df_sorted = df.copy().sort_index()
            c = df_sorted["Close"].to_numpy(dtype=float)
            dates = [str(d)[:10] for d in df_sorted.index]
            n = len(c)

            if n <= warmup_sessions + max_h:
                continue

            daily_rets = np.log(c[1:] / c[:-1])

            # Iterate over causal origin sessions
            for t_idx in range(warmup_sessions, n - max_h):
                t_date = dates[t_idx]

                # Check PIT multi-interval membership (fail closed)
                if membership_masks is not None:
                    intervals = membership_masks[sec_id]
                    if not cls.is_active_member(t_date, intervals):
                        continue

                history_df = df_sorted.iloc[: t_idx + 1]
                sec_sub = sector_ohlcv.loc[:t_date] if sector_ohlcv is not None else None
                mkt_sub = market_ohlcv.loc[:t_date] if market_ohlcv is not None else None

                # 1. 34 Numeric Features
                feats = EnrichedFeatureExtractor.extract_from_series(
                    target_df=history_df, sector_df=sec_sub, market_df=mkt_sub
                )
                num_arr = feats.to_array()

                # 2. 19 Causal News Features (strictly <= t_date and bound to sec_id)
                if news_articles:
                    news_agg = MultiDimensionalNewsAggregator.aggregate_causal_window(
                        articles=news_articles,
                        target_ticker=sec_id,
                        cutoff_iso=f"{t_date}T20:00:00Z",
                    )
                    news_arr = news_agg.to_array()

                    # Causal Delayed Control B: News from 10 sessions prior
                    delayed_t_idx = max(0, t_idx - 10)
                    delayed_date = dates[delayed_t_idx]
                    delayed_agg = MultiDimensionalNewsAggregator.aggregate_causal_window(
                        articles=news_articles,
                        target_ticker=sec_id,
                        cutoff_iso=f"{delayed_date}T20:00:00Z",
                    )
                    delayed_news_arr = delayed_agg.to_array()
                else:
                    news_arr = np.zeros(len(MULTIMODAL_NEWS_FEATURE_COLUMNS_V11), dtype=float)
                    delayed_news_arr = np.zeros(
                        len(MULTIMODAL_NEWS_FEATURE_COLUMNS_V11), dtype=float
                    )

                # 3. Target Vectors for Horizons in (1, 3, 5, 7)
                h_rets = []
                h_rv = []
                p0 = c[t_idx]
                for h in horizons:
                    p_h = c[t_idx + h]
                    cum_ret = float(np.log(p_h / p0))
                    step_rets = daily_rets[t_idx : t_idx + h]
                    cum_rv = float(np.sum(step_rets**2))
                    h_rets.append(cum_ret)
                    h_rv.append(cum_rv)

                all_dates.append(t_date)
                all_sec_ids.append(sec_id)
                all_num_feats.append(num_arr)
                all_news_feats.append(news_arr)
                all_delayed_news_feats.append(delayed_news_arr)
                all_rets.append(np.array(h_rets, dtype=float))
                all_rv.append(np.array(h_rv, dtype=float))

        # Sort chronologically by date, then security_id
        sort_order = np.lexsort((all_sec_ids, all_dates))
        dates_sorted = [all_dates[i] for i in sort_order]
        sec_ids_sorted = [all_sec_ids[i] for i in sort_order]
        num_mat = np.array(all_num_feats, dtype=float)[sort_order]
        news_mat = np.array(all_news_feats, dtype=float)[sort_order]
        delayed_news_mat = np.array(all_delayed_news_feats, dtype=float)[sort_order]
        rets_mat = np.array(all_rets, dtype=float)[sort_order]
        rv_mat = np.array(all_rv, dtype=float)[sort_order]

        # 4. Generate Causal Same-Origin Cross-Sectional Shuffle (M3 Control A)
        same_origin_shuffled = news_mat.copy()
        df_group = pd.DataFrame({"date": dates_sorted, "idx": np.arange(len(dates_sorted))})
        rng = np.random.default_rng(2026)
        for _, grp in df_group.groupby("date"):
            grp_indices = grp["idx"].to_numpy()
            if len(grp_indices) > 1:
                shuffled_idx = rng.permutation(grp_indices)
                same_origin_shuffled[grp_indices] = news_mat[shuffled_idx]

        # 5. Content-Addressed SHA-256 Digest of Full Dataset
        h = hashlib.sha256()
        h.update(f"{','.join(dates_sorted)}".encode())
        h.update(f"{','.join(sec_ids_sorted)}".encode())
        h.update(num_mat.tobytes())
        h.update(news_mat.tobytes())
        h.update(rets_mat.tobytes())
        h.update(rv_mat.tobytes())
        panel_digest = h.hexdigest()

        return HistoricalPanelDataset(
            dates=dates_sorted,
            security_ids=sec_ids_sorted,
            numeric_features=num_mat,
            news_features=news_mat,
            same_origin_shuffled_news=same_origin_shuffled,
            causal_delayed_news=delayed_news_mat,
            returns_targets=rets_mat,
            rv_targets=rv_mat,
            panel_sha256=panel_digest,
        )
