"""Cross-sectional candidate model family for Protocol V3.

Every candidate outputs a continuous cross-sectional ranking SCORE for each ticker at time t.
Scores need not be calibrated return forecasts — only their relative ordering matters.
No artificial scalar shrinkage is applied to ranking scores.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge

from panel.cross_sectional import V3_INTERACTION_COLUMNS, V3_RANKED_COLUMNS

# Default feature set for ML cross-sectional candidates
V3_ML_FEATURE_COLUMNS: list[str] = list(V3_RANKED_COLUMNS) + list(V3_INTERACTION_COLUMNS)


class BaseV3Candidate(ABC):
    """Abstract base class for cross-sectional ranking candidates."""

    name: str

    @abstractmethod
    def fit(
        self,
        features_by_ticker: dict[str, pd.DataFrame],
        relative_targets_by_ticker: dict[str, pd.Series],
    ) -> None:
        """Fit candidate using development tickers on training window."""

    @abstractmethod
    def predict(
        self,
        features_by_ticker: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Predict continuous ranking scores. Returns Date x Ticker DataFrame."""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name}


class MomentumRank20DCandidate(BaseV3Candidate):
    """Deterministic 20-day return cross-sectional rank score."""

    name = "momentum_rank_20d"

    def fit(
        self,
        features_by_ticker: dict[str, pd.DataFrame],
        relative_targets_by_ticker: dict[str, pd.Series],
    ) -> None:
        pass  # Pure deterministic baseline

    def predict(
        self,
        features_by_ticker: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        scores: dict[str, pd.Series] = {}
        for ticker, df in features_by_ticker.items():
            if "Return_20D_CS_Rank" in df.columns:
                scores[ticker] = df["Return_20D_CS_Rank"]
            else:
                scores[ticker] = pd.Series(np.nan, index=df.index)
        return pd.DataFrame(scores)


class MomentumRankCompositeCandidate(BaseV3Candidate):
    """Deterministic composite medium-term momentum rank score (5D, 10D, 20D)."""

    name = "momentum_rank_composite"

    def fit(
        self,
        features_by_ticker: dict[str, pd.DataFrame],
        relative_targets_by_ticker: dict[str, pd.Series],
    ) -> None:
        pass  # Pure deterministic baseline

    def predict(
        self,
        features_by_ticker: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        scores: dict[str, pd.Series] = {}
        cols = ["Return_5D_CS_Rank", "Return_10D_CS_Rank", "Return_20D_CS_Rank"]
        for ticker, df in features_by_ticker.items():
            available = [c for c in cols if c in df.columns]
            if available:
                scores[ticker] = df[available].mean(axis=1)
            else:
                scores[ticker] = pd.Series(np.nan, index=df.index)
        return pd.DataFrame(scores)


class ShortTermReversalRankCandidate(BaseV3Candidate):
    """Deterministic short-term reversal rank score (-1 * 1D, Overnight, OpenToClose)."""

    name = "short_term_reversal_rank"

    def fit(
        self,
        features_by_ticker: dict[str, pd.DataFrame],
        relative_targets_by_ticker: dict[str, pd.Series],
    ) -> None:
        pass  # Pure deterministic baseline

    def predict(
        self,
        features_by_ticker: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        scores: dict[str, pd.Series] = {}
        cols = ["Return_1D_CS_Rank", "Overnight_Return_CS_Rank", "OpenToClose_Return_CS_Rank"]
        for ticker, df in features_by_ticker.items():
            available = [c for c in cols if c in df.columns]
            if available:
                scores[ticker] = -1.0 * df[available].mean(axis=1)
            else:
                scores[ticker] = pd.Series(np.nan, index=df.index)
        return pd.DataFrame(scores)


def _prepare_tabular_data(
    features_by_ticker: dict[str, pd.DataFrame],
    relative_targets_by_ticker: dict[str, pd.Series] | None,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, list[tuple[Any, str]]]:
    """Flattens panel data across tickers for pooled tabular training."""
    x_list: list[np.ndarray] = []
    y_list: list[float] = []
    index_list: list[tuple[Any, str]] = []

    for ticker, df in features_by_ticker.items():
        avail_features = [c for c in feature_cols if c in df.columns]
        if len(avail_features) != len(feature_cols):
            continue

        feat_mat = df[feature_cols].values
        dates = df.index

        if relative_targets_by_ticker is not None and ticker in relative_targets_by_ticker:
            target_series = relative_targets_by_ticker[ticker]
            # Align on index
            aligned_targets = target_series.reindex(dates).values

            valid_mask = np.isfinite(feat_mat).all(axis=1) & np.isfinite(aligned_targets)
            if np.any(valid_mask):
                x_list.append(feat_mat[valid_mask])
                y_list.extend(aligned_targets[valid_mask].tolist())
                index_list.extend([(dates[i], ticker) for i in np.where(valid_mask)[0]])
        else:
            valid_mask = np.isfinite(feat_mat).all(axis=1)
            if np.any(valid_mask):
                x_list.append(feat_mat[valid_mask])
                index_list.extend([(dates[i], ticker) for i in np.where(valid_mask)[0]])

    x_arr = np.vstack(x_list) if x_list else np.empty((0, len(feature_cols)), dtype=float)
    y_arr = np.asarray(y_list, dtype=float) if y_list else np.empty((0,), dtype=float)
    return x_arr, y_arr, index_list


class RidgeCrossSectionalCandidate(BaseV3Candidate):
    """Ridge regression on cross-sectional rank features."""

    name = "ridge_cross_sectional"

    def __init__(
        self,
        alpha: float = 100.0,
        feature_cols: list[str] | None = None,
    ):
        self.alpha = alpha
        self.feature_cols = feature_cols or V3_ML_FEATURE_COLUMNS
        self.model = Ridge(alpha=self.alpha, fit_intercept=True)
        self.is_fitted = False

    def fit(
        self,
        features_by_ticker: dict[str, pd.DataFrame],
        relative_targets_by_ticker: dict[str, pd.Series],
    ) -> None:
        x_mat, y_vec, _ = _prepare_tabular_data(
            features_by_ticker, relative_targets_by_ticker, self.feature_cols
        )
        if len(x_mat) >= 100 and len(y_vec) == len(x_mat):
            self.model.fit(x_mat, y_vec)
            self.is_fitted = True
        else:
            self.is_fitted = False

    def predict(
        self,
        features_by_ticker: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        scores: dict[str, pd.Series] = {}
        for ticker, df in features_by_ticker.items():
            avail = [c for c in self.feature_cols if c in df.columns]
            if not self.is_fitted or len(avail) != len(self.feature_cols):
                scores[ticker] = pd.Series(np.nan, index=df.index)
                continue

            x_mat = df[self.feature_cols].values
            valid_mask = np.isfinite(x_mat).all(axis=1)
            pred_vec = np.full(len(df), np.nan, dtype=float)
            if np.any(valid_mask):
                pred_vec[valid_mask] = self.model.predict(x_mat[valid_mask])
            scores[ticker] = pd.Series(pred_vec, index=df.index)
        return pd.DataFrame(scores)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "alpha": self.alpha,
            "feature_cols": self.feature_cols,
        }


class ElasticNetCrossSectionalCandidate(BaseV3Candidate):
    """ElasticNet regression on cross-sectional rank features."""

    name = "elastic_net_cross_sectional"

    def __init__(
        self,
        alpha: float = 0.01,
        l1_ratio: float = 0.5,
        feature_cols: list[str] | None = None,
        seed: int = 42,
    ):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.feature_cols = feature_cols or V3_ML_FEATURE_COLUMNS
        self.seed = seed
        self.model = ElasticNet(
            alpha=self.alpha, l1_ratio=self.l1_ratio, random_state=self.seed, max_iter=2000
        )
        self.is_fitted = False

    def fit(
        self,
        features_by_ticker: dict[str, pd.DataFrame],
        relative_targets_by_ticker: dict[str, pd.Series],
    ) -> None:
        x_mat, y_vec, _ = _prepare_tabular_data(
            features_by_ticker, relative_targets_by_ticker, self.feature_cols
        )
        if len(x_mat) >= 100 and len(y_vec) == len(x_mat):
            self.model.fit(x_mat, y_vec)
            self.is_fitted = True
        else:
            self.is_fitted = False

    def predict(
        self,
        features_by_ticker: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        scores: dict[str, pd.Series] = {}
        for ticker, df in features_by_ticker.items():
            avail = [c for c in self.feature_cols if c in df.columns]
            if not self.is_fitted or len(avail) != len(self.feature_cols):
                scores[ticker] = pd.Series(np.nan, index=df.index)
                continue

            x_mat = df[self.feature_cols].values
            valid_mask = np.isfinite(x_mat).all(axis=1)
            pred_vec = np.full(len(df), np.nan, dtype=float)
            if np.any(valid_mask):
                pred_vec[valid_mask] = self.model.predict(x_mat[valid_mask])
            scores[ticker] = pd.Series(pred_vec, index=df.index)
        return pd.DataFrame(scores)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "alpha": self.alpha,
            "l1_ratio": self.l1_ratio,
            "seed": self.seed,
            "feature_cols": self.feature_cols,
        }


class HistGradientBoostCrossSectionalCandidate(BaseV3Candidate):
    """Histogram Gradient Boosted Trees for cross-sectional ranking."""

    name = "hist_gradient_boost_cross_sectional"

    def __init__(
        self,
        max_iter: int = 50,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        min_samples_leaf: int = 50,
        feature_cols: list[str] | None = None,
        seed: int = 42,
    ):
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.feature_cols = feature_cols or V3_ML_FEATURE_COLUMNS
        self.seed = seed
        self.model = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.seed,
        )
        self.is_fitted = False

    def fit(
        self,
        features_by_ticker: dict[str, pd.DataFrame],
        relative_targets_by_ticker: dict[str, pd.Series],
    ) -> None:
        x_mat, y_vec, _ = _prepare_tabular_data(
            features_by_ticker, relative_targets_by_ticker, self.feature_cols
        )
        if len(x_mat) >= 100 and len(y_vec) == len(x_mat):
            self.model.fit(x_mat, y_vec)
            self.is_fitted = True
        else:
            self.is_fitted = False

    def predict(
        self,
        features_by_ticker: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        scores: dict[str, pd.Series] = {}
        for ticker, df in features_by_ticker.items():
            avail = [c for c in self.feature_cols if c in df.columns]
            if not self.is_fitted or len(avail) != len(self.feature_cols):
                scores[ticker] = pd.Series(np.nan, index=df.index)
                continue

            x_mat = df[self.feature_cols].values
            valid_mask = np.isfinite(x_mat).all(axis=1)
            pred_vec = np.full(len(df), np.nan, dtype=float)
            if np.any(valid_mask):
                pred_vec[valid_mask] = self.model.predict(x_mat[valid_mask])
            scores[ticker] = pd.Series(pred_vec, index=df.index)
        return pd.DataFrame(scores)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "max_iter": self.max_iter,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "min_samples_leaf": self.min_samples_leaf,
            "seed": self.seed,
            "feature_cols": self.feature_cols,
        }


class DLinearCrossSectionalCandidate(BaseV3Candidate):
    """Direct Linear cross-sectional model with Trend & Remainder decomposition."""

    name = "dlinear_cross_sectional"

    def __init__(
        self,
        feature_cols: list[str] | None = None,
        alpha: float = 10.0,
    ):
        self.feature_cols = feature_cols or V3_ML_FEATURE_COLUMNS
        self.alpha = alpha
        self.trend_model = Ridge(alpha=self.alpha, fit_intercept=True)
        self.rem_model = Ridge(alpha=self.alpha, fit_intercept=True)
        self.is_fitted = False

    def fit(
        self,
        features_by_ticker: dict[str, pd.DataFrame],
        relative_targets_by_ticker: dict[str, pd.Series],
    ) -> None:
        x_mat, y_vec, _ = _prepare_tabular_data(
            features_by_ticker, relative_targets_by_ticker, self.feature_cols
        )
        if len(x_mat) >= 100 and len(y_vec) == len(x_mat):
            # Split features into trend (returns) and remainder (vol/liq)
            trend_cols = [c for c in self.feature_cols if "Return" in c or "Streak" in c]
            rem_cols = [c for c in self.feature_cols if c not in trend_cols]

            trend_idx = [self.feature_cols.index(c) for c in trend_cols]
            rem_idx = [self.feature_cols.index(c) for c in rem_cols]

            if trend_idx and rem_idx:
                self.trend_model.fit(x_mat[:, trend_idx], y_vec * 0.5)
                self.rem_model.fit(x_mat[:, rem_idx], y_vec * 0.5)
                self.is_fitted = True
            else:
                self.trend_model.fit(x_mat, y_vec)
                self.is_fitted = True
        else:
            self.is_fitted = False

    def predict(
        self,
        features_by_ticker: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        scores: dict[str, pd.Series] = {}
        trend_cols = [c for c in self.feature_cols if "Return" in c or "Streak" in c]
        rem_cols = [c for c in self.feature_cols if c not in trend_cols]

        trend_idx = [self.feature_cols.index(c) for c in trend_cols]
        rem_idx = [self.feature_cols.index(c) for c in rem_cols]

        for ticker, df in features_by_ticker.items():
            avail = [c for c in self.feature_cols if c in df.columns]
            if not self.is_fitted or len(avail) != len(self.feature_cols):
                scores[ticker] = pd.Series(np.nan, index=df.index)
                continue

            x_mat = df[self.feature_cols].values
            valid_mask = np.isfinite(x_mat).all(axis=1)
            pred_vec = np.full(len(df), np.nan, dtype=float)
            if np.any(valid_mask):
                if trend_idx and rem_idx:
                    p_trend = self.trend_model.predict(x_mat[valid_mask][:, trend_idx])
                    p_rem = self.rem_model.predict(x_mat[valid_mask][:, rem_idx])
                    pred_vec[valid_mask] = p_trend + p_rem
                else:
                    pred_vec[valid_mask] = self.trend_model.predict(x_mat[valid_mask])
            scores[ticker] = pd.Series(pred_vec, index=df.index)
        return pd.DataFrame(scores)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "alpha": self.alpha,
            "feature_cols": self.feature_cols,
        }


V3_CANDIDATE_REGISTRY: dict[str, type[BaseV3Candidate]] = {
    "momentum_rank_20d": MomentumRank20DCandidate,
    "momentum_rank_composite": MomentumRankCompositeCandidate,
    "short_term_reversal_rank": ShortTermReversalRankCandidate,
    "ridge_cross_sectional": RidgeCrossSectionalCandidate,
    "elastic_net_cross_sectional": ElasticNetCrossSectionalCandidate,
    "hist_gradient_boost_cross_sectional": HistGradientBoostCrossSectionalCandidate,
    "dlinear_cross_sectional": DLinearCrossSectionalCandidate,
}
