"""Matched baselines suite for return distribution, direction, and volatility forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge


@dataclass(frozen=True)
class BaselinePredictionResult:
    family: str
    target_contract: str
    horizon: int
    predictions: np.ndarray  # (N,) array
    metadata: dict[str, Any]


class MatchedReturnBaselines:
    """Matched baselines for cumulative log-return prediction."""

    @staticmethod
    def zero_return(n_samples: int, horizon: int) -> BaselinePredictionResult:
        """Zero cumulative return benchmark: R_{t,h} = 0."""
        preds = np.zeros(n_samples, dtype=float)
        return BaselinePredictionResult(
            family="zero_return",
            target_contract="price-return-distribution-v1",
            horizon=horizon,
            predictions=preds,
            metadata={"description": "Zero return / random walk martingale benchmark"},
        )

    @staticmethod
    def historical_mean_return(
        train_returns: np.ndarray, n_samples: int, horizon: int, shrinkage: float = 0.5
    ) -> BaselinePredictionResult:
        """Historical mean daily return scaled to horizon with shrinkage."""
        daily_mean = float(np.mean(train_returns)) if len(train_returns) > 0 else 0.0
        shrunk_mean = daily_mean * shrinkage
        preds = np.full(n_samples, shrunk_mean * horizon, dtype=float)
        return BaselinePredictionResult(
            family="historical_mean",
            target_contract="price-return-distribution-v1",
            horizon=horizon,
            predictions=preds,
            metadata={"daily_mean": daily_mean, "shrunk_mean": shrunk_mean},
        )

    @staticmethod
    def rolling_drift(
        historical_returns: np.ndarray, horizon: int, window: int = 20
    ) -> BaselinePredictionResult:
        """Rolling window drift return benchmark."""
        n = len(historical_returns)
        preds = np.zeros(n, dtype=float)
        for i in range(n):
            start = max(0, i - window + 1)
            mean_ret = float(np.mean(historical_returns[start : i + 1]))
            preds[i] = mean_ret * horizon
        return BaselinePredictionResult(
            family="rolling_drift",
            target_contract="price-return-distribution-v1",
            horizon=horizon,
            predictions=preds,
            metadata={"window": window},
        )

    @staticmethod
    def ridge_return(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_eval: np.ndarray,
        horizon: int,
        alpha: float = 1.0,
    ) -> BaselinePredictionResult:
        """Ridge linear regression baseline for multi-horizon cumulative return."""
        # Flatten sequence input (N, L, F) -> (N, L*F)
        N_tr = X_train.shape[0]
        N_ev = X_eval.shape[0]
        X_tr_2d = X_train.reshape(N_tr, -1)
        X_ev_2d = X_eval.reshape(N_ev, -1)

        model = Ridge(alpha=alpha)
        model.fit(X_tr_2d, y_train)
        preds = model.predict(X_ev_2d).astype(float)
        return BaselinePredictionResult(
            family="ridge",
            target_contract="price-return-distribution-v1",
            horizon=horizon,
            predictions=preds,
            metadata={"alpha": alpha, "coef_norm": float(np.linalg.norm(model.coef_))},
        )


class MatchedDirectionBaselines:
    """Matched baselines for cumulative three-way direction forecasting."""

    @staticmethod
    def majority_class(
        train_labels: np.ndarray, n_samples: int, horizon: int
    ) -> BaselinePredictionResult:
        """Pre-evaluation majority class baseline (-1, 0, or 1)."""
        unique, counts = np.unique(train_labels, return_counts=True)
        majority = int(unique[np.argmax(counts)]) if len(unique) > 0 else 0
        preds = np.full(n_samples, majority, dtype=int)
        return BaselinePredictionResult(
            family="majority_class",
            target_contract="direction-cumulative-three-way-v2",
            horizon=horizon,
            predictions=preds,
            metadata={"majority_label": majority},
        )

    @staticmethod
    def momentum_sign(lagged_returns: np.ndarray, horizon: int) -> BaselinePredictionResult:
        """Sign of previous horizon return: sign(R_{t-h, h})."""
        signs = np.sign(lagged_returns).astype(int)
        return BaselinePredictionResult(
            family="momentum_sign",
            target_contract="direction-cumulative-three-way-v2",
            horizon=horizon,
            predictions=signs,
            metadata={"description": "Lagged momentum direction baseline"},
        )


class MatchedVolatilityBaselines:
    """Matched baselines for future cumulative realized variance forecasting."""

    @staticmethod
    def persistence(daily_variances: np.ndarray, horizon: int) -> BaselinePredictionResult:
        """Persistence benchmark: V_{t,h} = h * v_t."""
        preds = np.maximum(daily_variances * horizon, 1e-8).astype(float)
        return BaselinePredictionResult(
            family="persistence",
            target_contract="future-rv-total-v2",
            horizon=horizon,
            predictions=preds,
            metadata={"description": "Random walk persistence variance benchmark"},
        )

    @staticmethod
    def ewma_volatility(
        daily_variances: np.ndarray, horizon: int, decay: float = 0.94
    ) -> BaselinePredictionResult:
        """RiskMetrics EWMA variance benchmark: V_{t,h} = h * EWMA(v_t)."""
        n = len(daily_variances)
        ewma_arr = np.zeros(n, dtype=float)
        current = daily_variances[0]
        for t in range(n):
            current = decay * current + (1.0 - decay) * daily_variances[t]
            ewma_arr[t] = current * horizon
        return BaselinePredictionResult(
            family="ewma",
            target_contract="future-rv-total-v2",
            horizon=horizon,
            predictions=ewma_arr,
            metadata={"decay": decay},
        )

    @staticmethod
    def har_rv(daily_variances: np.ndarray, horizon: int) -> BaselinePredictionResult:
        """Heterogeneous Autoregressive (HAR-RV) model: Daily, Weekly, Monthly lags."""
        n = len(daily_variances)
        preds = np.zeros(n, dtype=float)
        for t in range(n):
            v_d = daily_variances[t]
            v_w = np.mean(daily_variances[max(0, t - 4) : t + 1])
            v_m = np.mean(daily_variances[max(0, t - 21) : t + 1])
            # Classical HAR weights normalized to horizon
            est_daily = 0.1 * v_d + 0.5 * v_w + 0.4 * v_m
            preds[t] = max(1e-8, est_daily * horizon)
        return BaselinePredictionResult(
            family="har",
            target_contract="future-rv-total-v2",
            horizon=horizon,
            predictions=preds,
            metadata={"description": "Heterogeneous Autoregressive Realized Variance"},
        )
