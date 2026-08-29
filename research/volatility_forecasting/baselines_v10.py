"""Matched causal volatility baseline suite for StockLSTM V10.

Implements all preregistered causal baseline competitors:
1. PersistenceBaseline: V(t, h) = h * daily_variance_t
2. EWMABaseline: Exponentially weighted moving average variance
3. HARRVBaseline: Heterogeneous Autoregressive model of Realized Variance (HAR-RV)
4. GARCHBaseline: Classical GARCH(1,1) recursive forward term structure
5. GJRGARCHBaseline: Glosten-Jagannathan-Runkle GARCH with leverage effect
6. RidgeVolatilityBaseline: L2-regularized linear model on deployable features
7. ElasticNetVolatilityBaseline: L1/L2 regularized linear model on deployable features
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import ElasticNet, Ridge


@dataclass(frozen=True)
class PersistenceBaseline:
    """Carries forward the latest daily variance scaled by horizon h."""

    def predict(self, latest_daily_variance: float, horizon: int) -> float:
        if horizon <= 0:
            raise ValueError("Horizon must be positive.")
        return max(float(latest_daily_variance) * horizon, 1e-12)


@dataclass(frozen=True)
class EWMABaseline:
    """Causal EWMA variance forecaster."""

    lambda_param: float = 0.94

    def predict(self, variance_series: np.ndarray | list[float], horizon: int) -> float:
        arr = np.asarray(variance_series, dtype=np.float64)
        if len(arr) == 0:
            raise ValueError("Variance series cannot be empty.")
        weights = (1.0 - self.lambda_param) * (self.lambda_param ** np.arange(len(arr) - 1, -1, -1))
        weights /= weights.sum()
        current_daily_ewma = float(np.sum(weights * arr))
        return max(current_daily_ewma * horizon, 1e-12)


class HARRVBaseline:
    """Heterogeneous Autoregressive model of Realized Variance (Corsi 2009)."""

    def __init__(self) -> None:
        self.beta_0: float = 0.0
        self.beta_d: float = 0.0
        self.beta_w: float = 0.0
        self.beta_m: float = 0.0
        self.is_fitted: bool = False

    def fit(self, daily_variances: np.ndarray | list[float]) -> HARRVBaseline:
        arr = np.asarray(daily_variances, dtype=np.float64)
        if len(arr) < 30:
            raise ValueError(f"HAR-RV requires at least 30 observations to fit, got {len(arr)}")

        rv_d = arr[21:-1]
        rv_w = np.array([np.mean(arr[i - 5 : i]) for i in range(22, len(arr))])
        rv_m = np.array([np.mean(arr[i - 22 : i]) for i in range(22, len(arr))])
        y = arr[22:]

        X = np.column_stack([np.ones(len(y)), rv_d, rv_w, rv_m])
        coefs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

        self.beta_0 = float(max(coefs[0], 0.0))
        self.beta_d = float(max(coefs[1], 0.0))
        self.beta_w = float(max(coefs[2], 0.0))
        self.beta_m = float(max(coefs[3], 0.0))
        self.is_fitted = True
        return self

    def predict(self, recent_daily_variances: np.ndarray | list[float], horizon: int) -> float:
        if not self.is_fitted:
            raise ValueError("HARRVBaseline must be fitted before predict.")
        arr = np.asarray(recent_daily_variances, dtype=np.float64)
        if len(arr) < 22:
            raise ValueError(f"HAR-RV prediction requires at least 22 recent days, got {len(arr)}")

        rv_d = arr[-1]
        rv_w = np.mean(arr[-5:])
        rv_m = np.mean(arr[-22:])

        daily_pred = self.beta_0 + self.beta_d * rv_d + self.beta_w * rv_w + self.beta_m * rv_m
        daily_pred = max(float(daily_pred), 1e-12)
        return daily_pred * horizon


class GARCHBaseline:
    """GARCH(1,1) recursive term-structure forecaster."""

    def __init__(self, omega: float = 1e-6, alpha: float = 0.05, beta: float = 0.90) -> None:
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.long_run_var = omega / max(1.0 - alpha - beta, 1e-4)

    def fit(self, returns: np.ndarray | list[float]) -> GARCHBaseline:
        # Stationary calibration fallback to sample empirical moments
        arr = np.asarray(returns, dtype=np.float64)
        sample_var = float(np.var(arr)) if len(arr) > 1 else 1e-4
        self.long_run_var = max(sample_var, 1e-8)
        self.omega = self.long_run_var * max(1.0 - self.alpha - self.beta, 1e-4)
        return self

    def predict(self, current_variance: float, horizon: int) -> float:
        persistence = self.alpha + self.beta
        cum_var = 0.0
        h_t = current_variance
        for _ in range(horizon):
            h_t = self.omega + persistence * h_t
            cum_var += max(h_t, 1e-12)
        return max(float(cum_var), 1e-12)


class GJRGARCHBaseline:
    """GJR-GARCH asymmetric leverage forecaster."""

    def __init__(
        self, omega: float = 1e-6, alpha: float = 0.03, gamma: float = 0.08, beta: float = 0.88
    ) -> None:
        self.omega = omega
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta

    def predict(self, current_variance: float, horizon: int) -> float:
        # Unconditional expected asymmetric term is gamma / 2 for symmetric zero-mean innovations
        persistence = self.alpha + self.gamma / 2.0 + self.beta
        cum_var = 0.0
        h_t = current_variance
        for _ in range(horizon):
            h_t = self.omega + persistence * h_t
            cum_var += max(h_t, 1e-12)
        return max(float(cum_var), 1e-12)


class RidgeVolatilityBaseline:
    """L2-regularized linear baseline over stationary feature matrix."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.model = Ridge(alpha=alpha, positive=False)
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> RidgeVolatilityBaseline:
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("RidgeVolatilityBaseline must be fitted before predict.")
        preds = self.model.predict(X)
        return np.maximum(preds, 1e-12)


class ElasticNetVolatilityBaseline:
    """L1/L2-regularized linear baseline over stationary feature matrix."""

    def __init__(self, alpha: float = 0.1, l1_ratio: float = 0.5) -> None:
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=2000)
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> ElasticNetVolatilityBaseline:
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("ElasticNetVolatilityBaseline must be fitted before predict.")
        preds = self.model.predict(X)
        return np.maximum(preds, 1e-12)
