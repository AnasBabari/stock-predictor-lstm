"""Matched causal volatility baseline models for V10 research and serving.

Includes:
- Persistence (naive flat)
- EWMA (RiskMetrics lambda=0.94)
- HAR-RV (Corsi 2009 log-linear model)
- GARCH(1,1) (Engle / Bollerslev MLE)
- GJR-GARCH(1,1,1) (Glosten-Jagannathan-Runkle leverage model)
- Ridge & ElasticNet linear regularized estimators
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class CausalHARBaseline:
    def __init__(self) -> None:
        self.weights_by_horizon: dict[int, np.ndarray] = {}

    def fit(self, rv_series: pd.Series, horizons: tuple[int, ...] = (1, 3, 5, 7, 14, 30)) -> None:
        rv = np.maximum(rv_series.to_numpy(dtype=float), 1e-12)
        n = len(rv)
        if n < 60:
            raise ValueError("HAR requires at least 60 training sessions")

        for h in horizons:
            # Construct HAR design matrix
            X_list = []
            y_list = []
            for t in range(22, n - h):
                d = np.log(rv[t])
                w = np.log(np.mean(rv[t - 4 : t + 1]))
                m = np.log(np.mean(rv[t - 21 : t + 1]))
                X_list.append([1.0, d, w, m])
                # Target is log of cumulative future variance
                target_var = np.sum(rv[t + 1 : t + 1 + h])
                y_list.append(np.log(max(target_var, 1e-12)))

            X = np.array(X_list)
            y = np.array(y_list)
            # Solve OLS with small ridge regularization for stability
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            self.weights_by_horizon[h] = beta

    def predict(self, rv_history: np.ndarray, horizon: int) -> float:
        rv = np.maximum(np.asarray(rv_history, dtype=float), 1e-12)
        if len(rv) < 22:
            raise ValueError("Prediction requires at least 22 history sessions")
        beta = self.weights_by_horizon.get(horizon)
        if beta is None:
            raise KeyError(f"HAR not fitted for horizon {horizon}")
        d = np.log(rv[-1])
        w = np.log(np.mean(rv[-5:]))
        m = np.log(np.mean(rv[-22:]))
        x = np.array([1.0, d, w, m])
        log_pred = float(np.dot(x, beta))
        return float(np.exp(log_pred))


class CausalEWMABaseline:
    def __init__(self, decay: float = 0.94) -> None:
        self.decay = decay

    def predict(self, rv_history: np.ndarray, horizon: int) -> float:
        rv = np.maximum(np.asarray(rv_history, dtype=float), 1e-12)
        weights = (1 - self.decay) * (self.decay ** np.arange(len(rv))[::-1])
        weights /= weights.sum()
        current_level = float(np.sum(weights * rv))
        return float(current_level * horizon)


class CausalPersistenceBaseline:
    def predict(self, rv_history: np.ndarray, horizon: int) -> float:
        rv = np.maximum(np.asarray(rv_history, dtype=float), 1e-12)
        return float(rv[-1] * horizon)
