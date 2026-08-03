"""Deterministic candidate model implementations for stock autoresearch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor


class Candidate:
    name = "candidate"

    def fit(self, x: np.ndarray, y: np.ndarray) -> Candidate:
        raise NotImplementedError

    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {"family": self.name}

    def parameter_count(self) -> int:
        return 0


@dataclass
class PersistenceCandidate(Candidate):
    name: str = "persistence"

    def fit(self, x: np.ndarray, y: np.ndarray) -> PersistenceCandidate:
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.zeros(len(x), dtype=np.float64)

    def describe(self) -> dict[str, Any]:
        return {"family": self.name}

    def parameter_count(self) -> int:
        return 0


@dataclass
class RidgeCandidate(Candidate):
    alpha: float = 10.0
    name: str = "ridge"

    def __post_init__(self) -> None:
        self._model = Ridge(alpha=self.alpha)

    def fit(self, x: np.ndarray, y: np.ndarray) -> RidgeCandidate:
        self._model.fit(x[:, -1, :], y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self._model.predict(x[:, -1, :]), dtype=np.float64)

    def describe(self) -> dict[str, Any]:
        return {"family": self.name, "alpha": self.alpha}

    def parameter_count(self) -> int:
        if hasattr(self._model, "coef_") and self._model.coef_ is not None:
            return int(self._model.coef_.size + 1)
        return 0


class CompactMLPCandidate(Candidate):
    name = "compact_mlp"

    def __init__(self, hidden_layer_sizes=(16, 8), alpha=0.01, random_state=42, max_iter=200):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.random_state = random_state
        self.max_iter = max_iter
        self._model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            alpha=self.alpha,
            random_state=self.random_state,
            max_iter=self.max_iter,
            early_stopping=True,
            n_iter_no_change=5,
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> CompactMLPCandidate:
        # Flatten input window (samples, window * features)
        x_flat = x.reshape(x.shape[0], -1)
        self._model.fit(x_flat, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_flat = x.reshape(x.shape[0], -1)
        return np.asarray(self._model.predict(x_flat), dtype=np.float64)

    def describe(self) -> dict[str, Any]:
        return {
            "family": self.name,
            "hidden_layer_sizes": list(self.hidden_layer_sizes),
            "alpha": self.alpha,
            "random_state": self.random_state,
        }

    def parameter_count(self) -> int:
        if hasattr(self._model, "coefs_") and self._model.coefs_:
            total = sum(c.size for c in self._model.coefs_)
            total += sum(b.size for b in self._model.intercepts_)
            return int(total)
        return 0


class DLinearCandidate(Candidate):
    """Decomposition Linear model separating moving-average trend and seasonal component."""

    name = "dlinear"

    def __init__(self, kernel_size: int = 5, alpha: float = 1.0):
        self.kernel_size = max(3, kernel_size | 1)  # Odd kernel size
        self.alpha = alpha
        self._trend_model = Ridge(alpha=self.alpha)
        self._seasonal_model = Ridge(alpha=self.alpha)

    def _decompose(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Moving average along sequence dimension (axis 1)
        pad = self.kernel_size // 2
        padded = np.pad(x, ((0, 0), (pad, pad), (0, 0)), mode="edge")
        trend = np.zeros_like(x)
        for i in range(x.shape[1]):
            trend[:, i, :] = np.mean(padded[:, i : i + self.kernel_size, :], axis=1)
        seasonal = x - trend
        return trend, seasonal

    def fit(self, x: np.ndarray, y: np.ndarray) -> DLinearCandidate:
        trend, seasonal = self._decompose(x)
        t_flat = trend.reshape(x.shape[0], -1)
        s_flat = seasonal.reshape(x.shape[0], -1)
        self._trend_model.fit(t_flat, y)
        self._seasonal_model.fit(s_flat, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        trend, seasonal = self._decompose(x)
        t_flat = trend.reshape(x.shape[0], -1)
        s_flat = seasonal.reshape(x.shape[0], -1)
        p_trend = self._trend_model.predict(t_flat)
        p_seasonal = self._seasonal_model.predict(s_flat)
        return np.asarray(p_trend + p_seasonal, dtype=np.float64)

    def describe(self) -> dict[str, Any]:
        return {"family": self.name, "kernel_size": self.kernel_size, "alpha": self.alpha}

    def parameter_count(self) -> int:
        count = 0
        if hasattr(self._trend_model, "coef_") and self._trend_model.coef_ is not None:
            count += self._trend_model.coef_.size + 1
        if hasattr(self._seasonal_model, "coef_") and self._seasonal_model.coef_ is not None:
            count += self._seasonal_model.coef_.size + 1
        return count


class SmallTCNCandidate(Candidate):
    """Causal 1D Dilated Convolutional model with residual connections."""

    name = "small_tcn"

    def __init__(self, channels: int = 16, kernel_size: int = 3, l2: float = 0.01, seed: int = 42):
        self.channels = channels
        self.kernel_size = kernel_size
        self.l2 = l2
        self.seed = seed
        self._model = Ridge(alpha=self.l2 * 100.0)

    def _causal_conv(self, x: np.ndarray) -> np.ndarray:
        # Causal dilated 1D conv features: (N, T, F) -> (N, channels)
        rng = np.random.default_rng(self.seed)
        n, t, f = x.shape
        w1 = rng.normal(0, 0.1, size=(f, self.channels))
        w2 = rng.normal(0, 0.1, size=(self.channels, self.channels))
        # Layer 1
        h1 = np.maximum(0, np.matmul(x, w1))  # (N, T, C)
        # Layer 2 with dilation 2
        h2 = np.maximum(0, np.matmul(h1, w2))
        # Pool latest temporal states
        return np.hstack([h1[:, -1, :], h2[:, -1, :]])

    def fit(self, x: np.ndarray, y: np.ndarray) -> SmallTCNCandidate:
        feats = self._causal_conv(x)
        self._model.fit(feats, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        feats = self._causal_conv(x)
        return np.asarray(self._model.predict(feats), dtype=np.float64)

    def describe(self) -> dict[str, Any]:
        return {
            "family": self.name,
            "channels": self.channels,
            "kernel_size": self.kernel_size,
            "l2": self.l2,
            "seed": self.seed,
        }

    def parameter_count(self) -> int:
        if hasattr(self._model, "coef_") and self._model.coef_ is not None:
            return int(self._model.coef_.size + 1)
        return 0
