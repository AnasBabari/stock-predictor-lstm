"""Deterministic candidate model implementations for stock autoresearch."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler


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
        self._scaler = RobustScaler()

    def fit(self, x: np.ndarray, y: np.ndarray) -> RidgeCandidate:
        x_flat = x[:, -1, :]
        x_scaled = self._scaler.fit_transform(x_flat)
        self._model.fit(x_scaled, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_flat = x[:, -1, :]
        x_scaled = self._scaler.transform(x_flat)
        return np.asarray(self._model.predict(x_scaled), dtype=np.float64)

    def describe(self) -> dict[str, Any]:
        return {"family": self.name, "alpha": self.alpha}

    def parameter_count(self) -> int:
        if hasattr(self._model, "coef_") and self._model.coef_ is not None:
            return int(self._model.coef_.size + 1)
        return 0


@dataclass
class ElasticNetCandidate(Candidate):
    """Elastic Net linear model over the latest window step.

    Tuned grid variants are named ``elastic_net_a<alpha>_l<l1_ratio * 100>``
    where the alpha tag drops the decimal point (``a01`` = alpha 0.1,
    ``a001`` = alpha 0.01, ``a100`` = alpha 10.0) and the l1 tag is the
    l1_ratio percentage (``l15`` = 0.15, ``l50`` = 0.5, ``l85`` = 0.85).
    ``describe()`` always reports the exact ``alpha`` and ``l1_ratio`` so the
    encoded name is only a convenience for ledger grouping.
    """

    alpha: float = 1.0
    l1_ratio: float = 0.5
    name: str = "elastic_net"

    def __post_init__(self) -> None:
        self._model = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=2000)
        self._scaler = RobustScaler()

    def fit(self, x: np.ndarray, y: np.ndarray) -> ElasticNetCandidate:
        x_flat = x[:, -1, :]
        x_scaled = self._scaler.fit_transform(x_flat)
        self._model.fit(x_scaled, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_flat = x[:, -1, :]
        x_scaled = self._scaler.transform(x_flat)
        return np.asarray(self._model.predict(x_scaled), dtype=np.float64)

    def describe(self) -> dict[str, Any]:
        return {"family": self.name, "alpha": self.alpha, "l1_ratio": self.l1_ratio}

    def parameter_count(self) -> int:
        if hasattr(self._model, "coef_") and self._model.coef_ is not None:
            return int(self._model.coef_.size + 1)
        return 0


# Tuned Elastic Net grid explored against the horizon-10 promotion gates.
# Family names follow ``elastic_net_a<alpha>_l<l1_ratio * 100>`` (see the
# ElasticNetCandidate docstring). The baseline ``elastic_net`` family
# (alpha=1.0, l1_ratio=0.5) is deliberately not part of this grid.
#
# Wave-1 probing (run tag tune-h10) showed that every l1_ratio >= 0.15 with
# alpha >= 0.1 zeroes all coefficients on this snapshot, collapsing to the
# same constant predictor as the baseline. The grid therefore concentrates on
# the ridge-leaning corner (l1_ratio <= 0.15) where coefficients survive.
ELASTIC_NET_TUNING_GRID: tuple[tuple[float, float], ...] = (
    (0.01, 0.01),  # near-OLS, ridge-dominated penalty
    (0.01, 0.05),  # weak shrinkage, slight lasso share
    (0.01, 0.15),  # weak shrinkage, strongest surviving lasso share
    (0.1, 0.01),  # mild shrinkage, ridge-dominated
    (0.3, 0.01),  # moderate shrinkage, ridge-dominated
    (1.0, 0.01),  # baseline alpha with ridge-dominated penalty
    (3.0, 0.01),  # stronger shrinkage than baseline
    (10.0, 0.01),  # much stronger shrinkage than baseline
)


def elastic_net_family_name(alpha: float, l1_ratio: float) -> str:
    """Encode (alpha, l1_ratio) as ``elastic_net_a<alpha>_l<l1_ratio * 100>``."""
    alpha_tag = str(alpha).replace(".", "")
    l1_tag = str(int(round(l1_ratio * 100)))
    return f"elastic_net_a{alpha_tag}_l{l1_tag}"


def elastic_net_family_factories() -> dict[str, Callable[[int], ElasticNetCandidate]]:
    """Return named factories for every tuned Elastic Net grid point.

    Each factory builds an ``ElasticNetCandidate`` with the grid point's
    ``alpha``/``l1_ratio`` and the family name from
    ``elastic_net_family_name``. Registration dictionaries merge the returned
    mapping so the existing ``elastic_net`` baseline factory stays unchanged.
    """

    def make_factory(
        family: str, alpha: float, l1_ratio: float
    ) -> Callable[[int], ElasticNetCandidate]:
        return lambda seed: ElasticNetCandidate(alpha=alpha, l1_ratio=l1_ratio, name=family)

    return {
        elastic_net_family_name(alpha, l1_ratio): make_factory(
            elastic_net_family_name(alpha, l1_ratio), alpha, l1_ratio
        )
        for alpha, l1_ratio in ELASTIC_NET_TUNING_GRID
    }


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
        self._scaler = RobustScaler()

    def fit(self, x: np.ndarray, y: np.ndarray) -> CompactMLPCandidate:
        # Flatten input window (samples, window * features)
        x_flat = x.reshape(x.shape[0], -1)
        x_scaled = self._scaler.fit_transform(x_flat)
        self._model.fit(x_scaled, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_flat = x.reshape(x.shape[0], -1)
        x_scaled = self._scaler.transform(x_flat)
        return np.asarray(self._model.predict(x_scaled), dtype=np.float64)

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
        self._trend_scaler = RobustScaler()
        self._seasonal_scaler = RobustScaler()

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
        t_scaled = self._trend_scaler.fit_transform(t_flat)
        s_scaled = self._seasonal_scaler.fit_transform(s_flat)
        self._trend_model.fit(t_scaled, y)
        self._seasonal_model.fit(s_scaled, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        trend, seasonal = self._decompose(x)
        t_flat = trend.reshape(x.shape[0], -1)
        s_flat = seasonal.reshape(x.shape[0], -1)
        t_scaled = self._trend_scaler.transform(t_flat)
        s_scaled = self._seasonal_scaler.transform(s_flat)
        p_trend = self._trend_model.predict(t_scaled)
        p_seasonal = self._seasonal_model.predict(s_scaled)
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
