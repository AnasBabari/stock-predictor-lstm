"""Global multi-asset candidate models for return forecasting (slice 8).

Every candidate consumes POOLED cross-ticker rows: X of shape
(n_rows, window, features) and y of shape (n_rows,) holding the cumulative
h-step log return, volatility-normalized by the caller (spec §4.4 z-score).
Ticker identity enters only through provided feature columns — a ticker not
seen in training remains usable.

Registry contract: name -> factory(seed) -> object exposing fit(X, y) and
predict(X) -> (point_forecast, quantiles{q: array} | None). Deterministic
under seed. Neural candidates lazily import TensorFlow and raise a clear
error when unavailable, mirroring backend/experiments/baselines.py.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Prediction:
    point: np.ndarray
    quantiles: dict[str, np.ndarray] | None = None  # keys '0.1', '0.5', '0.9'


class Candidate:
    name: str = "candidate"

    def fit(self, x: np.ndarray, y: np.ndarray) -> Candidate:
        raise NotImplementedError

    def predict(self, x: np.ndarray) -> Prediction:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"family": self.name}


def _flatten(x: np.ndarray) -> np.ndarray:
    if x.ndim == 3:
        return x.reshape(x.shape[0], -1)
    return x


# ── Baselines ────────────────────────────────────────────────────────────


class PersistenceCandidate(Candidate):
    """Zero cumulative excess return — the no-edge reference."""

    name = "persistence"

    def fit(self, x: np.ndarray, y: np.ndarray) -> PersistenceCandidate:
        return self

    def predict(self, x: np.ndarray) -> Prediction:
        zeros = np.zeros(len(x))
        return Prediction(
            point=zeros,
            quantiles={"0.1": zeros * -1.645, "0.5": zeros, "0.9": zeros * 1.645},
        )


class RollingMeanCandidate(Candidate):
    """Shrunk trailing mean of normalized targets seen in training."""

    name = "rolling_mean_shrunk"

    def __init__(self, shrinkage: float = 0.5):
        self.shrinkage = shrinkage
        self._mean = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> RollingMeanCandidate:
        mean = float(np.mean(y)) if len(y) else 0.0
        self._mean = mean * (1 - self.shrinkage)
        return self

    def predict(self, x: np.ndarray) -> Prediction:
        values = np.full(len(x), self._mean)
        spread = float(np.std([self._mean])) or 0.01
        return Prediction(
            point=values,
            quantiles={
                "0.1": values - 1.2816 * spread,
                "0.5": values,
                "0.9": values + 1.2816 * spread,
            },
        )

    def describe(self) -> dict:
        return {"family": self.name, "shrinkage": self.shrinkage}


# ── Regularised linear / DLinear ────────────────────────────────────────


class RidgeCandidate(Candidate):
    name = "ridge_global"

    def __init__(self, alpha: float = 10.0):
        self.alpha = alpha
        self._model: Any | None = None
        self._scaler_mean: Any | None = None
        self._scaler_scale: Any | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> RidgeCandidate:
        from sklearn.linear_model import Ridge

        flat = _flatten(x)
        self._scaler_mean = flat.mean(axis=0)
        self._scaler_scale = flat.std(axis=0) + 1e-12
        scaled = (flat - self._scaler_mean) / self._scaler_scale
        self._model = Ridge(alpha=self.alpha).fit(scaled, y)
        return self

    def predict(self, x: np.ndarray) -> Prediction:
        if self._model is None or self._scaler_mean is None or self._scaler_scale is None:
            raise RuntimeError("RidgeCandidate used before fit().")
        flat = (_flatten(x) - self._scaler_mean) / self._scaler_scale
        return Prediction(point=self._model.predict(flat))


class ElasticNetCandidate(Candidate):
    name = "elastic_net_global"

    def __init__(self, alpha: float = 0.01, l1_ratio: float = 0.15):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self._model: Any | None = None
        self._mean: Any | None = None
        self._scale: Any | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> ElasticNetCandidate:
        from sklearn.linear_model import ElasticNet

        flat = _flatten(x)
        self._mean = flat.mean(axis=0)
        self._scale = flat.std(axis=0) + 1e-12
        scaled = (flat - self._mean) / self._scale
        self._model = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=5000)
        self._model.fit(scaled, y)
        return self

    def predict(self, x: np.ndarray) -> Prediction:
        if self._model is None or self._mean is None or self._scale is None:
            raise RuntimeError("ElasticNetCandidate used before fit().")
        flat = (_flatten(x) - self._mean) / self._scale
        return Prediction(point=self._model.predict(flat))


class DLinearGlobalCandidate(Candidate):
    """Decomposition linear model over the shared window (per-feature trend)."""

    name = "dlinear_global"

    def __init__(self, kernel: int = 21, ridge_alpha: float = 5.0):
        self.kernel = kernel
        self.ridge_alpha = ridge_alpha
        self._model: Any | None = None
        self._trend_model: Any | None = None

    @staticmethod
    def _moving_average(window: np.ndarray, k: int) -> np.ndarray:
        pad = k // 2
        padded = np.pad(window, ((0, 0), (pad, pad), (0, 0)), mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, k, axis=1)
        return windows.mean(axis=-1)

    def fit(self, x: np.ndarray, y: np.ndarray) -> DLinearGlobalCandidate:
        from sklearn.linear_model import Ridge

        trend = self._moving_average(x, self.kernel).reshape(len(x), -1)
        seasonal = x.reshape(len(x), -1) - trend
        self._trend_model = Ridge(alpha=self.ridge_alpha).fit(trend, y)
        self._model = Ridge(alpha=self.ridge_alpha).fit(seasonal, y)
        return self

    def predict(self, x: np.ndarray) -> Prediction:
        if self._trend_model is None or self._model is None:
            raise RuntimeError("DLinearGlobalCandidate used before fit().")
        trend = self._moving_average(x, self.kernel).reshape(len(x), -1)
        seasonal = x.reshape(len(x), -1) - trend
        return Prediction(point=self._trend_model.predict(trend) + self._model.predict(seasonal))


# ── Global recurrent / convolutional candidates (lazy TensorFlow) ────────


def _require_tf():
    try:
        import tensorflow as tf  # type: ignore[import-untyped]
    except (ImportError, OSError, RuntimeError) as exc:
        raise RuntimeError(
            f"{exc.__class__.__name__}: TensorFlow is required for this "
            "candidate but is not installed. Install the opt-in `training` "
            "dependency group."
        ) from exc
    return tf


def _pinball_loss(q: float):
    import tensorflow as tf  # type: ignore[import-untyped]

    def loss(y_true, y_pred):
        error = y_true - y_pred
        return tf.reduce_mean(tf.maximum(q * error, (q - 1) * error))

    return loss


class GlobalRecurrentCandidate(Candidate):
    """Shared-encoder LSTM/GRU with quantile + direction heads.

    Heads: pinball at {0.1, 0.5, 0.9} for the z-scored return and a three-way
    softmax for direction (down/neutral/up), weighted per spec §4.4 defaults.
    """

    name = "global_lstm"

    def __init__(
        self,
        *,
        architecture: str = "lstm",
        units: tuple[int, int] = (64, 32),
        dropout: float = 0.2,
        lookback: int = 60,
        epochs: int = 12,
        batch_size: int = 64,
        direction_labels: np.ndarray | None = None,
    ):
        self.architecture = architecture
        self.units = units
        self.dropout = dropout
        self.lookback = lookback
        self.epochs = epochs
        self.batch_size = batch_size
        self.direction_labels = direction_labels
        self._model: Any | None = None

    def _build(self, n_features: int):
        tf = _require_tf()
        layers = tf.keras.layers
        inp = layers.Input(shape=(self.lookback, n_features))
        rec = (
            layers.GRU(self.units[0], return_sequences=True)
            if self.architecture == "gru"
            else layers.LSTM(self.units[0], return_sequences=True)
        )
        h = rec(inp)
        h = layers.Dropout(self.dropout)(h)
        core = (
            layers.GRU(self.units[1]) if self.architecture == "gru" else layers.LSTM(self.units[1])
        )
        h = core(h)
        h = layers.Dense(16, activation="relu")(h)
        q_head = layers.Dense(3, name="quantiles")(h)
        d_head = layers.Dense(3, activation="softmax", name="direction")(h)
        model = tf.keras.Model(inp, [q_head, d_head])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss={
                "quantiles": lambda yt, yp: sum(
                    _pinball_loss(q)(yt[:, i], yp[:, i]) for i, q in enumerate((0.1, 0.5, 0.9))
                ),
                "direction": "categorical_crossentropy",
            },
            loss_weights={"quantiles": 0.4, "direction": 0.2},
        )
        return model

    def fit(self, x: np.ndarray, y: np.ndarray) -> GlobalRecurrentCandidate:
        tf = _require_tf()
        self._model = self._build(x.shape[-1])
        ys_quantile = np.column_stack([y, y, y])
        if self.direction_labels is not None and len(self.direction_labels) == len(y):
            ys_direction = tf.one_hot(self.direction_labels.astype(int), 3).numpy()
        else:
            neutral = np.zeros((len(y), 3))
            neutral[:, 1] = 1.0
            ys_direction = neutral
        self._model.fit(
            x,
            {"quantiles": ys_quantile, "direction": ys_direction},
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=False,
            verbose=0,
        )
        return self

    def predict(self, x: np.ndarray) -> Prediction:
        if self._model is None:
            raise RuntimeError("GlobalRecurrentCandidate used before fit().")
        q, _ = self._model.predict(x, verbose=0)
        return Prediction(point=q[:, 1], quantiles={"0.1": q[:, 0], "0.5": q[:, 1], "0.9": q[:, 2]})


class TemporalConvolutionCandidate(GlobalRecurrentCandidate):
    """Residual causal convolutions, dilations 1..16, 32 channels."""

    name = "global_tcn"

    def __init__(self, **kwargs):
        kwargs.setdefault("epochs", 12)
        super().__init__(**kwargs)

    def _build(self, n_features: int):
        tf = _require_tf()
        layers = tf.keras.layers
        inp = layers.Input(shape=(self.lookback, n_features))
        h = inp
        for dilation in (1, 2, 4, 8, 16):
            block = layers.Conv1D(
                32, kernel_size=3, dilation_rate=dilation, padding="causal", activation="relu"
            )(h)
            block = layers.Conv1D(32, kernel_size=1, activation="relu")(block)
            if h.shape[-1] != 32:
                h = layers.Conv1D(32, kernel_size=1)(h)
            h = layers.Add()([h, block])
        h = layers.GlobalAveragePooling1D()(h)
        q_head = layers.Dense(3, name="quantiles")(h)
        d_head = layers.Dense(3, activation="softmax", name="direction")(h)
        model = tf.keras.Model(inp, [q_head, d_head])
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
        return model


REGISTRY: dict[str, Callable[[int], Candidate]] = {
    PersistenceCandidate.name: lambda seed: PersistenceCandidate(),
    RollingMeanCandidate.name: lambda seed: RollingMeanCandidate(),
    RidgeCandidate.name: lambda seed: RidgeCandidate(),
    ElasticNetCandidate.name: lambda seed: ElasticNetCandidate(),
    DLinearGlobalCandidate.name: lambda seed: DLinearGlobalCandidate(),
}


def register_neural_candidates(registry: dict[str, Callable[[int], Candidate]]) -> None:
    """Opt-in registration so importing this module never requires TF."""
    registry["global_lstm"] = lambda seed: GlobalRecurrentCandidate()
    registry["global_gru"] = lambda seed: GlobalRecurrentCandidate(architecture="gru")
    registry["global_tcn"] = lambda seed: TemporalConvolutionCandidate()
