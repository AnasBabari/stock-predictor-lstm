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
class CandidateTargets:
    """Explicit multi-task training targets for global candidates."""

    cumulative_returns: np.ndarray | None = None
    direction_classes: np.ndarray | None = None  # integer class indices: 0: down, 1: neutral, 2: up
    realized_variance: np.ndarray | None = None
    quantile_targets: np.ndarray | None = None
    horizons: tuple[int, ...] = (1,)

    @classmethod
    def from_returns(cls, y: np.ndarray, horizons: tuple[int, ...] = (1,)) -> CandidateTargets:
        return cls(cumulative_returns=np.asarray(y, dtype=float), horizons=horizons)


@dataclass(frozen=True)
class CandidatePrediction:
    """Explicit multi-task predictions emitted by global candidates."""

    return_point: np.ndarray | None = None
    return_quantiles: dict[str, np.ndarray] | None = None  # keys '0.1', '0.5', '0.9'
    direction_probabilities: np.ndarray | None = None  # shape [n, 3], columns [down, neutral, up]
    variance_forecast: np.ndarray | None = None

    @property
    def point(self) -> np.ndarray:
        """Backward compatibility alias for return_point."""
        if self.return_point is None:
            raise AttributeError("CandidatePrediction has no return_point forecast.")
        return self.return_point

    @property
    def quantiles(self) -> dict[str, np.ndarray] | None:
        """Backward compatibility alias for return_quantiles."""
        return self.return_quantiles


# Alias for backwards compatibility
Prediction = CandidatePrediction


class Candidate:
    name: str = "candidate"
    supported_tasks: tuple[str, ...] = ("returns",)

    def fit(self, x: np.ndarray, targets: CandidateTargets | np.ndarray) -> Candidate:
        raise NotImplementedError

    def predict(self, x: np.ndarray) -> CandidatePrediction:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {
            "family": self.name,
            "architecture": getattr(self, "architecture", self.name),
            "seed": getattr(self, "seed", None),
            "tasks": self.supported_tasks,
            "horizons": getattr(self, "horizons", (1,)),
        }


def _ensure_targets(targets: CandidateTargets | np.ndarray) -> CandidateTargets:
    if isinstance(targets, CandidateTargets):
        return targets
    if isinstance(targets, np.ndarray):
        return CandidateTargets.from_returns(targets)
    raise TypeError(f"Expected CandidateTargets or np.ndarray, got {type(targets)}")


def _flatten(x: np.ndarray) -> np.ndarray:
    if x.ndim == 3:
        return x.reshape(x.shape[0], -1)
    return x


# ── Baselines ────────────────────────────────────────────────────────────


class PersistenceCandidate(Candidate):
    """Zero cumulative excess return — the no-edge reference."""

    name = "persistence"
    supported_tasks = ("returns", "direction", "volatility")

    def fit(self, x: np.ndarray, targets: CandidateTargets | np.ndarray) -> PersistenceCandidate:
        return self

    def predict(self, x: np.ndarray) -> CandidatePrediction:
        zeros = np.zeros(len(x), dtype=float)
        unif_dir = np.full((len(x), 3), 1.0 / 3.0, dtype=float)
        return CandidatePrediction(
            return_point=zeros,
            return_quantiles={"0.1": zeros, "0.5": zeros, "0.9": zeros},
            direction_probabilities=unif_dir,
            variance_forecast=zeros,
        )


class RollingMeanCandidate(Candidate):
    """Shrunk trailing mean of normalized targets seen in training."""

    name = "rolling_mean_shrunk"
    supported_tasks = ("returns",)

    def __init__(self, shrinkage: float = 0.5):
        self.shrinkage = shrinkage
        self._mean = 0.0

    def fit(self, x: np.ndarray, targets: CandidateTargets | np.ndarray) -> RollingMeanCandidate:
        tgt = _ensure_targets(targets)
        if tgt.cumulative_returns is None or len(tgt.cumulative_returns) == 0:
            self._mean = 0.0
            return self
        mean = float(np.mean(tgt.cumulative_returns))
        self._mean = mean * (1 - self.shrinkage)
        return self

    def predict(self, x: np.ndarray) -> CandidatePrediction:
        values = np.full(len(x), self._mean, dtype=float)
        spread = float(np.std([self._mean])) or 0.01
        return CandidatePrediction(
            return_point=values,
            return_quantiles={
                "0.1": values - 1.2816 * spread,
                "0.5": values,
                "0.9": values + 1.2816 * spread,
            },
        )

    def describe(self) -> dict[str, Any]:
        return {
            "family": self.name,
            "shrinkage": self.shrinkage,
            "tasks": self.supported_tasks,
        }


# ── Regularised linear / DLinear ────────────────────────────────────────


# ── Regularised linear / DLinear ────────────────────────────────────────


class RidgeCandidate(Candidate):
    name = "ridge_global"
    supported_tasks = ("returns",)

    def __init__(self, alpha: float = 10.0, seed: int | None = None):
        self.alpha = alpha
        self.seed = seed
        self._model: Any | None = None
        self._scaler_mean: Any | None = None
        self._scaler_scale: Any | None = None

    def fit(self, x: np.ndarray, targets: CandidateTargets | np.ndarray) -> RidgeCandidate:
        from sklearn.linear_model import Ridge

        tgt = _ensure_targets(targets)
        if tgt.cumulative_returns is None:
            raise ValueError(f"{self.name} requires cumulative_returns in targets.")
        y = tgt.cumulative_returns
        flat = _flatten(x)
        self._scaler_mean = flat.mean(axis=0)
        self._scaler_scale = flat.std(axis=0) + 1e-12
        scaled = (flat - self._scaler_mean) / self._scaler_scale
        self._model = Ridge(alpha=self.alpha, random_state=self.seed).fit(scaled, y)
        return self

    def predict(self, x: np.ndarray) -> CandidatePrediction:
        if self._model is None or self._scaler_mean is None or self._scaler_scale is None:
            raise RuntimeError("RidgeCandidate used before fit().")
        flat = (_flatten(x) - self._scaler_mean) / self._scaler_scale
        point = self._model.predict(flat)
        return CandidatePrediction(return_point=point)


class ElasticNetCandidate(Candidate):
    name = "elastic_net_global"
    supported_tasks = ("returns",)

    def __init__(self, alpha: float = 0.01, l1_ratio: float = 0.15, seed: int | None = None):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.seed = seed
        self._model: Any | None = None
        self._mean: Any | None = None
        self._scale: Any | None = None

    def fit(self, x: np.ndarray, targets: CandidateTargets | np.ndarray) -> ElasticNetCandidate:
        from sklearn.linear_model import ElasticNet

        tgt = _ensure_targets(targets)
        if tgt.cumulative_returns is None:
            raise ValueError(f"{self.name} requires cumulative_returns in targets.")
        y = tgt.cumulative_returns
        flat = _flatten(x)
        self._mean = flat.mean(axis=0)
        self._scale = flat.std(axis=0) + 1e-12
        scaled = (flat - self._mean) / self._scale
        self._model = ElasticNet(
            alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=5000, random_state=self.seed
        )
        self._model.fit(scaled, y)
        return self

    def predict(self, x: np.ndarray) -> CandidatePrediction:
        if self._model is None or self._mean is None or self._scale is None:
            raise RuntimeError("ElasticNetCandidate used before fit().")
        flat = (_flatten(x) - self._mean) / self._scale
        point = self._model.predict(flat)
        return CandidatePrediction(return_point=point)


class DLinearGlobalCandidate(Candidate):
    """Decomposition linear model over the shared window (per-feature trend)."""

    name = "dlinear_global"
    supported_tasks = ("returns",)

    def __init__(self, kernel: int = 21, ridge_alpha: float = 5.0, seed: int | None = None):
        self.kernel = kernel
        self.ridge_alpha = ridge_alpha
        self.seed = seed
        self._model: Any | None = None
        self._trend_model: Any | None = None

    @staticmethod
    def _moving_average(window: np.ndarray, k: int) -> np.ndarray:
        pad = k // 2
        padded = np.pad(window, ((0, 0), (pad, pad), (0, 0)), mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, k, axis=1)
        return windows.mean(axis=-1)

    def fit(self, x: np.ndarray, targets: CandidateTargets | np.ndarray) -> DLinearGlobalCandidate:
        from sklearn.linear_model import Ridge

        tgt = _ensure_targets(targets)
        if tgt.cumulative_returns is None:
            raise ValueError(f"{self.name} requires cumulative_returns in targets.")
        y = tgt.cumulative_returns
        trend = self._moving_average(x, self.kernel).reshape(len(x), -1)
        seasonal = x.reshape(len(x), -1) - trend
        self._trend_model = Ridge(alpha=self.ridge_alpha, random_state=self.seed).fit(trend, y)
        self._model = Ridge(alpha=self.ridge_alpha, random_state=self.seed).fit(seasonal, y)
        return self

    def predict(self, x: np.ndarray) -> CandidatePrediction:
        if self._trend_model is None or self._model is None:
            raise RuntimeError("DLinearGlobalCandidate used before fit().")
        trend = self._moving_average(x, self.kernel).reshape(len(x), -1)
        seasonal = x.reshape(len(x), -1) - trend
        point = self._trend_model.predict(trend) + self._model.predict(seasonal)
        return CandidatePrediction(return_point=point)


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


def _set_deterministic_seed(seed: int | None) -> None:
    if seed is not None:
        import random

        random.seed(seed)
        np.random.seed(seed)
        try:
            import tensorflow as tf  # type: ignore[import-untyped]

            tf.random.set_seed(seed)
        except Exception:
            pass


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
    supported_tasks = ("returns", "direction")

    def __init__(
        self,
        *,
        architecture: str = "lstm",
        units: tuple[int, int] = (64, 32),
        dropout: float = 0.2,
        lookback: int = 60,
        epochs: int = 12,
        batch_size: int = 64,
        tasks: tuple[str, ...] = ("returns", "direction"),
        seed: int | None = None,
    ):
        self.architecture = architecture
        self.units = units
        self.dropout = dropout
        self.lookback = lookback
        self.epochs = epochs
        self.batch_size = batch_size
        self.tasks = tasks
        self.seed = seed
        self._model: Any | None = None
        self.diagnostics: dict[str, Any] = {}

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

    def fit(
        self, x: np.ndarray, targets: CandidateTargets | np.ndarray
    ) -> GlobalRecurrentCandidate:
        tf = _require_tf()
        tgt = _ensure_targets(targets)

        if "returns" in self.tasks and tgt.cumulative_returns is None:
            raise ValueError(f"{self.name} requires cumulative_returns in targets.")
        if "direction" in self.tasks and tgt.direction_classes is None:
            raise ValueError(
                f"{self.name} requires direction_classes in targets when direction task is enabled; "
                "implicit neutral substitution is forbidden."
            )

        _set_deterministic_seed(self.seed)
        self._model = self._build(x.shape[-1])

        y_ret = (
            tgt.cumulative_returns
            if tgt.cumulative_returns is not None
            else np.zeros(len(x), dtype=float)
        )
        ys_quantile = np.column_stack([y_ret, y_ret, y_ret])

        if tgt.direction_classes is not None:
            dir_classes = np.asarray(tgt.direction_classes, dtype=int)
            if (dir_classes < 0).any() or (dir_classes > 2).any():
                raise ValueError("direction_classes must contain class indices in {0, 1, 2}.")
            ys_direction = tf.one_hot(dir_classes, 3).numpy()
        else:
            # If direction is not in tasks, provide dummy one-hot (uniform) for compilation
            ys_direction = np.full((len(x), 3), 1.0 / 3.0)

        history = self._model.fit(
            x,
            {"quantiles": ys_quantile, "direction": ys_direction},
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=False,
            verbose=0,
        )
        self.diagnostics = {
            "completed_epochs": len(history.history.get("loss", [])),
            "final_loss": float(history.history["loss"][-1])
            if history.history.get("loss")
            else None,
            "tasks": self.tasks,
            "seed": self.seed,
        }
        return self

    def predict(self, x: np.ndarray) -> CandidatePrediction:
        if self._model is None:
            raise RuntimeError(f"{self.name} used before fit().")
        q, d = self._model.predict(x, verbose=0)
        # Prevent quantile crossing by monotonic projection
        sorted_q = np.sort(q, axis=-1)
        dir_probs = np.clip(d, 0.0, 1.0)
        dir_probs = dir_probs / np.sum(dir_probs, axis=-1, keepdims=True)
        return CandidatePrediction(
            return_point=sorted_q[:, 1],
            return_quantiles={"0.1": sorted_q[:, 0], "0.5": sorted_q[:, 1], "0.9": sorted_q[:, 2]},
            direction_probabilities=dir_probs,
        )


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


REGISTRY: dict[str, Callable[[int], Candidate]] = {
    PersistenceCandidate.name: lambda seed: PersistenceCandidate(),
    RollingMeanCandidate.name: lambda seed: RollingMeanCandidate(),
    RidgeCandidate.name: lambda seed: RidgeCandidate(seed=seed),
    ElasticNetCandidate.name: lambda seed: ElasticNetCandidate(seed=seed),
    DLinearGlobalCandidate.name: lambda seed: DLinearGlobalCandidate(seed=seed),
}


def register_neural_candidates(registry: dict[str, Callable[[int], Candidate]]) -> None:
    """Opt-in registration so importing this module never requires TF."""
    registry["global_lstm"] = lambda seed: GlobalRecurrentCandidate(seed=seed)
    registry["global_gru"] = lambda seed: GlobalRecurrentCandidate(architecture="gru", seed=seed)
    registry["global_tcn"] = lambda seed: TemporalConvolutionCandidate(seed=seed)
