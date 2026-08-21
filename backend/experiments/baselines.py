"""Small, auditable forecasting baselines for offline model comparison."""

from __future__ import annotations

import inspect
from collections.abc import Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge

# PyTorch is an opt-in research dependency of the offline benchmark ladder.
# Import it lazily-tolerantly so importing this module never breaks CPU-only
# environments; SmallTCNForecaster raises a clear error at construction when
# torch is unavailable.
try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - depends on the environment
    torch = None
    nn = None

# scikit-learn renamed the quantile-loss parameter between releases
# (alpha -> quantile_alpha -> quantile); detect the supported spelling.
_HGB_PARAMETERS = inspect.signature(HistGradientBoostingRegressor.__init__).parameters
_QUANTILE_PARAMETER = next(
    name for name in ("quantile", "quantile_alpha", "alpha") if name in _HGB_PARAMETERS
)


def _flatten_features(features) -> np.ndarray:
    array = np.asarray(features, dtype=float)
    if array.ndim != 3 or array.shape[0] == 0:
        raise ValueError("Features must be a non-empty (samples, lookback, features) array.")
    if not np.isfinite(array).all():
        raise ValueError("Features contain non-finite values.")
    return array.reshape(array.shape[0], -1)


def _window_features(features) -> np.ndarray:
    """Validate features for window-consuming models, keeping the 3D layout."""

    array = np.asarray(features, dtype=float)
    if array.ndim != 3 or array.shape[0] == 0:
        raise ValueError("Features must be a non-empty (samples, lookback, features) array.")
    if not np.isfinite(array).all():
        raise ValueError("Features contain non-finite values.")
    return array


class PersistenceForecaster:
    """Predict no price change from the forecast origin."""

    name = "persistence"

    def fit(self, features, targets):
        return self

    def predict_prices(self, *, origins, horizons) -> np.ndarray:
        origin_array = np.asarray(origins, dtype=float).reshape(-1)
        return np.repeat(origin_array[:, None], len(tuple(horizons)), axis=1)


class DriftForecaster:
    """Extrapolate the average price change observed in each input window."""

    name = "drift"

    def __init__(self, close_feature_index: int):
        self.close_feature_index = int(close_feature_index)

    def fit(self, features, targets):
        _flatten_features(features)
        return self

    def predict_prices(self, features, *, origins, horizons) -> np.ndarray:
        array = np.asarray(features, dtype=float)
        origins_array = np.asarray(origins, dtype=float).reshape(-1)
        if array.shape[0] != len(origins_array):
            raise ValueError("Origins must align with feature windows.")
        close_history = array[:, :, self.close_feature_index]
        slope = (close_history[:, -1] - close_history[:, 0]) / (close_history.shape[1] - 1)
        return np.column_stack([origins_array + slope * int(horizon) for horizon in horizons])


class RidgeForecaster:
    name = "ridge"

    def __init__(self, alpha: float = 1.0):
        self.model = Ridge(alpha=alpha)

    def fit(self, features, targets):
        target_array = np.asarray(targets, dtype=float)
        if target_array.ndim != 2:
            raise ValueError("Targets must be a two-dimensional direct-horizon matrix.")
        self.model.fit(_flatten_features(features), target_array)
        return self

    def predict(self, features) -> np.ndarray:
        prediction = np.asarray(self.model.predict(_flatten_features(features)), dtype=float)
        return prediction.reshape(len(prediction), -1)


class ElasticNetForecaster:
    name = "elastic_net"

    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5, max_iter: int = 2000):
        self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter)

    def fit(self, features, targets):
        target_array = np.asarray(targets, dtype=float)
        if target_array.ndim != 2:
            raise ValueError("Targets must be a two-dimensional direct-horizon matrix.")
        self.model.fit(_flatten_features(features), target_array)
        return self

    def predict(self, features) -> np.ndarray:
        prediction = np.asarray(self.model.predict(_flatten_features(features)), dtype=float)
        return prediction.reshape(len(prediction), -1)


class HistogramGradientBoostingForecaster:
    """One deterministic tree regressor per direct forecast horizon."""

    name = "hist_gradient_boosting"

    def __init__(
        self,
        *,
        learning_rate: float = 0.05,
        max_iter: int = 60,
        max_leaf_nodes: int = 15,
        random_state: int = 42,
    ):
        self.parameters = {
            "learning_rate": learning_rate,
            "max_iter": max_iter,
            "max_leaf_nodes": max_leaf_nodes,
            "random_state": random_state,
        }
        self.models: list[HistGradientBoostingRegressor] = []

    def fit(self, features, targets):
        flattened = _flatten_features(features)
        target_array = np.asarray(targets, dtype=float)
        if target_array.ndim != 2 or len(target_array) != len(flattened):
            raise ValueError("Targets must align with feature samples.")
        self.models = []
        for column in range(target_array.shape[1]):
            model = HistGradientBoostingRegressor(**self.parameters)
            model.fit(flattened, target_array[:, column])
            self.models.append(model)
        return self

    def predict(self, features) -> np.ndarray:
        if not self.models:
            raise ValueError("Forecaster must be fitted before prediction.")
        flattened = _flatten_features(features)
        return np.column_stack([model.predict(flattened) for model in self.models])


class QuantileForecaster:
    """One quantile tree regressor per direct forecast horizon and quantile."""

    name = "quantile_hist_gradient_boosting"

    def __init__(
        self,
        *,
        quantiles: Sequence[float] = (0.05, 0.5, 0.95),
        learning_rate: float = 0.05,
        max_iter: int = 60,
        max_leaf_nodes: int = 15,
        random_state: int = 42,
    ):
        quantile_array = np.asarray(list(quantiles), dtype=float)
        if quantile_array.ndim != 1 or quantile_array.size < 2:
            raise ValueError("quantiles must be a sequence of at least two values.")
        if not np.isfinite(quantile_array).all() or np.any(
            (quantile_array <= 0.0) | (quantile_array >= 1.0)
        ):
            raise ValueError("Each quantile must lie strictly between zero and one.")
        if np.any(np.diff(quantile_array) <= 0.0):
            raise ValueError("quantiles must be strictly increasing.")
        self.quantiles: tuple[float, ...] = tuple(float(value) for value in quantile_array)
        self.parameters = {
            "learning_rate": learning_rate,
            "max_iter": max_iter,
            "max_leaf_nodes": max_leaf_nodes,
            "random_state": random_state,
        }
        self.models: list[dict[float, HistGradientBoostingRegressor]] = []

    def fit(self, features, targets):
        flattened = _flatten_features(features)
        target_array = np.asarray(targets, dtype=float)
        if target_array.ndim != 2 or len(target_array) != len(flattened):
            raise ValueError("Targets must align with feature samples.")
        if not np.isfinite(target_array).all():
            raise ValueError("Targets contain non-finite values.")
        self.models = []
        for column in range(target_array.shape[1]):
            per_quantile: dict[float, HistGradientBoostingRegressor] = {}
            for tau in self.quantiles:
                model = HistGradientBoostingRegressor(
                    loss="quantile", **{_QUANTILE_PARAMETER: tau}, **self.parameters
                )
                model.fit(flattened, target_array[:, column])
                per_quantile[tau] = model
            self.models.append(per_quantile)
        return self

    def predict(self, features) -> np.ndarray:
        """Return ``(samples, horizons, len(quantiles))`` quantile predictions."""

        if not self.models:
            raise ValueError("Forecaster must be fitted before prediction.")
        flattened = _flatten_features(features)
        stacked = np.stack(
            [
                np.column_stack([per_quantile[tau].predict(flattened) for tau in self.quantiles])
                for per_quantile in self.models
            ],
            axis=1,
        )
        return stacked

    def predict_median(self, features) -> np.ndarray:
        """Return ``(samples, horizons)`` point forecasts from the central quantile."""

        predicted = self.predict(features)
        closest = min(range(len(self.quantiles)), key=lambda i: abs(self.quantiles[i] - 0.5))
        return predicted[:, :, closest]


_TCN_TORCH_ERROR = (
    "SmallTCNForecaster requires PyTorch, which is not a backend dependency. "
    "Install torch into the active environment before enabling include_tcn."
)


if torch is not None:

    class _SmallTCNResidualBlock(torch.nn.Module):
        """Causal dilated convolution with a residual connection."""

        def __init__(self, channels: int, kernel_size: int, dilation: int):
            super().__init__()
            self.padding = (kernel_size - 1) * dilation
            self.conv = torch.nn.Conv1d(
                channels,
                channels,
                kernel_size,
                dilation=dilation,
                padding=self.padding,
            )

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            convolved = self.conv(values)[..., : values.shape[-1]]
            return torch.relu(convolved) + values

    class _SmallTCNModule(torch.nn.Module):
        """Bounded causal dilated TCN: pointwise input projection, stacked
        residual blocks with exponentially growing dilation, last-step pool,
        and one linear head producing every direct horizon at once."""

        def __init__(
            self,
            feature_count: int,
            horizon_count: int,
            channels: int,
            kernel_size: int,
            blocks: int,
        ):
            super().__init__()
            self.input_projection = torch.nn.Conv1d(feature_count, channels, kernel_size=1)
            self.residual_blocks = torch.nn.ModuleList(
                [_SmallTCNResidualBlock(channels, kernel_size, 2**depth) for depth in range(blocks)]
            )
            self.head = torch.nn.Linear(channels, horizon_count)

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            hidden = torch.relu(self.input_projection(values.transpose(1, 2)))
            for block in self.residual_blocks:
                hidden = block(hidden)
            return self.head(hidden[..., -1])


class SmallTCNForecaster:
    """Small causal dilated temporal-convolutional direct-horizon forecaster.

    A genuine TCN: bounded channel count, causal dilated convolutions with
    residual connections, and L2 regularisation applied as weight decay.
    Training is stochastic and fully determined by ``seed`` (fixed seed,
    full-batch, unshuffled).

    Note: this is NOT the model behind the research ledger's former
    ``small_tcn`` family. That family was renamed ``random_features_ridge``
    after review showed it was a fixed random-feature projection with a Ridge
    readout, not a convolutional network. The name ``small_tcn`` is retained
    here because this implementation actually performs temporal convolution.

    Construction raises a clear ``RuntimeError`` when PyTorch is missing; the
    default benchmark ladder never instantiates this forecaster.
    """

    name = "small_tcn"

    def __init__(
        self,
        *,
        channels: int = 16,
        kernel_size: int = 3,
        blocks: int = 3,
        l2: float = 0.01,
        epochs: int = 12,
        learning_rate: float = 0.01,
        seed: int = 42,
    ):
        if torch is None or nn is None:
            raise RuntimeError(_TCN_TORCH_ERROR)
        if channels < 1 or kernel_size < 2 or blocks < 1 or epochs < 1:
            raise ValueError("TCN architecture and training settings must be positive.")
        if l2 < 0.0 or learning_rate <= 0.0:
            raise ValueError("TCN regularisation and learning rate settings are invalid.")
        self.channels = channels
        self.kernel_size = kernel_size
        self.blocks = blocks
        self.l2 = l2
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.seed = seed
        self.model = None
        self.selected_epoch: int | None = None

    def _build_model(self, feature_count: int, horizon_count: int):
        assert torch is not None
        torch.manual_seed(self.seed)
        return _SmallTCNModule(
            feature_count, horizon_count, self.channels, self.kernel_size, self.blocks
        )

    def _train(
        self,
        model,
        training_tensor,
        target_tensor,
        epochs: int,
        validation_tensors: tuple | None,
    ) -> int:
        assert torch is not None
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.learning_rate, weight_decay=self.l2
        )
        loss_fn = nn.MSELoss()
        best_epoch = epochs
        best_validation_loss = float("inf")
        model.train()
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad(set_to_none=True)
            loss_fn(model(training_tensor), target_tensor).backward()
            optimizer.step()
            if validation_tensors is not None:
                validation_features, validation_targets = validation_tensors
                model.eval()
                with torch.no_grad():
                    validation_loss = float(loss_fn(model(validation_features), validation_targets))
                model.train()
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_epoch = epoch
        return best_epoch

    def _prepare(self, features, targets=None):
        assert torch is not None
        feature_array = _window_features(features)
        tensor = torch.as_tensor(feature_array, dtype=torch.float32)
        if targets is None:
            return tensor
        target_array = np.asarray(targets, dtype=float)
        if target_array.ndim != 2 or len(target_array) != len(feature_array):
            raise ValueError("Targets must align with feature samples.")
        if not np.isfinite(target_array).all():
            raise ValueError("Targets contain non-finite values.")
        return tensor, torch.as_tensor(target_array, dtype=torch.float32)

    def fit(self, features, targets, *, validation_data=None):
        feature_tensor, target_tensor = self._prepare(features, targets)
        validation_tensors = None
        if validation_data is not None:
            validation_features, validation_targets = validation_data
            validation_tensors = self._prepare(validation_features, validation_targets)
        self.model = self._build_model(feature_tensor.shape[2], target_tensor.shape[1])
        self.selected_epoch = self._train(
            self.model, feature_tensor, target_tensor, self.epochs, validation_tensors
        )
        return self

    def refit(self, features, targets):
        """Rebuild a fresh model and refit all rows for the selected epoch."""

        if self.selected_epoch is None:
            raise ValueError("Candidate must select an epoch before final refitting.")
        feature_tensor, target_tensor = self._prepare(features, targets)
        self.model = self._build_model(feature_tensor.shape[2], target_tensor.shape[1])
        self._train(self.model, feature_tensor, target_tensor, self.selected_epoch, None)
        return self

    def predict(self, features) -> np.ndarray:
        if self.model is None:
            raise ValueError("Forecaster must be fitted before prediction.")
        assert torch is not None
        feature_tensor = self._prepare(features)
        self.model.eval()
        with torch.no_grad():
            output = self.model(feature_tensor).numpy()
        return np.asarray(output, dtype=float).reshape(len(feature_tensor), -1)

    def parameter_count(self) -> int:
        if self.model is None:
            return 0
        return int(sum(parameter.numel() for parameter in self.model.parameters()))

    def metadata(self) -> dict:
        return {
            "architecture": self.name,
            "target_type": "regression",
            "channels": self.channels,
            "kernel_size": self.kernel_size,
            "blocks": self.blocks,
            "l2": self.l2,
            "epochs": self.epochs,
            "selected_epoch": self.selected_epoch,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "parameter_count": self.parameter_count(),
        }


def quantile_crossing_rate(predicted) -> float:
    """Return the fraction of rows whose adjacent quantiles are out of order.

    ``predicted`` must be a ``(samples, horizons, quantiles)`` array with
    quantiles ordered from low to high. A row counts as a crossing when any
    adjacent lower quantile strictly exceeds the next higher quantile.
    """

    array = np.asarray(predicted, dtype=float)
    if array.ndim != 3 or array.shape[0] == 0 or array.shape[2] < 2:
        raise ValueError(
            "predicted must be a non-empty (samples, horizons, quantiles) array "
            "with at least two quantiles."
        )
    if not np.isfinite(array).all():
        raise ValueError("predicted contains non-finite values.")
    crossings = np.any(array[:, :, :-1] > array[:, :, 1:], axis=(1, 2))
    return float(np.mean(crossings))
