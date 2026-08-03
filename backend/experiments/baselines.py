"""Small, auditable forecasting baselines for offline model comparison."""

from __future__ import annotations

import inspect
from collections.abc import Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge

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
