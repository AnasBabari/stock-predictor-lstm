"""Small, auditable forecasting baselines for offline model comparison."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge


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
