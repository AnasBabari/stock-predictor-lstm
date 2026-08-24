"""Causal panel construction for multi-horizon volatility forecasting.

Each origin contains a 60-session stationary feature window ending at t.
Targets begin at t+1. The matched baseline is a causal log-HAR forecast whose
coefficients may use realized observations through t, but never any target
observation after the forecast origin.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backend.panel.features import build_features_v5
from backend.panel.volatility import (
    causal_log_har_forecasts,
    cumulative_variance_target,
    realized_variance_proxies,
)

from .contracts import VolatilityForecastProtocol

_EPSILON_VARIANCE = 1e-12


@dataclass(frozen=True)
class VolatilityPanelExamples:
    """Pooled examples plus auditable identity for every origin."""

    features: np.ndarray
    baseline_variance: np.ndarray
    realized_variance: np.ndarray
    cumulative_returns: np.ndarray
    direction_classes: np.ndarray
    tickers: np.ndarray
    origin_dates: np.ndarray
    origin_closes: np.ndarray
    horizons: tuple[int, ...]
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        rows = len(self.features)
        horizon_count = len(self.horizons)
        if self.features.ndim != 3:
            raise ValueError("features must have shape [rows, window, features]")
        for name, values in (
            ("baseline_variance", self.baseline_variance),
            ("realized_variance", self.realized_variance),
            ("cumulative_returns", self.cumulative_returns),
            ("direction_classes", self.direction_classes),
        ):
            if values.shape != (rows, horizon_count):
                raise ValueError(f"{name} must have shape {(rows, horizon_count)}")
        if any(
            len(values) != rows for values in (self.tickers, self.origin_dates, self.origin_closes)
        ):
            raise ValueError("origin identity arrays must match feature rows")
        numeric = (
            self.features,
            self.baseline_variance,
            self.realized_variance,
            self.cumulative_returns,
            self.origin_closes,
        )
        if not all(np.isfinite(values).all() for values in numeric):
            raise ValueError("panel examples contain non-finite numeric values")
        if (self.baseline_variance <= 0).any() or (self.realized_variance <= 0).any():
            raise ValueError("variance targets and baselines must be strictly positive")
        if not np.isin(self.direction_classes, (0, 1, 2)).all():
            raise ValueError("direction class must be down=0, neutral=1, or up=2")


def _direction_class(cumulative_return: float, baseline_variance: float) -> int:
    threshold = max(0.0005, 0.10 * float(np.sqrt(max(baseline_variance, 0.0))))
    if cumulative_return < -threshold:
        return 0
    if cumulative_return > threshold:
        return 2
    return 1


def build_volatility_panel_examples(
    panel: Mapping[str, pd.DataFrame],
    protocol: VolatilityForecastProtocol | None = None,
    *,
    minimum_har_history: int = 60,
) -> VolatilityPanelExamples:
    """Build pooled, leakage-safe examples from immutable OHLCV frames."""
    contract = protocol or VolatilityForecastProtocol()
    if not panel:
        raise ValueError("panel must contain at least one ticker")

    x_rows: list[np.ndarray] = []
    baseline_rows: list[np.ndarray] = []
    variance_rows: list[np.ndarray] = []
    return_rows: list[np.ndarray] = []
    direction_rows: list[np.ndarray] = []
    ticker_rows: list[str] = []
    date_rows: list[np.datetime64] = []
    close_rows: list[float] = []
    maximum_horizon = max(contract.horizons)

    for ticker, raw in sorted(panel.items()):
        frame = raw.sort_index().copy()
        if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
            raise ValueError(f"{ticker}: dates must be unique and increasing")
        feature_frame = build_features_v5(frame)
        missing = [name for name in contract.feature_names if name not in feature_frame]
        if missing:
            raise ValueError(f"{ticker}: missing features {missing}")

        proxy = realized_variance_proxies(frame)["RV_Total"].clip(lower=_EPSILON_VARIANCE)
        baselines = causal_log_har_forecasts(
            proxy,
            contract.horizons,
            minimum_history=minimum_har_history,
        )
        variance_targets = np.column_stack(
            [
                cumulative_variance_target(proxy, horizon).to_numpy(dtype=np.float64)
                for horizon in contract.horizons
            ]
        )
        close = frame["Close"].to_numpy(dtype=np.float64)
        feature_values = feature_frame[list(contract.feature_names)].to_numpy(dtype=np.float64)

        first_origin = max(contract.window_size - 1, minimum_har_history)
        last_origin = len(frame) - maximum_horizon
        for origin in range(first_origin, last_origin):
            window = feature_values[origin - contract.window_size + 1 : origin + 1]
            base = baselines[origin]
            target_var = variance_targets[origin]
            if not (
                np.isfinite(window).all()
                and np.isfinite(base).all()
                and np.isfinite(target_var).all()
                and (base > 0).all()
                and (target_var > 0).all()
            ):
                continue

            cumulative_returns = np.array(
                [np.log(close[origin + horizon] / close[origin]) for horizon in contract.horizons],
                dtype=np.float64,
            )
            if not np.isfinite(cumulative_returns).all():
                continue
            classes = np.array(
                [
                    _direction_class(ret, baseline)
                    for ret, baseline in zip(cumulative_returns, base, strict=True)
                ],
                dtype=np.int64,
            )

            x_rows.append(window)
            baseline_rows.append(base)
            variance_rows.append(target_var)
            return_rows.append(cumulative_returns)
            direction_rows.append(classes)
            ticker_rows.append(str(ticker).upper())
            date_rows.append(np.datetime64(pd.Timestamp(frame.index[origin]).date()))
            close_rows.append(float(close[origin]))

    if not x_rows:
        raise ValueError("panel did not yield any finite volatility examples")

    return VolatilityPanelExamples(
        features=np.asarray(x_rows, dtype=np.float32),
        baseline_variance=np.asarray(baseline_rows, dtype=np.float32),
        realized_variance=np.asarray(variance_rows, dtype=np.float32),
        cumulative_returns=np.asarray(return_rows, dtype=np.float32),
        direction_classes=np.asarray(direction_rows, dtype=np.int64),
        tickers=np.asarray(ticker_rows, dtype=str),
        origin_dates=np.asarray(date_rows, dtype="datetime64[D]"),
        origin_closes=np.asarray(close_rows, dtype=np.float64),
        horizons=contract.horizons,
        feature_names=contract.feature_names,
    )
