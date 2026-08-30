"""Versioned contracts for return distribution, three-way direction, and volatility forecasting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

# Version constants
RETURN_CONTRACT_V1 = "price-return-distribution-v1"
DIRECTION_CONTRACT_V2 = "direction-cumulative-three-way-v2"
VOLATILITY_CONTRACT_V2 = "future-rv-total-v2"

SUPPORTED_HORIZONS = (1, 3, 5, 7, 10, 20, 30)


@dataclass(frozen=True)
class PriceReturnDistributionContract:
    """Contract for cumulative log-return distribution and anchored price reconstruction."""

    contract_id: str = RETURN_CONTRACT_V1
    version: int = 1

    @staticmethod
    def compute_cumulative_log_returns(close_prices: np.ndarray, horizon: int) -> np.ndarray:
        """Compute R_{t,h} = log(Close_{t+h} / Close_t)."""
        if horizon < 1:
            raise ValueError("Horizon must be at least 1.")
        close_arr = np.asarray(close_prices, dtype=float)
        if len(close_arr) <= horizon:
            raise ValueError("Insufficient close prices for requested horizon.")
        if (close_arr <= 0).any() or not np.isfinite(close_arr).all():
            raise ValueError("Close prices must be strictly positive and finite.")
        return np.log(close_arr[horizon:] / close_arr[:-horizon])

    @staticmethod
    def reconstruct_anchored_prices(
        base_price: float,
        predicted_cumulative_returns_median: np.ndarray | list[float],
    ) -> list[float]:
        """Reconstruct P_{t+h}^{median} = P_0 * exp(R_{t,h}^{median}).

        Guarantees that Day 0 anchor is exact.
        """
        if base_price <= 0 or not math.isfinite(base_price):
            raise ValueError("Base price P0 must be strictly positive and finite.")
        rets = np.asarray(predicted_cumulative_returns_median, dtype=float)
        if not np.isfinite(rets).all():
            raise ValueError("Predicted returns contain non-finite values.")
        reconstructed = base_price * np.exp(rets)
        return [float(p) for p in reconstructed]

    @staticmethod
    def format_anchored_forecast_response(
        ticker: str,
        base_date: str,
        base_price: float,
        horizons: list[int],
        predicted_medians: list[float],
        predicted_intervals: dict[str, list[float]] | None = None,
    ) -> dict[str, Any]:
        """Produce standard API response containing explicit Day 0 base and anchored horizons."""
        reconstructed_prices = PriceReturnDistributionContract.reconstruct_anchored_prices(
            base_price, predicted_medians
        )
        return {
            "contract_id": RETURN_CONTRACT_V1,
            "ticker": ticker,
            "base_date": base_date,
            "base_price": float(base_price),
            "horizons": [
                {
                    "horizon": int(h),
                    "predicted_cumulative_return_median": float(ret),
                    "reconstructed_median_price": float(price),
                    "intervals": {
                        k: float(v[i]) for k, v in (predicted_intervals or {}).items() if i < len(v)
                    },
                }
                for i, (h, ret, price) in enumerate(
                    zip(horizons, predicted_medians, reconstructed_prices, strict=False)
                )
            ],
        }


@dataclass(frozen=True)
class DirectionThreeWayContract:
    """Contract for cumulative three-way direction forecasting with neutral band."""

    contract_id: str = DIRECTION_CONTRACT_V2
    version: int = 2

    @staticmethod
    def compute_direction_labels(
        cumulative_log_returns: np.ndarray,
        tau: float | np.ndarray,
    ) -> np.ndarray:
        """Categorize into: -1 (Down), 0 (Neutral), +1 (Up).

        Down if R < -tau
        Neutral if |R| <= tau
        Up if R > tau
        """
        rets = np.asarray(cumulative_log_returns, dtype=float)
        tau_val = np.asarray(tau, dtype=float)
        if (tau_val < 0).any() or not np.isfinite(tau_val).all():
            raise ValueError("Neutral threshold tau must be non-negative and finite.")

        labels = np.zeros_like(rets, dtype=int)
        labels[rets > tau_val] = 1
        labels[rets < -tau_val] = -1
        return labels


@dataclass(frozen=True)
class FutureRealizedVarianceContract:
    """Contract for multi-horizon cumulative realized variance forecasting."""

    contract_id: str = VOLATILITY_CONTRACT_V2
    version: int = 2

    @staticmethod
    def compute_cumulative_target(daily_variances: np.ndarray, horizon: int) -> np.ndarray:
        """Compute V_{t,h} = sum_{k=1}^h v_{t+k}."""
        if horizon < 1:
            raise ValueError("Horizon must be at least 1.")
        var_arr = np.asarray(daily_variances, dtype=float)
        if (var_arr <= 0).any() or not np.isfinite(var_arr).all():
            raise ValueError("Daily variances must be strictly positive and finite.")
        n = len(var_arr) - horizon
        if n <= 0:
            raise ValueError("Insufficient daily variance observations for horizon.")
        # Convolution sum
        kernel = np.ones(horizon, dtype=float)
        rolled = np.convolve(var_arr, kernel, mode="valid")
        # Target starts from t+1
        return rolled[1:]
