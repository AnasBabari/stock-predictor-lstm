"""Small production volatility forecaster built from causal market features.

The active application deliberately uses these transparent baselines until an
offline learned model has been evaluated on the same historical target.  No
model files are written and no certification or release gate is involved.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from services.volatility_snapshot import VolatilityInferenceSnapshot

SUPPORTED_BASELINES = ("auto", "rolling_mean", "har_rv", "ewma", "persistence")
_QUANTILE_Z = {
    "p05": -1.6448536269514722,
    "p10": -1.2815515655446004,
    "p25": -0.6744897501960817,
    "p50": 0.0,
    "p75": 0.6744897501960817,
    "p90": 1.2815515655446004,
    "p95": 1.6448536269514722,
}
_HORIZONS = (1, 3, 5, 7, 14, 30)
_TRADING_SESSIONS_PER_YEAR = 252.0


def _candidate_name(model: str, horizon: int = 5) -> str:
    name = str(model).strip().lower()
    if name not in SUPPORTED_BASELINES:
        raise ValueError(f"model must be one of {', '.join(SUPPORTED_BASELINES)}")
    if name == "auto":
        # Empirically selected validation winner across multi-horizon benchmarks
        return "rolling_mean"
    return name


def _variance_for(snapshot: VolatilityInferenceSnapshot, model: str, horizon: int) -> float:
    name = _candidate_name(model, horizon)
    if horizon not in _HORIZONS:
        raise ValueError(f"horizon must be one of {list(_HORIZONS)}")
    mapping = {
        "persistence": "rolling_c2c_20",
        "rolling_mean": "rolling_c2c_60",
        "ewma": "riskmetrics_ewma_c2c",
        "har_rv": "causal_log_har",
    }
    candidate = snapshot.baseline_candidates.get(mapping[name])
    if candidate is None:
        raise ValueError(f"baseline candidate {mapping[name]!r} is unavailable")
    index = _HORIZONS.index(horizon)
    variance = float(np.asarray(candidate, dtype=np.float64)[index])
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("baseline produced a non-positive variance")
    return variance


def _price_quantiles(
    current_price: float, cumulative_variance: float | np.ndarray, horizon: int
) -> dict[str, list[float]]:
    raw_path = np.asarray(cumulative_variance, dtype=np.float64).reshape(-1)
    if raw_path.size == 1:
        terminal = float(raw_path[0])
        if not np.isfinite(terminal) or terminal <= 0:
            raise ValueError("cumulative variance must be finite and positive")
        path = terminal * np.arange(1, horizon + 1, dtype=np.float64) / horizon
    else:
        if raw_path.size < horizon:
            raise ValueError("cumulative variance path is shorter than the requested horizon")
        path = raw_path[:horizon]
        if not np.isfinite(path).all() or (path <= 0).any() or np.any(np.diff(path) < -1e-12):
            raise ValueError("cumulative variance path must be finite, positive, and cumulative")
    return {
        name: [
            float(current_price * np.exp(score * np.sqrt(path[day - 1])))
            for day in range(1, horizon + 1)
        ]
        for name, score in _QUANTILE_Z.items()
    }


def _variance_path_for(
    snapshot: VolatilityInferenceSnapshot, model: str, horizon: int
) -> np.ndarray:
    """Return a cumulative variance path with horizon-coherent interpolation."""

    name = _candidate_name(model, horizon)
    mapping = {
        "persistence": "rolling_c2c_20",
        "rolling_mean": "rolling_c2c_60",
        "ewma": "riskmetrics_ewma_c2c",
        "har_rv": "causal_log_har",
    }
    key = mapping[name]
    paths = getattr(snapshot, "baseline_variance_paths", None) or {}
    path = paths.get(key)
    if path is not None:
        values = np.asarray(path, dtype=np.float64).reshape(-1)
        prefix = values[:horizon]
        if (
            values.size >= horizon
            and np.isfinite(prefix).all()
            and (prefix > 0).all()
            and not np.any(np.diff(prefix) < -1e-12)
        ):
            return prefix
    # Compatibility snapshots created by older callers only carry terminal
    # candidates.  Their linear path is mathematically equivalent to the old
    # random-walk-in-variance assumption and remains explicitly bounded.
    terminal = _variance_for(snapshot, name, horizon)
    return terminal * np.arange(1, horizon + 1, dtype=np.float64) / horizon


def build_live_volatility_forecast(
    snapshot: VolatilityInferenceSnapshot,
    *,
    horizon: int,
    model: str = "auto",
) -> dict[str, Any]:
    """Return a transparent volatility cone and its matched baseline metadata."""

    model_name = _candidate_name(model, horizon)
    variance = _variance_for(snapshot, model_name, horizon)
    variance_path = _variance_path_for(snapshot, model_name, horizon)
    annualized = float(np.sqrt(variance / horizon * _TRADING_SESSIONS_PER_YEAR))
    if not np.isfinite(annualized) or annualized <= 0:
        raise ValueError("baseline produced an invalid annualized volatility")
    future_dates = tuple(snapshot.future_dates[:horizon])
    if len(future_dates) != horizon:
        raise ValueError("calendar did not provide the requested forecast horizon")
    return {
        "ticker": snapshot.ticker,
        "as_of": snapshot.origin_date,
        "horizon": horizon,
        "current_price": float(snapshot.origin_close),
        "historical_dates": list(snapshot.historical_dates),
        "historical_prices": snapshot.historical_prices.astype(float).tolist(),
        "forecast": {
            "future_dates": list(future_dates),
            "price_quantiles": _price_quantiles(snapshot.origin_close, variance_path, horizon),
            "expected_cumulative_variance_path": variance_path.astype(float).tolist(),
            "expected_cumulative_variance": variance,
            "expected_annualized_volatility": annualized,
            "predicted_volatility": annualized,
            "volatility_unit": "annualized_sigma",
            "model": model_name,
            "baseline": True,
        },
        "evidence": {
            "model_status": "baseline",
            "model_family": "statistical_baseline",
            "model_name": model_name,
            "baseline": True,
            "metric_source": "baseline_definition",
            "interval_method": "gaussian_reference_scenario",
            "interval_nominal_coverage": 0.90,
            "interval_scope": "pointwise_marginal_reference_not_empirically_calibrated",
            "target": "future_realized_volatility_close_to_close",
            "target_definition": "sqrt(252 / H * sum(next H close-to-close log returns squared))",
            "snapshot_id": snapshot.snapshot_id,
            "data_provider": getattr(snapshot, "data_provider", "unknown"),
            "data_as_of": snapshot.origin_date,
            "market_data_cache": getattr(snapshot, "market_data_cache", "unknown"),
            "schema_version": "deployable_v5",
            "feature_count": len(snapshot.feature_names),
            "feature_names": list(snapshot.feature_names),
            "window_size": len(snapshot.features),
            "news_status": "not_used",
        },
    }


__all__ = ["SUPPORTED_BASELINES", "build_live_volatility_forecast"]
