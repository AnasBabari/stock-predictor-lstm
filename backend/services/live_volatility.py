"""Small production volatility forecaster built from causal market features.

The active application deliberately uses these transparent baselines until an
offline learned model has been evaluated on the same historical target.  No
model files are written and no certification or release gate is involved.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from services.volatility_snapshot import VolatilityInferenceSnapshot

SUPPORTED_BASELINES = ("persistence", "rolling_mean", "ewma", "har_rv")
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


def _candidate_name(model: str) -> str:
    name = str(model).strip().lower()
    if name not in SUPPORTED_BASELINES:
        raise ValueError(f"model must be one of {', '.join(SUPPORTED_BASELINES)}")
    return name


def _variance_for(snapshot: VolatilityInferenceSnapshot, model: str, horizon: int) -> float:
    name = _candidate_name(model)
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
    current_price: float, cumulative_variance: float, horizon: int
) -> dict[str, list[float]]:
    daily = np.sqrt(cumulative_variance / horizon)
    return {
        name: [
            float(current_price * np.exp(score * daily * np.sqrt(day)))
            for day in range(1, horizon + 1)
        ]
        for name, score in _QUANTILE_Z.items()
    }


def build_live_volatility_forecast(
    snapshot: VolatilityInferenceSnapshot,
    *,
    horizon: int,
    model: str = "har_rv",
) -> dict[str, Any]:
    """Return a transparent volatility cone and its matched baseline metadata."""

    model_name = _candidate_name(model)
    variance = _variance_for(snapshot, model_name, horizon)
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
            "price_quantiles": _price_quantiles(snapshot.origin_close, variance, horizon),
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
            "target": "future_realized_volatility_close_to_close",
            "target_definition": "sqrt(252 / H * sum(next H close-to-close log returns squared))",
            "snapshot_id": snapshot.snapshot_id,
            "schema_version": "deployable_v5",
            "feature_count": len(snapshot.feature_names),
            "feature_names": list(snapshot.feature_names),
            "window_size": len(snapshot.features),
            "news_status": "not_used",
        },
    }


__all__ = ["SUPPORTED_BASELINES", "build_live_volatility_forecast"]
