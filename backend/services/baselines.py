"""Honest, deterministic serving fallbacks when no learned candidate qualifies."""

from __future__ import annotations

import numpy as np


def persistence_price_forecast(closing_prices, days: int) -> tuple[list[float], dict]:
    """Repeat the latest observed close and label the output as a baseline."""

    values = np.asarray(closing_prices, dtype=float).reshape(-1)
    if days < 1 or not len(values) or not np.isfinite(values).all() or values[-1] <= 0:
        raise ValueError("Persistence forecast requires positive finite history and horizon.")
    return [float(values[-1])] * days, {
        "metric_source": "baseline_definition",
        "metric_scope": "not_out_of_sample_evidence",
        "relative_mae": 1.0,
        "relative_rmse": 1.0,
        "detail": "Persistence repeats the latest observed close.",
    }


def base_rate_direction_forecast(
    closing_prices,
    days: int,
    *,
    lookback: int = 252,
) -> tuple[list[str], list[float], dict]:
    """Return a smoothed recent up-session probability for every horizon."""

    values = np.asarray(closing_prices, dtype=float).reshape(-1)
    if days < 1 or len(values) < 3 or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Base-rate forecast requires positive finite price history.")
    changes = np.diff(values[-(lookback + 1) :])
    up_count = int(np.sum(changes > 0))
    # Beta(1, 1) smoothing avoids unjustified probabilities of exactly zero or one.
    probability = float((up_count + 1) / (len(changes) + 2))
    label = "Up" if probability >= 0.5 else "Down"
    return [label] * days, [probability] * days, {
        "metric_source": "baseline_definition",
        "metric_scope": "recent_observed_base_rate",
        "naive_baseline": round(max(probability, 1 - probability), 6),
        "detail": "Direction probability is the smoothed recent up-session frequency.",
    }
