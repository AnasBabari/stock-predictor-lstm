"""Promotion gates for server-pretrained bundles, harmonized with the browser
policy (``frontend/src/ml/promotionPolicy.js``).

The browser gates a learned candidate on pooled relative metrics, day-1 and
selected-horizon persistence caps, a minimum number of evaluated origins, and
volatility plausibility. Server training applies the same checks to the
walk-forward report so the server and browser engines promote under one
standard, while the runner's own promotion decision (fold wins, fold
degradation cap, MASE/RMSSE) is always required first.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

MAXIMUM_RELATIVE_MAE = 0.98
MAXIMUM_RELATIVE_RMSE = 0.98
HARD_HORIZON_RELATIVE_CAP = 1.0
MINIMUM_EVALUATION_ROWS = 60
MAXIMUM_VOLATILITY_MULTIPLE = 4.0
VOLATILITY_PERCENTILE = 0.995
RECENT_VOLATILITY_WINDOW = 60


def _relative(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _horizon_beats_persistence(entry_mae: float | None, entry_rmse: float | None) -> bool:
    return (
        entry_mae is not None
        and entry_rmse is not None
        and entry_mae < HARD_HORIZON_RELATIVE_CAP
        and entry_rmse < HARD_HORIZON_RELATIVE_CAP
    )


def _horizon_rows(per_horizon: dict[str, Any], horizon: int) -> int:
    entry = per_horizon.get(str(horizon))
    if not isinstance(entry, dict):
        return 0
    value = entry.get("rows")
    if value is None:
        value = entry.get("sample_count")
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _volatility_plausible(
    close_values: Any, predicted_cumulative_return: float | None, horizon: int
) -> bool:
    """Mirror the browser volatility gate: the learned cumulative return must
    stay inside the historically observed horizon-return range."""
    if predicted_cumulative_return is None:
        return True
    if not math.isfinite(predicted_cumulative_return):
        return False
    prices = np.asarray(close_values, dtype=np.float64)
    if len(prices) < 2:
        return True
    log_returns = np.diff(np.log(prices))
    recent = log_returns[-RECENT_VOLATILITY_WINDOW:]
    if len(recent) < 30:
        return True
    daily_vol = float(np.std(recent, ddof=1))
    horizon_vol = daily_vol * math.sqrt(max(1, horizon))
    multiple_limit = MAXIMUM_VOLATILITY_MULTIPLE * horizon_vol
    predicted = abs(float(predicted_cumulative_return))
    if predicted > multiple_limit:
        return False
    if len(prices) > horizon:
        history = np.abs(np.log(prices[horizon:] / prices[:-horizon]))
        if len(history) >= 30:
            percentile_limit = float(np.percentile(history, VOLATILITY_PERCENTILE * 100))
            if predicted > percentile_limit:
                return False
    return True


def assess_server_promotion(
    report: dict[str, Any],
    *,
    selected_horizon: int,
    close_values: Any = None,
    predicted_cumulative_return: float | None = None,
) -> tuple[bool, list[str]]:
    """Evaluate one candidate report against the harmonized promotion gates.

    Returns ``(promoted, reasons)``. Pass ``predicted_cumulative_return`` and
    ``close_values`` to also enforce the volatility gate (requires a forecast).
    """
    reasons: list[str] = []
    aggregate = report.get("aggregate") or {}
    pooled = aggregate.get("pooled") or {}
    per_horizon = aggregate.get("per_horizon") or {}
    if not isinstance(per_horizon, dict):
        per_horizon = {
            str(entry.get("horizon")): entry
            for entry in per_horizon
            if isinstance(entry, dict) and entry.get("horizon") is not None
        }
    promotion = report.get("promotion") or {}

    if not promotion.get("promoted"):
        reasons.extend(
            promotion.get("reasons")
            or ["The walk-forward evaluation did not pass the promotion policy."]
        )

    relative_mae = _relative(pooled.get("relative_mae"))
    relative_rmse = _relative(pooled.get("relative_rmse"))
    if relative_mae is None or relative_mae >= MAXIMUM_RELATIVE_MAE:
        reasons.append("Relative MAE did not beat persistence.")
    if relative_rmse is None or relative_rmse >= MAXIMUM_RELATIVE_RMSE:
        reasons.append("Relative RMSE did not beat persistence.")

    horizon_checks = [("one-day", 1)]
    if selected_horizon != 1:
        horizon_checks.append(("selected", selected_horizon))
    for label, h in horizon_checks:
        entry = per_horizon.get(str(h))
        if not isinstance(entry, dict):
            reasons.append(f"The {label} horizon is missing from the evaluation report.")
            continue
        entry_mae = _relative(entry.get("relative_mae"))
        entry_rmse = _relative(entry.get("relative_rmse"))
        if not _horizon_beats_persistence(entry_mae, entry_rmse):
            reasons.append(f"The model did not beat persistence at the {label} horizon.")

    rows = _horizon_rows(per_horizon, selected_horizon)
    if rows < MINIMUM_EVALUATION_ROWS:
        reasons.append("The selected horizon has too few evaluated observations.")

    if (
        predicted_cumulative_return is not None
        and close_values is not None
        and not _volatility_plausible(close_values, predicted_cumulative_return, selected_horizon)
    ):
        reasons.append("The learned forecast exceeded its historically observed volatility range.")

    return len(reasons) == 0, reasons
