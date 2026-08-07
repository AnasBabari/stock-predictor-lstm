"""Normalize server-pretrained evaluation evidence into the browser metrics
contract.

The browser ``MetricsCard`` and the promotion UI read a flat ``metrics`` object
with top-level ``rmse``/``mae``/``relative_*``/``r2``/``mape`` keys plus a
``per_horizon`` list. Server walk-forward evaluation nests the same values
under ``pooled`` and a ``per_horizon`` dict keyed by horizon. Serving therefore
normalizes the evidence once so both engines render through the exact same
frontend contract and metric labels.
"""

from __future__ import annotations

from typing import Any

from config import TARGET_MODE


def _horizon_rows(entry: dict[str, Any]) -> int:
    """Evaluated origins for one horizon (the browser's ``rows`` field)."""
    value = entry.get("rows")
    if value is None:
        value = entry.get("sample_count")
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def build_served_metrics(evidence: dict[str, Any], *, horizon: int | None = None) -> dict[str, Any]:
    """Convert server ``evidence`` to the flat browser metric shape.

    ``per_horizon`` in the evidence may be a dict keyed by horizon or a list;
    both are accepted and emitted as a sorted list of horizon entries.
    """
    pooled = evidence.get("pooled") or {}
    raw_horizons = evidence.get("per_horizon") or {}
    if isinstance(raw_horizons, dict):
        items = ((str(key), entry) for key, entry in raw_horizons.items())
    else:
        items = (
            (str(entry.get("horizon")), entry) for entry in raw_horizons if isinstance(entry, dict)
        )

    per_horizon: list[dict[str, Any]] = []
    for key, entry in sorted(items, key=lambda item: int(item[0] or 0)):
        if not isinstance(entry, dict):
            continue
        per_horizon.append(
            {
                "horizon": int(key),
                "rows": _horizon_rows(entry),
                "mae": entry.get("mae"),
                "rmse": entry.get("rmse"),
                "mape": entry.get("mape"),
                "relative_mae": entry.get("relative_mae"),
                "relative_rmse": entry.get("relative_rmse"),
                "directional_accuracy": entry.get(
                    "directional_accuracy", entry.get("direction_accuracy")
                ),
            }
        )

    evaluation_rows = 0
    if horizon is not None:
        for entry in per_horizon:
            if entry["horizon"] == horizon:
                evaluation_rows = entry["rows"]
                break

    return {
        "metric_source": evidence.get("metric_source", "server_purged_walk_forward"),
        "metric_scope": pooled.get("metric_scope", "forecast_origin_horizon_pairs"),
        "family": evidence.get("family"),
        "target_mode": TARGET_MODE,
        "horizon": horizon,
        "mae": pooled.get("mae"),
        "mse": pooled.get("mse"),
        "rmse": pooled.get("rmse"),
        "mape": pooled.get("mape"),
        "r2": pooled.get("r2"),
        "relative_mae": pooled.get("relative_mae"),
        "relative_rmse": pooled.get("relative_rmse"),
        "directional_accuracy": pooled.get("direction_accuracy"),
        "per_horizon": per_horizon,
        "evaluation_rows": evaluation_rows,
    }
