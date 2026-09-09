"""Canonical contract check for normalized provider frames."""

from __future__ import annotations

import pandas as pd

from .schemas import NORMALIZED_COLUMNS


def check_normalized(frame: pd.DataFrame) -> list[str]:
    """Return a list of contract violations (empty means conformant)."""
    errors: list[str] = []
    missing = [c for c in NORMALIZED_COLUMNS if c not in frame.columns]
    if missing:
        errors.append(f"missing columns: {missing}")
        return errors
    if frame.empty:
        errors.append("empty frame")
        return errors
    for col in ("dte_days", "strike", "underlying_px", "call_delta"):
        if not pd.api.types.is_numeric_dtype(frame[col]):
            errors.append(f"non-numeric column: {col}")
    if (pd.to_numeric(frame["dte_days"], errors="coerce") < 0).any():
        errors.append("negative dte_days present")
    if frame[["ticker", "trade_date", "expir_date"]].isna().any().any():
        errors.append("null keys present")
    return errors
