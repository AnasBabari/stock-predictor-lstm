"""Deterministic validation + study-eligibility gate for feature rows."""

from __future__ import annotations

import numpy as np

from .schemas import FEATURE_ROW_FIELDS, STUDY_ELIGIBLE_SOURCES

_IV_FIELDS = (
    "atm_iv_30d",
    "atm_iv_60d",
    "put_iv_25d",
    "call_iv_25d",
)


def check_feature_row(row: dict) -> list[str]:
    """Return contract violations (empty means a complete, sane row)."""
    errors: list[str] = []
    for field in FEATURE_ROW_FIELDS:
        if field not in row:
            errors.append(f"missing field: {field}")
    if errors:
        return errors
    numeric = [
        f for f in FEATURE_ROW_FIELDS if f not in ("date", "ticker", "source", "source_timestamp")
    ]
    for field in numeric:
        try:
            value = float(row[field])
        except (TypeError, ValueError):
            errors.append(f"non-numeric field: {field}")
            continue
        if not np.isfinite(value):
            errors.append(f"non-finite field: {field}")
    for field in _IV_FIELDS:
        try:
            value = float(row[field])
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and not 0.0 < value < 5.0:
            errors.append(f"IV out of (0,5) bounds: {field}={value}")
    for field in ("option_volume", "open_interest"):
        try:
            if float(row[field]) < 0:
                errors.append(f"negative flow field: {field}")
        except (TypeError, ValueError):
            pass
    return errors


def assert_study_eligible(row: dict) -> None:
    """Refuse any source outside the allow-listed historical vendor set.

    yfinance plumbing output (and any unknown source) raises here, making
    accidental merge into the study dataset structurally impossible.
    """
    source = str(row.get("source", ""))
    if source not in STUDY_ELIGIBLE_SOURCES:
        raise ValueError(
            f"Source '{source}' is not study eligible (allow-list: {list(STUDY_ELIGIBLE_SOURCES)})"
        )
