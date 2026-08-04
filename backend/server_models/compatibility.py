"""Compatibility and freshness checks against the frozen browser contract.

Mirrors the frontend ``validCachedModel`` decision: a server record is only
usable when it matches schema v4, the exact 28-feature order, the frozen
target mode and the 60-step window.  Freshness is a separate, explicit state.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from config import (
    FEATURES_V4,
    SNAPSHOT_SCHEMA_VERSION,
    TARGET_MODE,
    WINDOW_SIZE,
    settings,
)
from server_models.contracts import CompatibilityReport, ServerModelRecord


def check_record_compatibility(record: ServerModelRecord) -> CompatibilityReport:
    """Validate a server artifact record against the frozen schema-v4 contract."""

    key = record.key
    if key.schema_version != SNAPSHOT_SCHEMA_VERSION:
        return CompatibilityReport(
            compatible=False,
            reason=f"schema_version {key.schema_version} != frozen v{SNAPSHOT_SCHEMA_VERSION}",
        )
    features = record.reproducibility.feature_names
    if list(features) != list(FEATURES_V4):
        return CompatibilityReport(
            compatible=False,
            reason="feature_names do not match the frozen FEATURES_V4 order",
        )
    if record.reproducibility.target_mode != TARGET_MODE:
        return CompatibilityReport(
            compatible=False,
            reason=f"target_mode '{record.reproducibility.target_mode}' != '{TARGET_MODE}'",
        )
    if record.reproducibility.window_size != WINDOW_SIZE:
        return CompatibilityReport(
            compatible=False,
            reason=f"window_size {record.reproducibility.window_size} != {WINDOW_SIZE}",
        )
    return CompatibilityReport(compatible=True, reason="compatible with schema v4 contract")


def is_fresh(trained_at: datetime, max_age_hours: int | None = None) -> bool:
    """Return True when an artifact is younger than the freshness threshold."""

    limit_hours = settings.server_forecast_max_age_hours if max_age_hours is None else max_age_hours
    moment = trained_at
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return datetime.now(UTC) - moment <= timedelta(hours=limit_hours)
