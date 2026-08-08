"""Retention sweep for immutable server-forecast bundle objects."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from server_models.db import ModelRegistry
from server_models.storage import ObjectStore


def sweep_expired_bundles(
    registry: ModelRegistry,
    storage: ObjectStore,
    *,
    retention_days: int,
    now: datetime | None = None,
) -> list[str]:
    """Delete expired non-pointer blobs while preserving registry/audit rows.

    Candidate selection and deletion run inside the registry's promotion-lock
    snapshot. Both the current champion and the saved rollback target are
    protected, and pruned rows are tombstoned so they cannot later be promoted.
    """
    if retention_days < 1:
        raise ValueError("retention_days must be at least one day.")
    reference = now or datetime.now(UTC)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=UTC)
    cutoff = reference.astimezone(UTC) - timedelta(days=retention_days)
    return registry.prune_bundle_objects(cutoff, storage.delete_bundle)
