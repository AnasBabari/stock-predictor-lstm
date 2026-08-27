"""Bundle retention / GC stub for v8 (GitHub Issue #2).

This module is a placeholder for the production retention policy that must
be implemented before open beta.  The current behavior is fail-closed:
it refuses to delete anything and requires an explicit dry-run flag.

Required policy (to be implemented):
- Protect active release (pointer in Render/Vercel env or S3)
- Protect previous known-good release
- Protect audit-retention window (e.g. 30 days)
- Generation-aware locking (do not delete bundle being loaded)
- Dry-run mode (log but do not delete)
- Staged-object cleanup
- Audit logs for every deletion
- Never delete active, previous, or audit-required bundles

Until this is implemented, the correct operational posture is to retain
all bundles and alert on storage budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RetentionPolicy:
    active_pointer: str | None = None
    previous_pointer: str | None = None
    audit_retention_days: int = 30
    dry_run: bool = True


def dry_run_gc(bundles: list[Path], policy: RetentionPolicy) -> list[Path]:
    """Return list of bundles that *would* be deleted under policy, but do not delete.

    This is the only safe operation until the full GC with locking is implemented.
    """
    # Fail-closed: if active pointer not set, do not propose any deletion
    if not policy.active_pointer:
        return []
    # Placeholder: in real implementation, filter by created_at, generation, etc.
    # For now, return empty to avoid accidental deletion
    return []


def assert_retention_safety(bundles: list[Path], policy: RetentionPolicy) -> None:
    """Fail closed if any bundle that should be protected would be deleted."""
    to_delete = dry_run_gc(bundles, policy)
    protected = {policy.active_pointer, policy.previous_pointer}
    for p in to_delete:
        if str(p) in protected:
            raise RuntimeError(f"retention would delete protected bundle {p}")
