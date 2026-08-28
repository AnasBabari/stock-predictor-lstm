"""Fail-closed retention planning for immutable signed volatility releases.

Deletion is an explicit operator action, never part of request handling. The
planner protects the active and rollback releases, leases held by running
loaders, a time-based audit window, and a minimum newest-release count. It
refuses malformed or ambiguous inventories rather than guessing from mtimes.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Iterable
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any


class RetentionError(RuntimeError):
    """A release inventory or retention operation is unsafe."""


@dataclass(frozen=True)
class RetentionPolicy:
    active_release_id: str
    previous_release_id: str | None = None
    audit_retention_days: int = 30
    minimum_releases_to_keep: int = 3
    staged_retention_hours: int = 24
    dry_run: bool = True

    def __post_init__(self) -> None:
        if not self.active_release_id.strip():
            raise ValueError("active_release_id is required")
        if self.audit_retention_days < 1:
            raise ValueError("audit_retention_days must be positive")
        if self.minimum_releases_to_keep < 2:
            raise ValueError("minimum_releases_to_keep must preserve at least two releases")
        if self.staged_retention_hours < 1:
            raise ValueError("staged_retention_hours must be positive")


@dataclass(frozen=True)
class ReleaseRecord:
    release_id: str
    path: Path
    created_at_utc: datetime
    model_id: str
    manifest_sha256: str


@dataclass(frozen=True)
class StagedRecord:
    stage_id: str
    path: Path
    modified_at_utc: datetime


@dataclass(frozen=True)
class RetentionDecision:
    release_id: str
    path: str
    action: str
    reason: str


@dataclass(frozen=True)
class RetentionPlan:
    generated_at_utc: str
    root: str
    active_release_id: str
    previous_release_id: str | None
    dry_run: bool
    inventory_sha256: str
    decisions: tuple[RetentionDecision, ...]


def _parse_utc(value: object, *, context: str) -> datetime:
    if not isinstance(value, str):
        raise RetentionError(f"{context} creation timestamp is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise RetentionError(f"{context} creation timestamp is invalid") from error
    if parsed.tzinfo is None:
        raise RetentionError(f"{context} creation timestamp must be timezone-aware")
    return parsed.astimezone(UTC)


def _manifest_record(path: Path) -> ReleaseRecord:
    manifest_path = path / "manifest.json"
    try:
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise RetentionError(f"release manifest is unreadable: {manifest_path}") from error
    if not isinstance(manifest, dict):
        raise RetentionError(f"release manifest is not an object: {manifest_path}")
    metadata = manifest.get("metadata")
    files = manifest.get("files")
    signature = manifest.get("signature")
    if not isinstance(metadata, dict) or not isinstance(files, dict) or not files:
        raise RetentionError(f"release manifest is incomplete: {manifest_path}")
    if not isinstance(signature, str) or not signature:
        raise RetentionError(f"release manifest is unsigned: {manifest_path}")
    total_bytes = 0
    for relative_name, expected_checksum in files.items():
        if not isinstance(relative_name, str) or not isinstance(expected_checksum, str):
            raise RetentionError(f"release file table is invalid: {manifest_path}")
        relative = PurePosixPath(relative_name)
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise RetentionError(f"release file path is unsafe: {relative_name}")
        artifact = path.joinpath(*relative.parts).resolve()
        try:
            artifact.relative_to(path.resolve())
            payload = artifact.read_bytes()
        except (ValueError, OSError) as error:
            raise RetentionError(f"release artifact is unavailable: {relative_name}") from error
        total_bytes += len(payload)
        if len(expected_checksum) != 64 or hashlib.sha256(payload).hexdigest() != expected_checksum:
            raise RetentionError(f"release artifact checksum mismatch: {relative_name}")
    if total_bytes <= 0:
        raise RetentionError(f"release artifacts are empty: {manifest_path}")
    model_id = metadata.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise RetentionError(f"release model_id is missing: {manifest_path}")
    digest = hashlib.sha256(raw).hexdigest()
    return ReleaseRecord(
        release_id=f"release-{digest[:24]}",
        path=path.resolve(),
        created_at_utc=_parse_utc(manifest.get("created_at_utc"), context=path.name),
        model_id=model_id,
        manifest_sha256=digest,
    )


def discover_release_inventory(root: Path) -> tuple[ReleaseRecord, ...]:
    """Discover complete signed directories; staged directories are separate."""

    resolved = root.resolve()
    if not resolved.is_dir():
        raise RetentionError(f"release root does not exist: {resolved}")
    records: list[ReleaseRecord] = []
    for child in sorted(resolved.iterdir(), key=lambda item: item.name):
        if child.name.startswith("."):
            continue
        if child.is_symlink():
            raise RetentionError(f"release inventory contains a symlink: {child}")
        if child.is_dir():
            records.append(_manifest_record(child))
    identities = [record.release_id for record in records]
    if len(identities) != len(set(identities)):
        raise RetentionError("release inventory contains duplicate manifest identities")
    return tuple(records)


def _discover_staged_inventory(root: Path) -> tuple[StagedRecord, ...]:
    records: list[StagedRecord] = []
    for child in sorted(root.resolve().iterdir(), key=lambda item: item.name):
        if not child.name.startswith(".staged-"):
            continue
        if child.is_symlink() or not child.is_dir():
            raise RetentionError(f"staged inventory contains an unsafe path: {child}")
        modified = datetime.fromtimestamp(child.stat().st_mtime, tz=UTC)
        identity = hashlib.sha256(f"{child.name}:{modified.isoformat()}".encode()).hexdigest()[:24]
        records.append(
            StagedRecord(
                stage_id=f"stage-{identity}",
                path=child.resolve(),
                modified_at_utc=modified,
            )
        )
    return tuple(records)


def _inventory_digest(records: Iterable[ReleaseRecord], staged: Iterable[StagedRecord] = ()) -> str:
    payload: list[dict[str, str]] = [
        {
            "release_id": record.release_id,
            "path": record.path.name,
            "created_at_utc": record.created_at_utc.isoformat(),
            "model_id": record.model_id,
            "manifest_sha256": record.manifest_sha256,
        }
        for record in sorted(records, key=lambda item: item.release_id)
    ]
    payload.extend(
        {
            "release_id": record.stage_id,
            "path": record.path.name,
            "created_at_utc": record.modified_at_utc.isoformat(),
            "model_id": "staged",
            "manifest_sha256": "not_applicable",
        }
        for record in sorted(staged, key=lambda item: item.stage_id)
    )
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def plan_release_gc(
    root: Path,
    policy: RetentionPolicy,
    *,
    in_use_release_ids: Iterable[str] = (),
    now: datetime | None = None,
) -> RetentionPlan:
    """Return a deterministic keep/delete plan without changing storage."""

    reference = now or datetime.now(UTC)
    if reference.tzinfo is None:
        raise ValueError("retention reference time must be timezone-aware")
    reference = reference.astimezone(UTC)
    records = discover_release_inventory(root)
    staged = _discover_staged_inventory(root)
    by_id = {record.release_id: record for record in records}
    if policy.active_release_id not in by_id:
        raise RetentionError("active release is absent from the verified inventory")
    if policy.previous_release_id is not None and policy.previous_release_id not in by_id:
        raise RetentionError("previous release is absent from the verified inventory")
    leased = {str(value) for value in in_use_release_ids}
    unknown_leases = sorted(leased - set(by_id))
    if unknown_leases:
        raise RetentionError(
            "loader lease references unknown releases: " + ", ".join(unknown_leases)
        )

    newest = sorted(records, key=lambda item: (item.created_at_utc, item.release_id), reverse=True)
    minimum_keep = {record.release_id for record in newest[: policy.minimum_releases_to_keep]}
    cutoff = reference - timedelta(days=policy.audit_retention_days)
    protected = {
        policy.active_release_id,
        *(value for value in (policy.previous_release_id,) if value is not None),
        *leased,
        *minimum_keep,
    }
    decisions: list[RetentionDecision] = []
    for record in sorted(records, key=lambda item: (item.created_at_utc, item.release_id)):
        if record.release_id == policy.active_release_id:
            action, reason = "keep", "active_release"
        elif record.release_id == policy.previous_release_id:
            action, reason = "keep", "previous_known_good"
        elif record.release_id in leased:
            action, reason = "keep", "loader_lease"
        elif record.release_id in minimum_keep:
            action, reason = "keep", "minimum_newest_releases"
        elif record.created_at_utc >= cutoff:
            action, reason = "keep", "audit_retention_window"
        elif record.release_id in protected:  # defensive completeness
            action, reason = "keep", "protected"
        else:
            action, reason = "delete", "expired_unreferenced_release"
        decisions.append(
            RetentionDecision(
                release_id=record.release_id,
                path=record.path.name,
                action=action,
                reason=reason,
            )
        )
    staged_cutoff = reference - timedelta(hours=policy.staged_retention_hours)
    for record in staged:
        expired = record.modified_at_utc < staged_cutoff
        decisions.append(
            RetentionDecision(
                release_id=record.stage_id,
                path=record.path.name,
                action="delete" if expired else "keep",
                reason=("expired_staged_object" if expired else "staged_retention_window"),
            )
        )
    return RetentionPlan(
        generated_at_utc=reference.isoformat(),
        root=str(root.resolve()),
        active_release_id=policy.active_release_id,
        previous_release_id=policy.previous_release_id,
        dry_run=policy.dry_run,
        inventory_sha256=_inventory_digest(records, staged),
        decisions=tuple(decisions),
    )


@contextmanager
def _exclusive_retention_lock(root: Path):
    lock_path = root.resolve() / ".retention.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RetentionError("another retention operation holds the generation lock") from error
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        os.fsync(descriptor)
        os.close(descriptor)
        yield
    finally:
        with suppress(OSError):
            os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def _audit(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def execute_release_gc(
    plan: RetentionPlan,
    *,
    audit_log: Path,
) -> tuple[Path, ...]:
    """Revalidate inventory under lock, audit, then delete planned releases."""

    root = Path(plan.root).resolve()
    if plan.dry_run:
        _audit(
            audit_log,
            {"event": "retention_dry_run", "plan": _plan_payload(plan)},
        )
        return ()
    deleted: list[Path] = []
    with _exclusive_retention_lock(root):
        records = discover_release_inventory(root)
        staged = _discover_staged_inventory(root)
        if _inventory_digest(records, staged) != plan.inventory_sha256:
            raise RetentionError("release inventory changed after planning; regenerate the plan")
        by_name = {record.path.name: record for record in records}
        staged_by_name = {record.path.name: record for record in staged}
        for decision in plan.decisions:
            if decision.action != "delete":
                continue
            record = by_name.get(decision.path)
            staged_record = staged_by_name.get(decision.path)
            identity = (
                record.release_id
                if record is not None
                else (staged_record.stage_id if staged_record is not None else None)
            )
            path = (
                record.path
                if record is not None
                else (staged_record.path if staged_record is not None else None)
            )
            if identity != decision.release_id or path is None:
                raise RetentionError("planned release identity changed before deletion")
            try:
                path.relative_to(root)
            except ValueError as error:
                raise RetentionError("planned release escapes the retention root") from error
            if path.parent != root or path.is_symlink():
                raise RetentionError("planned release path is not a direct safe child")
            _audit(
                audit_log,
                {
                    "event": "release_delete_started",
                    "release_id": identity,
                    "manifest_sha256": (
                        record.manifest_sha256 if record is not None else "not_applicable"
                    ),
                    "path": path.name,
                    "reason": decision.reason,
                },
            )
            shutil.rmtree(path)
            deleted.append(path)
            _audit(
                audit_log,
                {
                    "event": "release_delete_completed",
                    "release_id": identity,
                    "path": path.name,
                },
            )
    return tuple(deleted)


def _plan_payload(plan: RetentionPlan) -> dict[str, Any]:
    payload = asdict(plan)
    payload["decisions"] = [asdict(decision) for decision in plan.decisions]
    return payload
