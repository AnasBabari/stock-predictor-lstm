from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from services.bundle_retention_v8 import (
    RetentionError,
    RetentionPolicy,
    discover_release_inventory,
    execute_release_gc,
    plan_release_gc,
)

NOW = datetime(2026, 8, 27, 12, tzinfo=UTC)


def _release(root: Path, name: str, age_days: int) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    payload = f"model:{name}".encode()
    (directory / "model.onnx").write_bytes(payload)
    manifest = {
        "schema_version": 1,
        "created_at_utc": (NOW - timedelta(days=age_days)).isoformat(),
        "metadata": {"model_id": name},
        "files": {"model.onnx": hashlib.sha256(payload).hexdigest()},
        "signature": "fixture-signature",
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return directory


def _inventory(root: Path) -> dict[str, str]:
    return {record.model_id: record.release_id for record in discover_release_inventory(root)}


def test_plan_protects_active_previous_lease_audit_and_minimum(tmp_path: Path) -> None:
    for name, age in (("old", 80), ("leased", 70), ("expired", 60), ("previous", 5), ("active", 1)):
        _release(tmp_path, name, age)
    identities = _inventory(tmp_path)
    policy = RetentionPolicy(
        active_release_id=identities["active"],
        previous_release_id=identities["previous"],
        audit_retention_days=30,
        minimum_releases_to_keep=2,
    )

    plan = plan_release_gc(
        tmp_path,
        policy,
        in_use_release_ids=[identities["leased"]],
        now=NOW,
    )
    by_path = {decision.path: decision for decision in plan.decisions}

    assert by_path["active"].reason == "active_release"
    assert by_path["previous"].reason == "previous_known_good"
    assert by_path["leased"].reason == "loader_lease"
    assert by_path["old"].action == "delete"
    assert by_path["expired"].action == "delete"


def test_dry_run_audits_without_deleting(tmp_path: Path) -> None:
    _release(tmp_path, "old", 80)
    _release(tmp_path, "previous", 5)
    _release(tmp_path, "active", 1)
    identities = _inventory(tmp_path)
    plan = plan_release_gc(
        tmp_path,
        RetentionPolicy(
            active_release_id=identities["active"],
            previous_release_id=identities["previous"],
            minimum_releases_to_keep=2,
            dry_run=True,
        ),
        now=NOW,
    )
    audit = tmp_path / "audit" / "retention.jsonl"

    assert execute_release_gc(plan, audit_log=audit) == ()
    assert (tmp_path / "old").exists()
    assert json.loads(audit.read_text(encoding="utf-8"))["event"] == "retention_dry_run"


def test_execute_deletes_only_planned_release_and_expired_stage(tmp_path: Path) -> None:
    _release(tmp_path, "old", 80)
    _release(tmp_path, "previous", 5)
    _release(tmp_path, "active", 1)
    stage = tmp_path / ".staged-abandoned"
    stage.mkdir()
    old_time = (NOW - timedelta(days=3)).timestamp()
    os.utime(stage, (old_time, old_time))
    identities = _inventory(tmp_path)
    plan = plan_release_gc(
        tmp_path,
        RetentionPolicy(
            active_release_id=identities["active"],
            previous_release_id=identities["previous"],
            minimum_releases_to_keep=2,
            dry_run=False,
        ),
        now=NOW,
    )

    deleted = execute_release_gc(plan, audit_log=tmp_path / "retention.jsonl")

    assert {path.name for path in deleted} == {"old", ".staged-abandoned"}
    assert (tmp_path / "active").is_dir()
    assert (tmp_path / "previous").is_dir()
    events = [
        json.loads(line)["event"]
        for line in (tmp_path / "retention.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert events.count("release_delete_completed") == 2


def test_inventory_tamper_and_unknown_pointer_fail_closed(tmp_path: Path) -> None:
    release = _release(tmp_path, "active", 1)
    (release / "model.onnx").write_bytes(b"tampered")
    with pytest.raises(RetentionError, match="checksum mismatch"):
        discover_release_inventory(tmp_path)

    for child in release.iterdir():
        child.unlink()
    release.rmdir()
    _release(tmp_path, "active", 1)
    with pytest.raises(RetentionError, match="active release is absent"):
        plan_release_gc(
            tmp_path,
            RetentionPolicy(active_release_id="release-unknown"),
            now=NOW,
        )


def test_execute_rejects_inventory_race(tmp_path: Path) -> None:
    _release(tmp_path, "old", 80)
    _release(tmp_path, "previous", 5)
    _release(tmp_path, "active", 1)
    identities = _inventory(tmp_path)
    plan = plan_release_gc(
        tmp_path,
        RetentionPolicy(
            active_release_id=identities["active"],
            previous_release_id=identities["previous"],
            minimum_releases_to_keep=2,
            dry_run=False,
        ),
        now=NOW,
    )
    _release(tmp_path, "concurrent", 0)

    with pytest.raises(RetentionError, match="inventory changed"):
        execute_release_gc(plan, audit_log=tmp_path / "retention.jsonl")
    assert (tmp_path / "old").exists()
