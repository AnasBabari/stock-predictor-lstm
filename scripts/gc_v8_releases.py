#!/usr/bin/env python3
"""Plan or execute local signed-release retention with explicit pointers."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from services.bundle_retention_v8 import (  # noqa: E402
    RetentionPolicy,
    discover_release_inventory,
    execute_release_gc,
    plan_release_gc,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply signed v8 release retention")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--active-release-id")
    parser.add_argument("--previous-release-id")
    parser.add_argument("--in-use-release-id", action="append", default=[])
    parser.add_argument("--audit-retention-days", type=int, default=30)
    parser.add_argument("--minimum-releases-to-keep", type=int, default=3)
    parser.add_argument("--staged-retention-hours", type=int, default=24)
    parser.add_argument("--audit-log", type=Path)
    parser.add_argument(
        "--list-inventory",
        action="store_true",
        help="Print verified release IDs without planning deletion",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Delete expired releases; omission is a dry run",
    )
    args = parser.parse_args()
    try:
        if args.list_inventory:
            inventory = discover_release_inventory(args.root.resolve())
            print(
                json.dumps(
                    [asdict(record) for record in inventory],
                    indent=2,
                    sort_keys=True,
                    default=str,
                )
            )
            return 0
        if not args.active_release_id or args.audit_log is None:
            parser.error("--active-release-id and --audit-log are required unless listing")
        policy = RetentionPolicy(
            active_release_id=args.active_release_id,
            previous_release_id=args.previous_release_id,
            audit_retention_days=args.audit_retention_days,
            minimum_releases_to_keep=args.minimum_releases_to_keep,
            staged_retention_hours=args.staged_retention_hours,
            dry_run=not args.execute,
        )
        plan = plan_release_gc(
            args.root.resolve(),
            policy,
            in_use_release_ids=args.in_use_release_id,
        )
        deleted = execute_release_gc(plan, audit_log=args.audit_log.resolve())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"v8 retention failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(asdict(plan), indent=2, sort_keys=True))
    print(f"deleted: {len(deleted)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
