"""Explicit, idempotent migration of a local forecast ledger to PostgreSQL.

This command is intentionally separate from API startup.  It copies existing
records only; it never creates a live observation and never silently resolves
an immutable logical-key conflict.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for path in (ROOT, ROOT / "backend"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from services.forecast_ledger import ForecastLedger  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sqlite-path",
        type=Path,
        required=True,
        help="Existing SQLite ledger to copy; it must already exist.",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Destination PostgreSQL URL (or use DATABASE_URL in the environment).",
    )
    parser.add_argument(
        "--record-source",
        choices=("live", "historical_replay"),
        default=None,
        help="Optional source track to migrate; by default both tracks are copied.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and count without writing."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_path = args.sqlite_path.expanduser().resolve()
    if not source_path.exists():
        raise SystemExit(f"SQLite ledger does not exist: {source_path}")

    source = ForecastLedger(source_path)
    records = source.export_records(record_source=args.record_source)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "records": len(records)}, sort_keys=True))
        return 0

    destination = ForecastLedger(
        database_url=args.database_url,
        database_required=True,
    )
    inserted = destination.import_records(records)
    print(
        json.dumps(
            {
                "source": str(source_path),
                "records": len(records),
                "inserted": inserted,
                "destination_backend": destination.storage_kind,
                "idempotent": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
