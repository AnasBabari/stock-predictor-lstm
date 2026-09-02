"""Export the forecast ledger deterministically without mixing live and replay tracks."""

from __future__ import annotations

import argparse
import csv
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
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--sqlite-path", type=Path, help="SQLite ledger path.")
    source.add_argument("--database-url", help="PostgreSQL URL (or use DATABASE_URL).")
    parser.add_argument(
        "--record-source",
        choices=("live", "historical_replay"),
        default=None,
        help="Export one track; omitting it includes an explicit source field per row.",
    )
    parser.add_argument("--format", choices=("json", "csv"), default="json")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sqlite_path is not None:
        ledger = ForecastLedger(args.sqlite_path.expanduser().resolve())
    elif args.database_url is not None:
        ledger = ForecastLedger(database_url=args.database_url, database_required=True)
    else:
        ledger = ForecastLedger()

    records = [
        record.to_dict() for record in ledger.export_records(record_source=args.record_source)
    ]
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "json":
        payload = {
            "record_source": args.record_source or "live_and_historical_replay",
            "storage_backend": ledger.storage_kind,
            "entries": records,
        }
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        fieldnames = list(records[0].keys()) if records else []
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if fieldnames:
                writer.writeheader()
                writer.writerows(records)
    print(json.dumps({"output": str(output), "records": len(records)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
