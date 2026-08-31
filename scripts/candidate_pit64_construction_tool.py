#!/usr/bin/env python3
"""Package operator-supplied PIT64 source material for independent review.

The tool performs deterministic structural checks only. It never invents CIKs,
FIGIs, membership dates, source digests, or reviewer approval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import date
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("PIT64 candidate must contain a JSON object")
    return payload


def package_candidate(universe_path: Path, sources_path: Path, output_path: Path) -> dict[str, Any]:
    """Write an unsigned review request bound to exact candidate/source bytes."""

    resolved_inputs = {universe_path.resolve(), sources_path.resolve()}
    if output_path.resolve() in resolved_inputs:
        raise ValueError("PIT64 review request must not overwrite an input file")
    universe = _json_object(universe_path)
    sources = _json_object(sources_path)
    securities = universe.get("securities")
    if not isinstance(securities, list) or len(securities) != 64:
        raise ValueError("PIT64 candidate must contain exactly 64 securities")
    required_identity = {"security_id", "cik", "figi", "exchange_mic"}
    unique_identity = {"security_id", "cik", "figi"}
    identities = {field: set() for field in unique_identity}
    for index, security in enumerate(securities):
        if not isinstance(security, dict) or any(
            not isinstance(security.get(field), str) or not security[field].strip()
            for field in required_identity
        ):
            raise ValueError(f"security {index} has incomplete permanent identity")
        for field in unique_identity:
            value = security[field].strip()
            if value in identities[field]:
                raise ValueError(f"security {index} duplicates {field}")
            identities[field].add(value)
        intervals = security.get("membership_intervals")
        if not isinstance(intervals, list) or not intervals:
            raise ValueError(f"security {index} has no membership interval evidence")
        for interval in intervals:
            if not isinstance(interval, dict) or any(
                not isinstance(interval.get(field), str) or not interval[field].strip()
                for field in ("start_date", "end_date", "source", "source_digest")
            ):
                raise ValueError(f"security {index} has an incomplete membership interval")
            try:
                start = date.fromisoformat(interval["start_date"])
                end = date.fromisoformat(interval["end_date"])
            except ValueError as error:
                raise ValueError(f"security {index} has an invalid membership date") from error
            if start > end:
                raise ValueError(f"security {index} has a reversed membership interval")
            source = interval["source"].strip()
            digest = interval["source_digest"].strip().lower()
            if source not in sources:
                raise ValueError(f"security {index} references an absent source")
            if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError(f"security {index} has an invalid source digest")
            source_record = sources[source]
            if isinstance(source_record, dict) and source_record.get("sha256") != digest:
                raise ValueError(f"security {index} source digest does not match the registry")
    if not sources:
        raise ValueError("PIT64 source registry must not be empty")
    request = {
        "artifact_role": "pit64_independent_review_request",
        "certification_eligible": False,
        "external_reviewer_verified": False,
        "universe_sha256": _sha256(universe_path),
        "sources_sha256": _sha256(sources_path),
        "security_count": 64,
        "review_scope": [
            "all membership transition boundaries",
            "all permanent identifiers",
            "all source digests",
            "random sample of non-transition sessions",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(request, indent=2, sort_keys=True), encoding="utf-8")
    return request


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = package_candidate(args.universe, args.sources, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
