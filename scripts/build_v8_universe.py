#!/usr/bin/env python3
"""Build an immutable v8 universe from supplied point-in-time source files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "research"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from volatility_forecasting.universe_ingest_v8 import (  # noqa: E402
    load_source_attestations,
    load_universe_members_csv,
    sha256_file,
    validate_attestation_evidence_files,
)
from volatility_forecasting.universe_v8 import (  # noqa: E402
    initial_v8_selection_policy,
    write_universe_manifest,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a checksummed point-in-time v8 universe manifest"
    )
    parser.add_argument("--members-csv", type=Path, required=True)
    parser.add_argument("--source-attestations", type=Path, required=True)
    parser.add_argument(
        "--evidence-file",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Immutable source input to checksum; repeat for every attested evidence file",
    )
    parser.add_argument("--selection-policy", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--diagnostic-allow-sparse",
        action="store_true",
        help="Produce an explicitly non-certifiable diagnostic manifest",
    )
    return parser.parse_args()


def _parse_evidence_files(values: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name.strip() or not raw_path.strip():
            raise ValueError("--evidence-file must use NAME=PATH")
        normalized = name.strip()
        if normalized in parsed:
            raise ValueError(f"duplicate evidence-file name {normalized!r}")
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise ValueError(f"evidence file does not exist: {path}")
        parsed[normalized] = path
    return parsed


def main() -> int:
    args = _parse_args()
    try:
        members_path = args.members_csv.resolve()
        attestations_path = args.source_attestations.resolve()
        evidence_files = _parse_evidence_files(args.evidence_file)
        members = load_universe_members_csv(members_path)
        attestations = load_source_attestations(attestations_path)
        source_checksums = {
            "members_csv": sha256_file(members_path),
            "source_attestations": sha256_file(attestations_path),
            **{name: sha256_file(path) for name, path in sorted(evidence_files.items())},
        }
        validate_attestation_evidence_files(attestations, source_checksums)

        if args.selection_policy:
            policy = json.loads(args.selection_policy.read_text(encoding="utf-8"))
            if not isinstance(policy, dict):
                raise ValueError("selection policy must be a JSON object")
            source_checksums["selection_policy"] = sha256_file(
                args.selection_policy.resolve()
            )
        else:
            policy = initial_v8_selection_policy()
        if args.diagnostic_allow_sparse:
            policy = {**policy, "allow_sparse": True}

        output = write_universe_manifest(
            args.out_dir.resolve(),
            members,
            source_checksums=source_checksums,
            source_attestations=attestations,
            selection_policy=policy,
        )
        manifest = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"v8 universe build failed: {error}", file=sys.stderr)
        return 2

    print(f"v8 universe manifest: {output}")
    print(f"sha256: {manifest['sha256']}")
    print(f"members: {manifest['total_members']}")
    print(f"per_exchange: {manifest['per_exchange_counts']}")
    print(f"coverage_certifiable: {manifest['coverage_certifiable']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
