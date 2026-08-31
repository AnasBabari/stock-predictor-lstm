#!/usr/bin/env python3
"""Hash a candidate artifact tree without tagging, signing, or certifying it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_candidate_manifest(root: Path, output: Path, generation: str) -> dict[str, Any]:
    """Create a deterministic unsigned inventory outside the inventoried tree."""

    source = root.resolve()
    destination = output.resolve()
    if not source.is_dir():
        raise ValueError("candidate root must be an existing directory")
    try:
        destination.relative_to(source)
    except ValueError:
        pass
    else:
        raise ValueError("candidate manifest output must be outside the inventoried tree")
    files: dict[str, str] = {}
    for path in sorted(source.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ValueError("candidate tree must not contain symbolic links")
        if path.is_file():
            files[path.relative_to(source).as_posix()] = _sha256(path)
    if not files:
        raise ValueError("candidate root contains no files")
    tree_digest = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest = {
        "artifact_role": "unsigned_candidate_version_manifest",
        "certification_eligible": False,
        "generation": generation,
        "tree_sha256": tree_digest,
        "files": files,
        "next": "An independent reviewer must verify and Cosign-sign the frozen evidence manifest.",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generation", required=True)
    args = parser.parse_args()
    result = create_candidate_manifest(args.root, args.output, args.generation)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
