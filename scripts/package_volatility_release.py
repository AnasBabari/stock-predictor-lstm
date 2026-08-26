#!/usr/bin/env python3
"""Package an already-signed volatility release for immutable HTTPS hosting."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "backend"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from release.bundle import verify_release  # noqa: E402


def package_release(release_dir: Path, public_key: Path, output: Path) -> dict[str, object]:
    """Verify and deterministically ZIP exactly one signed release bundle."""
    release = release_dir.resolve()
    manifest = verify_release(release, public_key_path=public_key.resolve())
    expected = {"manifest.json", "manifest.sig", *manifest["files"].keys()}
    actual = {path.relative_to(release).as_posix() for path in release.rglob("*") if path.is_file()}
    unexpected = sorted(actual - expected)
    missing = sorted(expected - actual)
    if unexpected or missing:
        raise ValueError(
            f"release directory file set is not canonical; unexpected={unexpected}, missing={missing}"
        )
    if output.exists():
        raise FileExistsError("output archive already exists; release packages are immutable")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(expected):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, (release / Path(*name.split("/"))).read_bytes())
    payload = output.read_bytes()
    return {
        "status": "packaged",
        "model_id": manifest.get("metadata", {}).get("model_id"),
        "archive": str(output.resolve()),
        "archive_sha256": hashlib.sha256(payload).hexdigest(),
        "archive_bytes": len(payload),
        "files": sorted(expected),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify and package a signed volatility release for immutable HTTPS hosting",
    )
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--public-key-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = package_release(args.release_dir, args.public_key_path, args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
