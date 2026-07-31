"""Import a licensed news archive into a validated snapshot directory.

The command writes only derived, timestamp-safe Parquet data and a manifest.
Do not commit either the source archive or output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from news_features import archive_manifest, load_news_archive


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate and snapshot a licensed news archive")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("News snapshot output must be absent or empty.")
    articles = load_news_archive(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    archive_path = args.output / "articles.parquet"
    pd.DataFrame(articles).to_parquet(archive_path, index=False)
    manifest = archive_manifest(articles, args.input)
    manifest["archive_path"] = archive_path.name
    manifest["archive_sha256"] = _sha256(archive_path)
    import os
    import uuid

    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    temp_path = args.output / f".manifest-{uuid.uuid4().hex}.json"
    temp_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temp_path, args.output / "manifest.json")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
