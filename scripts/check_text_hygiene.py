"""Reject UTF-8 BOMs in tracked text files."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

TEXT_SUFFIXES = {
    ".css",
    ".html",
    ".js",
    ".jsx",
    ".json",
    ".md",
    ".mjs",
    ".py",
    ".toml",
    ".tsx",
    ".ts",
    ".txt",
    ".yaml",
    ".yml",
}
BOM = b"\xef\xbb\xbf"
ROOT = Path(__file__).resolve().parents[1]


def tracked_text_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    return [
        ROOT / path.decode("utf-8")
        for path in result.stdout.split(b"\0")
        if path
        and (ROOT / path.decode("utf-8")).suffix.lower() in TEXT_SUFFIXES
        and (ROOT / path.decode("utf-8")).exists()
    ]


def main() -> None:
    affected = [
        path.relative_to(ROOT) for path in tracked_text_files() if path.read_bytes().startswith(BOM)
    ]
    if affected:
        print("UTF-8 BOM found in:\n" + "\n".join(map(str, affected)), file=sys.stderr)
        raise SystemExit(1)
    print("Tracked text files contain no UTF-8 BOM.")


if __name__ == "__main__":
    main()
