#!/usr/bin/env python3
"""Convert one locked-certification volatility candidate into a signed ONNX release."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "research", ROOT / "backend"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from volatility_forecasting.export import assemble_release_bundle  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export, parity-check, and sign the certified volatility ensemble",
    )
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-key-path", type=Path, required=True)
    parser.add_argument("--public-key-path", type=Path)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--parity-rows", type=int, default=7)
    args = parser.parse_args()
    if args.opset < 17:
        parser.error("--opset must be 17 or newer")
    if args.parity_rows < 2:
        parser.error("--parity-rows must be at least two")
    summary = assemble_release_bundle(
        args.candidate_dir,
        args.output_dir,
        private_key_path=args.private_key_path,
        public_key_path=args.public_key_path,
        opset_version=args.opset,
        parity_rows=args.parity_rows,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
