"""Create a signed production bundle from a passed V11.2 certification.

This command never opens a holdout and never trains.  It only accepts the
immutable V11.2 results directory plus the one-shot certification report, and
it refuses unsupported route families or any evidence mismatch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "research"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from research.volatility_forecasting.export_v11_2 import assemble_v11_2_release  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--certification-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-key-path", type=Path, required=True)
    parser.add_argument("--public-key-path", type=Path)
    parser.add_argument("--opset-version", type=int, default=18)
    parser.add_argument("--parity-rows", type=int, default=7)
    args = parser.parse_args()
    summary = assemble_v11_2_release(
        results_dir=args.results_dir,
        certification_dir=args.certification_dir,
        output_dir=args.output_dir,
        private_key_path=args.private_key_path,
        public_key_path=args.public_key_path,
        opset_version=args.opset_version,
        parity_rows=args.parity_rows,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
