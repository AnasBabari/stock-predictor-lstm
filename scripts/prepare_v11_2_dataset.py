"""Create a V11.2 development directory and encrypted final holdout.

Input is an offline, already-audited NPZ panel with arrays named ``dates``,
``security_ids``, ``features``, ``returns``, and ``rv``.  The command is the
only normal workflow that sees the complete panel; subsequent development
commands load only the development directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from research.volatility_forecasting.v11_2_protocol import V112Protocol, protocol_manifest
from research.volatility_forecasting.v11_2_sealed_store import seal_v112_dataset
from research.volatility_forecasting.v11_2_split import create_v112_split, save_split_manifest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--universe-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--key-path", type=Path, required=True)
    parser.add_argument("--schema-sha256", required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    protocol = V112Protocol()
    universe_payload = json.loads(args.universe_manifest.read_text(encoding="utf-8"))
    if int(universe_payload.get("universe_size", 0)) != protocol.universe_size:
        raise SystemExit("universe manifest must contain exactly 64 accepted securities")
    universe_sha = str(universe_payload.get("manifest_sha256", ""))
    if len(universe_sha) != 64:
        raise SystemExit("universe manifest must contain a SHA-256 manifest digest")

    with np.load(args.panel, allow_pickle=False) as panel:
        required = {"dates", "security_ids", "features", "returns", "rv"}
        missing = required.difference(panel.files)
        if missing:
            raise SystemExit(f"panel is missing arrays: {sorted(missing)}")
        dates = [str(value) for value in panel["dates"].tolist()]
        security_ids = [str(value) for value in panel["security_ids"].tolist()]
        features = np.asarray(panel["features"], dtype=np.float32)
        returns = np.asarray(panel["returns"], dtype=np.float32)
        rv = np.asarray(panel["rv"], dtype=np.float32)

    if len(set(security_ids)) != protocol.universe_size:
        raise SystemExit("panel must contain all 64 accepted securities")
    split = create_v112_split(dates, security_ids)
    panel_sha = _sha256_file(args.panel)
    metadata = seal_v112_dataset(
        dates=dates,
        features=features,
        returns=returns,
        rv=rv,
        split=split,
        output_dir=args.output_dir,
        panel_sha256=panel_sha,
        schema_sha256=args.schema_sha256,
        key_path=args.key_path,
        repository_root=args.repository_root,
    )
    (args.output_dir / "manifests" / "protocol.json").write_text(
        json.dumps(protocol_manifest(protocol), indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.output_dir / "manifests" / "universe.json").write_text(
        json.dumps(universe_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    save_split_manifest(split, args.output_dir / "manifests" / "split.json")
    print(
        json.dumps(
            {
                "protocol_sha256": protocol.digest(),
                "universe_sha256": universe_sha,
                "panel_sha256": panel_sha,
                "split_sha256": split.split_sha256,
                "sealed_ciphertext_sha256": metadata.ciphertext_sha256,
                "test_stock_origin_observations": metadata.test_stock_origin_observations,
                "test_unique_sessions": metadata.test_unique_sessions,
                "sealed_test_status": "LOCKED_UNOPENED",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
