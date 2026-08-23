"""Freeze Protocol V3 development decisions and produce immutable frozen configuration & model artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from panel.cross_sectional import (  # noqa: E402
    V3_FEATURE_CONTRACT_VERSION,
    V3_TARGET_CONTRACT_VERSION,
)
from panel.v3_certification import V3CertificationGateConfig  # noqa: E402
from panel.v3_selection import V3SelectionDecision  # noqa: E402


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit_sha() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def freeze_global_v3(
    run_dir: Path,
    output_frozen_config: Path,
    *,
    development_config_path: Path | None = None,
) -> dict[str, Any]:
    stages_dir = run_dir / "stages"
    selection_file = stages_dir / "06_selection.json"
    snapshot_file = stages_dir / "01_snapshot.json"
    if not snapshot_file.exists():
        snapshot_file = stages_dir / "01_snapshot_manifest.json"
    folds_file = stages_dir / "03_folds.json"

    if not selection_file.exists():
        raise FileNotFoundError(f"Selection file not found: {selection_file}")
    if not snapshot_file.exists():
        raise FileNotFoundError(f"Snapshot manifest not found: {snapshot_file}")

    selection_data = json.loads(selection_file.read_text(encoding="utf-8"))
    snapshot_manifest = json.loads(snapshot_file.read_text(encoding="utf-8"))
    folds_data = json.loads(folds_file.read_text(encoding="utf-8")) if folds_file.exists() else {}

    dev_cfg: dict[str, Any] = {}
    dev_config_sha256 = "none"
    if development_config_path and development_config_path.exists():
        dev_cfg = json.loads(development_config_path.read_text(encoding="utf-8"))
        dev_config_sha256 = compute_sha256(development_config_path)

    dev_cutoff = dev_cfg.get("development_cutoff", "2026-08-21")
    gate_cfg = V3CertificationGateConfig.from_dict(dev_cfg.get("certification_gate", {}))

    frozen_dict: dict[str, Any] = {
        "freeze_status": "frozen",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "git_commit_sha": get_git_commit_sha(),
        "development_config_sha256": dev_config_sha256,
        "run_id": f"{dev_cfg.get('run_id', 'global-v3')}-frozen",
        "research_protocol_version": "global-research-v3",
        "protocol_version": "global-cert-v3",
        "development_cutoff": dev_cutoff,
        "development_snapshot_digest": compute_sha256(snapshot_file),
        "development_selection_digest": compute_sha256(selection_file),
        "development_folds_digest": compute_sha256(folds_file) if folds_file.exists() else None,
        "universe_size": len(snapshot_manifest.get("tickers", [])),
        "feature_contract_version": V3_FEATURE_CONTRACT_VERSION,
        "target_contract_version": V3_TARGET_CONTRACT_VERSION,
        "horizons": dev_cfg.get("horizons", [1, 3, 5, 7, 14, 30]),
        "folds": dev_cfg.get("folds", 5),
        "embargo": dev_cfg.get("embargo", 5),
        "holdout_fraction": dev_cfg.get("holdout_fraction", 0.20),
        "asset_split_seed": dev_cfg.get("asset_split_seed", 42),
        "train_tickers": folds_data.get("train_tickers", []),
        "asset_transfer_holdout_tickers": folds_data.get("asset_transfer_holdout_tickers", []),
        "selected_candidates": {},
        "inference": dev_cfg.get(
            "inference",
            {
                "hac_lag_policy": "horizon_minus_one",
                "bootstrap_type": "moving_block",
                "bootstrap_block_policy": "max_5_or_horizon",
                "bootstrap_resamples": 2000,
                "bootstrap_seed": 42,
                "multiple_testing": "holm",
                "family_alpha": 0.05,
            },
        ),
        "certification_gate": gate_cfg.to_dict(),
    }

    for _h_str, dec_raw in selection_data.items():
        dec = V3SelectionDecision.from_dict(dec_raw)
        frozen_dict["selected_candidates"][str(dec.horizon)] = {
            "status": dec.status,
            "candidate": dec.candidate,
            "mean_spearman_ic": dec.mean_spearman_ic,
            "mean_ic_ci_lower_95": dec.mean_ic_ci_lower_95,
            "holm_adjusted_p": dec.holm_adjusted_p,
            "candidate_hyperparameters": dec.candidate_hyperparameters,
        }

    output_frozen_config.parent.mkdir(parents=True, exist_ok=True)
    output_frozen_config.write_text(json.dumps(frozen_dict, indent=2), encoding="utf-8")
    return frozen_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze Protocol V3 development run into immutable config.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Path to completed V3 development run directory")
    parser.add_argument("--output-config", type=Path, default=ROOT / "configs" / "global-v3-frozen.json", help="Path to write frozen config")
    parser.add_argument("--dev-config", type=Path, default=ROOT / "configs" / "global-v3-development.json", help="Original development config")

    args = parser.parse_args()
    freeze_global_v3(args.run_dir, args.output_config, development_config_path=args.dev_config)
    print(f"Successfully froze Protocol V3 run to {args.output_config}")
    print(f"Frozen Config SHA256: {compute_sha256(args.output_config)}")


if __name__ == "__main__":
    main()
