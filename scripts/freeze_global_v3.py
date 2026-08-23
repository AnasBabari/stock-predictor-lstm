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

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from panel.cross_sectional import (  # noqa: E402
    V3_FEATURE_CONTRACT_VERSION,
    V3_TARGET_CONTRACT_VERSION,
    compute_cross_sectional_ranks,
    compute_relative_forward_returns,
)
from panel.features import build_features_v5  # noqa: E402
from panel.snapshots import load_panel_from_directory  # noqa: E402
from panel.v3_candidates import V3_CANDIDATE_REGISTRY, save_candidate_artifact  # noqa: E402
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
    panel_dir: Path | None = None,
    universe_data: dict[str, Any] | None = None,
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

    train_tickers = list(folds_data.get("train_tickers", []))
    transfer_tickers = list(folds_data.get("asset_transfer_holdout_tickers", []))

    # Load panel data to fit models
    if universe_data is None:
        p_dir = panel_dir or dev_cfg.get("panel_dir")
        if p_dir:
            p_path = Path(p_dir)
            if p_path.exists() and p_path.is_dir():
                universe_data = load_panel_from_directory(p_path)

    # Physical data slicing <= development_cutoff
    cutoff_ts = pd.Timestamp(dev_cutoff)
    dev_data_sliced: dict[str, Any] = {}
    if universe_data:
        for t, df in universe_data.items():
            mask = df.index <= cutoff_ts
            if np.any(mask):
                dev_data_sliced[t] = df.loc[mask].copy()

    # Pre-extract V5 and ranked features for training reference assets
    dev_ranked: dict[str, Any] = {}
    fit_data_min_date = "unknown"
    fit_data_max_date = "unknown"
    if dev_data_sliced and train_tickers:
        v5_features = {
            t: build_features_v5(dev_data_sliced[t])
            for t in train_tickers
            if t in dev_data_sliced
        }
        dev_ranked = compute_cross_sectional_ranks(v5_features, dev_tickers=train_tickers)
        all_dates = [d for df in dev_data_sliced.values() for d in df.index]
        if all_dates:
            fit_data_min_date = pd.Timestamp(min(all_dates)).strftime("%Y-%m-%d")
            fit_data_max_date = pd.Timestamp(max(all_dates)).strftime("%Y-%m-%d")
            if pd.Timestamp(fit_data_max_date) > cutoff_ts:
                raise ValueError(
                    f"Fit data max date {fit_data_max_date} exceeds development cutoff {dev_cutoff}"
                )

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
        "train_tickers": train_tickers,
        "asset_transfer_holdout_tickers": transfer_tickers,
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

    train_tickers_hash = hashlib.sha256(",".join(sorted(train_tickers)).encode("utf-8")).hexdigest()

    for _h_str, dec_raw in selection_data.items():
        dec = V3SelectionDecision.from_dict(dec_raw)
        h_entry: dict[str, Any] = {
            "status": dec.status,
            "candidate": dec.candidate,
            "mean_spearman_ic": dec.mean_spearman_ic,
            "mean_ic_ci_lower_95": dec.mean_ic_ci_lower_95,
            "holm_adjusted_p": dec.holm_adjusted_p,
            "candidate_hyperparameters": dec.candidate_hyperparameters,
            "model_artifact": None,
        }

        # Fit and freeze model artifact if selected
        if dec.status == "selected" and dec.candidate and dec.candidate in V3_CANDIDATE_REGISTRY and dev_ranked:
            cand_cls = V3_CANDIDATE_REGISTRY[dec.candidate]
            hp = dec.candidate_hyperparameters or {}
            try:
                cand_obj = cand_cls(**{k: v for k, v in hp.items() if k != "name"})
            except Exception:
                cand_obj = cand_cls()

            _, dev_rel_targets = compute_relative_forward_returns(
                dev_data_sliced,
                horizon=dec.horizon,
                dev_tickers=train_tickers,
            )

            # Fit ONCE on development reference data <= development_cutoff
            cand_obj.fit(dev_ranked, dev_rel_targets)

            horizon_model_dir = run_dir / "frozen_models" / f"h{dec.horizon}"
            manifest = save_candidate_artifact(
                cand_obj,
                horizon_model_dir,
                horizon=dec.horizon,
                development_cutoff=dev_cutoff,
                feature_contract_version=V3_FEATURE_CONTRACT_VERSION,
                target_contract_version=V3_TARGET_CONTRACT_VERSION,
                train_ticker_digest=train_tickers_hash,
                fit_data_min_date=fit_data_min_date,
                fit_data_max_date=fit_data_max_date,
                protocol_version=dev_cfg.get("research_protocol_version", "global-research-v3"),
            )

            h_entry["model_artifact"] = {
                "directory": f"frozen_models/h{dec.horizon}",
                "manifest_file": f"frozen_models/h{dec.horizon}/model_manifest.json",
                "artifact_digest": manifest["artifact_digest"],
                "files": manifest["files"],
                "fit_data_min_date": fit_data_min_date,
                "fit_data_max_date": fit_data_max_date,
            }

        frozen_dict["selected_candidates"][str(dec.horizon)] = h_entry

    output_frozen_config.parent.mkdir(parents=True, exist_ok=True)
    output_frozen_config.write_text(json.dumps(frozen_dict, indent=2), encoding="utf-8")
    return frozen_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze Protocol V3 development run into immutable config.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Path to completed V3 development run directory")
    parser.add_argument("--output-config", type=Path, default=ROOT / "configs" / "global-v3-frozen.json", help="Path to write frozen config")
    parser.add_argument("--dev-config", type=Path, default=ROOT / "configs" / "global-v3-development.json", help="Original development config")
    parser.add_argument("--panel-dir", type=Path, default=None, help="Optional path to panel data directory")

    args = parser.parse_args()
    freeze_global_v3(
        args.run_dir,
        args.output_config,
        development_config_path=args.dev_config,
        panel_dir=args.panel_dir,
    )
    print(f"Successfully froze Protocol V3 run to {args.output_config}")
    print(f"Frozen Config SHA256: {compute_sha256(args.output_config)}")


if __name__ == "__main__":
    main()
