#!/usr/bin/env python3
"""v8 research runner — chronological 70/15/15, purge/embargo-clean, asset-transfer.

This is the v8 analogue of ``scripts/run_prospective_volatility_research.py``
but for the sealed historical split.  It trains only on train, validates
only on validation, and never opens the test set.  The test set is sealed
until ``scripts/certify_v8_candidate.py``.

For the initial numeric certification (news_status=not_certified) this
runner trains a lightweight Ridge baseline per horizon (CPU) to demonstrate
the pipeline is end-to-end.  The full RTX run will replace Ridge with the
fusion TCN (global-volatility-news-fusion-v8) on the same split — the split
manifest, universe, and panel checksums remain identical.

Usage:
  python scripts/run_v8_volatility_research.py \
    --panel-dir /path/to/v8-market-snapshot \
    --universe-manifest /path/to/universe-v8-manifest.json \
    --out /tmp/v8-candidate \
    --news-enabled false
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "research"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backend.panel.snapshots import load_panel_from_directory  # noqa: E402
from research.volatility_forecasting.baselines_v8 import evaluate_all_baselines  # noqa: E402
from research.volatility_forecasting.cache import (  # noqa: E402
    find_compatible_example_cache,
    load_example_cache,
    panel_fingerprint,
)
from research.volatility_forecasting.data import build_volatility_panel_examples  # noqa: E402
from research.volatility_forecasting.split_v8 import build_v8_chronological_split  # noqa: E402
from research.volatility_forecasting.v8_protocol import v8_manifest, v8_protocol  # noqa: E402


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="v8 research runner (train/val only, test sealed)")
    ap.add_argument("--panel-dir", type=Path, required=True)
    ap.add_argument("--universe-manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="Empty output dir for candidate")
    ap.add_argument("--news-enabled", type=lambda x: str(x).lower() == "true", default=False)
    ap.add_argument(
        "--holdouts",
        type=str,
        default="NMM,MSFT",
        help="Comma-separated holdouts, must match universe policy",
    )
    return ap.parse_args()


def _load_examples(panel_dir: Path, protocol):
    for root in [
        Path(r"C:\tmp\stocklstm-volatility-panel-v1\example-cache"),
        ROOT / "research" / ".cache" / "volatility-examples",
    ]:
        if not root.is_dir():
            continue
        try:
            fp = panel_fingerprint(panel_dir)
            compat = find_compatible_example_cache(root, panel_checksum=fp, protocol=protocol)
            if compat:
                return load_example_cache(compat, panel_checksum=fp, protocol=protocol), fp
        except Exception:
            continue
    panel = load_panel_from_directory(panel_dir)
    fp = panel_fingerprint(panel_dir) if (panel_dir / "manifest.json").exists() else "no-checksum"
    return build_volatility_panel_examples(panel, protocol), fp


def main() -> int:
    args = _parse_args()
    panel_dir = args.panel_dir.resolve()
    uni_path = args.universe_manifest.resolve()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        print(f"--out must be empty or non-existent: {out}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)

    holdouts = tuple(sorted({t.strip().upper() for t in args.holdouts.split(",") if t.strip()}))
    if not holdouts:
        print("holdouts required", file=sys.stderr)
        return 2

    protocol = v8_protocol(news_enabled=args.news_enabled)
    manifest = v8_manifest(news_enabled=args.news_enabled)
    print(f"v8 protocol {protocol.protocol_version} news_enabled={args.news_enabled}")
    print(f"holdouts {holdouts}")

    uni = json.loads(uni_path.read_text(encoding="utf-8"))
    uni_sha = uni.get("sha256")
    if not uni_sha:
        print("universe manifest missing sha256", file=sys.stderr)
        return 2

    examples, panel_fp = _load_examples(panel_dir, protocol)
    print(
        f"examples {len(examples.features)} rows, {len(np.unique(examples.origin_dates))} origins"
    )

    # Build chronological split — explicit holdouts, no test access beyond manifest
    split = build_v8_chronological_split(
        examples,
        protocol=protocol,
        required_asset_holdouts=holdouts,
        universe_manifest_sha256=uni_sha,
        panel_checksum=panel_fp,
        news_snapshot_checksum="sha256:"
        + hashlib.sha256(b"no_news" if not args.news_enabled else b"news").hexdigest(),
    )
    print(
        f"split train {split.manifest.train_rows} val {split.manifest.validation_rows} pooled_test {split.manifest.pooled_test_rows}"
    )
    # Save split manifest for audit (immutable)
    (out / "split-v8-manifest.json").write_text(
        json.dumps(split.manifest.__dict__, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    (out / "universe-v8-manifest.json").write_text(
        json.dumps(uni, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Baselines on validation (selection uses validation only, never test)
    # For dry-run we fit a Ridge stub on train and evaluate on val
    # Real RTX training would fit TCN with early stopping on val
    baselines = evaluate_all_baselines(examples, split)
    print(
        f"baselines val HAR qlike {baselines['har']['qlike']:.4f} (temporal {baselines['har_temporal']['qlike']:.4f} asset_transfer {baselines['har_asset_transfer']['qlike']:.4f})"
    )

    # Train lightweight Ridge per horizon on train (CPU) — placeholder for TCN on RTX
    try:
        from sklearn.linear_model import Ridge
    except ImportError:
        print(
            "sklearn not available — skipping Ridge training, writing stub candidate",
            file=sys.stderr,
        )
        ridge_qlike = baselines["ridge_stub"]["qlike"]
        model_type = "ridge_stub"
    else:
        # Fit Ridge on flattened window features (mean across window for speed)
        X_train = examples.features[split.train_indices].mean(axis=1)  # [n, 26]
        y_train = examples.realized_variance[split.train_indices]  # [n, 6]
        X_val = examples.features[split.validation_indices].mean(axis=1)
        y_val = examples.realized_variance[split.validation_indices]
        # Simple per-horizon Ridge
        preds = []
        for h in range(len(protocol.horizons)):
            rid = Ridge(alpha=1.0)
            rid.fit(X_train, y_train[:, h])
            preds.append(rid.predict(X_val))
        y_pred_val = np.column_stack(preds)
        # QLIKE on val
        eps = 1e-8
        ratio = np.clip(y_pred_val / np.clip(y_val, eps, None), eps, 1e6)
        qlike = float(np.mean(ratio - np.log(ratio) - 1))
        print(f"Ridge val qlike {qlike:.4f} vs HAR {baselines['har']['qlike']:.4f}")
        ridge_qlike = qlike
        model_type = "ridge_cpu"

    # Decide selection: must beat HAR on validation to be eligible (simplified gate)
    # Real gate uses QLIKE 0.98 etc. per horizon per seed
    har_qlike = baselines["har"]["qlike"]
    eligible = ridge_qlike < har_qlike * 0.99  # 1% improvement over HAR
    status = "selected" if eligible else "abstain"
    print(f"selection {status}: Ridge {ridge_qlike:.4f} vs HAR {har_qlike:.4f}")

    # GPU / runtime metadata (honest: this dry-run is CPU; RTX run will record GPU)
    gpu_info = "cpu-dry-run"
    try:
        import torch

        if torch.cuda.is_available():
            gpu_info = torch.cuda.get_device_name(0)
        else:
            gpu_info = f"cpu torch {torch.__version__}"
    except Exception:
        pass

    # Write candidate manifest (prospective_v8_development_candidate, not yet certified)
    # For full TCN, this would contain 3 seed members with weights; here we store Ridge coefs as stub
    candidate_manifest = {
        "artifact_role": "prospective_v8_development_candidate",
        "protocol_version": protocol.protocol_version,
        "model_version": manifest["model_version"],
        "architecture_version": manifest["architecture_version"],
        "target_version": protocol.target_version,
        "feature_schema_version": manifest["feature_schema_version"],
        "horizons": list(protocol.horizons),
        "required_horizons": list(manifest["required_horizons"]),
        "window_size": protocol.window_size,
        "seeds": list(protocol.seeds),
        "news_enabled": args.news_enabled,
        "news_status": "not_certified" if not args.news_enabled else "development",
        "release_eligible": False,
        "strict_release_policy": {
            "unsigned": True,
            "partial_release_allowed": False,
            "old_locked_holdout_reusable": False,
            "future_certification_required": False,
            "sealed_test_required": True,
        },
        "panel_checksum": panel_fp,
        "universe_manifest_sha256": uni_sha,
        "split_manifest": split.manifest.__dict__,
        "split_manifest_sha256": hashlib.sha256(
            json.dumps(split.manifest.__dict__, sort_keys=True, default=str).encode()
        ).hexdigest(),
        "model_type": model_type,
        "validation_metrics": {
            "ridge_qlike": ridge_qlike,
            "har_qlike": har_qlike,
            "eligible": eligible,
            "status": status,
        },
        "baseline_metrics": baselines,
        "runtime": {
            "gpu": gpu_info,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cuda": "n/a-cpu-dry-run",
            "torch": "n/a" if "torch" not in sys.modules else __import__("torch").__version__,  # type: ignore
            "duration_seconds": 0,
        },
        "created_at": __import__("datetime")
        .datetime.now(__import__("datetime").timezone.utc)
        .isoformat(),
        "note": "Dry-run Ridge CPU placeholder; full RTX training will replace with fusion TCN global-volatility-news-fusion-v8 on same split. Test set remains sealed.",
    }
    # Also write a stub member file for each seed (so certification can verify member existence)
    for seed in protocol.seeds:
        member_path = out / f"seed-{seed}.json"
        member_path.write_text(
            json.dumps({"seed": seed, "model_type": model_type, "placeholder": True}, indent=2)
            + "\n",
            encoding="utf-8",
        )
    (out / "candidate-manifest.json").write_text(
        json.dumps(candidate_manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"candidate written to {out} status {status} (seeds {protocol.seeds})")
    print(
        f"Next: certify with scripts/certify_v8_candidate.py --candidate-dir {out} --panel-dir {panel_dir} --universe-manifest {uni_path} --out /tmp/v8-cert"
    )
    return 0 if status == "selected" else 1


if __name__ == "__main__":
    raise SystemExit(main())
