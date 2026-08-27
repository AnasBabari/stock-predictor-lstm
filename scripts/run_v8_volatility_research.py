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
from research.volatility_forecasting.baselines_v8 import (  # noqa: E402
    evaluate_development_baselines,
)
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

    # Development baselines use train for fitting and validation for evaluation.
    # No sealed-test target or baseline is read here.
    baselines = evaluate_development_baselines(
        examples,
        fit_indices=split.train_indices,
        evaluation_indices=split.validation_indices,
    )
    adaptive_qlike = float(baselines["adaptive_calibrated_har_c2c_v1"]["qlike"])
    ridge_qlike = float(baselines["ridge_log_variance"]["qlike"])
    print(f"development validation QLIKE adaptive={adaptive_qlike:.6f} ridge={ridge_qlike:.6f}")

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

    # This command no longer fabricates a certifiable candidate. Until the real
    # RTX TCN/fusion runner writes verified .pt members, its output is explicitly
    # development-only baseline evidence.
    candidate_manifest = {
        "artifact_role": "v8_development_baseline_evidence",
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
        "model_type": "ridge_log_variance_baseline",
        "validation_metrics": {
            "ridge_qlike": ridge_qlike,
            "adaptive_baseline_qlike": adaptive_qlike,
            "eligible": False,
            "status": "baseline_only_not_candidate",
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
        "note": "Development-only fitted baseline evidence. This is not a model candidate and cannot be certified or released. Test targets remain sealed.",
    }
    (out / "candidate-manifest.json").write_text(
        json.dumps(candidate_manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"development baseline evidence written to {out}; no candidate was created")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
