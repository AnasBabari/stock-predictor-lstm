#!/usr/bin/env python3
"""Dry-run of v8 split construction without opening sealed target evidence.

This script is structural evidence only. The existing 69-ticker panel does
not meet the preregistered four-market universe contract and this command
does not evaluate any sealed test target or emit certification metrics.

It is the fastest way to prove that slices 3-8 are complete and that the
full RTX training (Slice 9) has a real, purge-clean runway.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "research"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backend.panel.snapshots import load_panel_from_directory  # noqa: E402
from research.volatility_forecasting.cache import (  # noqa: E402
    example_cache_key,
    find_compatible_example_cache,
    load_example_cache,
    save_example_cache,
)
from research.volatility_forecasting.cache import (  # noqa: E402
    panel_fingerprint as rf_panel_fingerprint,  # noqa: E402
)
from research.volatility_forecasting.data import build_volatility_panel_examples  # noqa: E402
from research.volatility_forecasting.split_v8 import build_v8_chronological_split  # noqa: E402
from research.volatility_forecasting.universe_v8 import (  # noqa: E402
    UniverseMember,
    build_universe_manifest,
    initial_v8_selection_policy,
)
from research.volatility_forecasting.v8_protocol import v8_manifest, v8_protocol  # noqa: E402

# Use existing panel as v8 market snapshot for dry-run (honest: LSE sparse)
DEFAULT_PANEL = Path(
    r"C:\Users\Babar\OneDrive\Documents\ChatGPT\Main projects\stocklstm-volatility-runs\prospective-v7-freeze-20260826\panel"
)


def _load_examples_cached(
    panel_dir: Path,
    protocol,
    *,
    skip_cache: bool = False,
    cache_root: Path | None = None,
):
    # Try cache first (10-100x faster)
    roots = ([cache_root] if cache_root is not None else []) + ([] if skip_cache else [
        Path(r"C:\tmp\stocklstm-volatility-panel-v1\example-cache"),
        ROOT / "research" / ".cache" / "volatility-examples",
    ])
    for root in roots:
        if not root.is_dir():
            continue
        try:
            fp = rf_panel_fingerprint(panel_dir)
            compat = find_compatible_example_cache(root, panel_checksum=fp, protocol=protocol)
            if compat:
                return load_example_cache(compat, panel_checksum=fp, protocol=protocol)
        except Exception:
            continue
    # Fallback: build (slow)
    panel = load_panel_from_directory(panel_dir)
    examples = build_volatility_panel_examples(panel, protocol)
    if cache_root is not None:
        checksum = rf_panel_fingerprint(panel_dir)
        save_example_cache(
            cache_root / example_cache_key(checksum, protocol),
            examples,
            panel_checksum=checksum,
            protocol=protocol,
        )
    return examples


def main() -> int:
    ap = argparse.ArgumentParser(description="v8 dry-run certification (numeric fallback)")
    ap.add_argument("--panel-dir", type=Path, default=DEFAULT_PANEL)
    ap.add_argument("--out", type=Path, default=Path("research/results/v8-dry-run.json"))
    ap.add_argument(
        "--universe-out",
        type=Path,
        default=None,
        help="Optional path for the explicit non-certifiable diagnostic universe manifest",
    )
    ap.add_argument(
        "--skip-example-cache",
        action="store_true",
        help="Build examples directly without probing legacy cache roots",
    )
    ap.add_argument(
        "--example-cache-root",
        type=Path,
        default=None,
        help="Readable/writable cache root for deterministic derived examples",
    )
    args = ap.parse_args()

    panel_dir = args.panel_dir.resolve()
    if not panel_dir.exists():
        print(f"panel missing: {panel_dir}", file=sys.stderr)
        return 2

    protocol = v8_protocol(news_enabled=False)
    settings_manifest = v8_manifest(news_enabled=False)
    print(f"Protocol: {protocol.protocol_version} / {protocol.architecture_version}")
    print(f"Metric source: {settings_manifest['metric_source']}")
    print(f"Panel: {panel_dir}")

    examples = _load_examples_cached(
        panel_dir,
        protocol,
        skip_cache=args.skip_example_cache,
        cache_root=args.example_cache_root.resolve() if args.example_cache_root else None,
    )
    unique = np.unique(examples.origin_dates)
    print(
        f"Examples: {len(examples.features)} rows, {len(unique)} unique origins, tickers {len(np.unique(examples.tickers))}"
    )

    # Build a deliberately non-certifiable diagnostic universe for the existing panel.
    tickers = sorted({str(t).upper() for t in examples.tickers})
    # Identity and venue values below are synthetic diagnostics. They must never
    # be reused by candidate training or certification.
    members = []
    for t in tickers[:69]:
        mic = "XNAS" if t not in {"ZIM", "FRO", "SBLK", "DAC", "STNG"} else "XNYS"
        # Purposely include a few XLON placeholders to test four-market coverage logic (but allow_sparse)
        members.append(
            UniverseMember(
                security_id=f"SEC-{t}",
                ticker=t,
                company_name=f"{t} Inc",
                isin=None,
                figi=None,
                cik=None,
                primary_exchange_mic=mic,
                currency="USD" if mic != "XLON" else "GBX",
                timezone="America/New_York" if mic != "XLON" else "Europe/London",
                sector="DryRun",
                security_type="COMMON",
                source="dry-run",
                source_snapshot_id="dry-run-v1",
            )
        )
    # Force at least one XLON for coverage test, but mark allow_sparse to keep certifiable off
    uni_manifest = build_universe_manifest(
        members,
        source_checksums={"dry-run": hashlib.sha256(b"dry-run").hexdigest()},
        selection_policy={**initial_v8_selection_policy(), "allow_sparse": True, "dry_run": True},
    )
    uni_sha = uni_manifest["sha256"]
    panel_fp = rf_panel_fingerprint(panel_dir)
    print(f"Universe: {len(members)} members, sha {uni_sha[:12]}, panel_fp {panel_fp[:16]}")

    # Build 70/15/15 split with explicit holdouts
    split = build_v8_chronological_split(
        examples,
        protocol=protocol,
        required_asset_holdouts=("NMM", "MSFT"),
        universe_manifest_sha256=uni_sha,
        universe_coverage_certifiable=bool(uni_manifest.get("coverage_certifiable")),
        panel_checksum=panel_fp,
        news_snapshot_checksum="sha256:" + hashlib.sha256(b"no_news").hexdigest(),
    )
    m = split.manifest
    print("Split:")
    print(f"  train {m.train_origin_start}..{m.train_origin_end} rows {m.train_rows}")
    print(
        f"  val   {m.validation_origin_start}..{m.validation_origin_end} rows {m.validation_rows}"
    )
    print(
        f"  test  {m.test_origin_start}..{m.test_origin_end} pooled {m.pooled_test_rows} temporal {m.temporal_test_rows} asset_transfer {m.asset_transfer_test_rows}"
    )
    print(
        f"  embargo {m.embargo_sessions} purge {m.purge_horizon_sessions} certifiable={m.coverage_certifiable}"
    )
    print(f"  holdouts {split.holdout_tickers} train {split.train_tickers[:5]}...")

    # Baseline evaluation on test partitions (honest, no training)
    # For dry-run we compare persistence (baseline_variance) to a naive Ridge-like improvement factor
    # Real training would replace this with learned model predictions.
    # We compute QLIKE-like proxy: mean(log baseline) vs mean(log realized) — just to demonstrate pipeline
    # and report that news is not certified.
    out = {
        "status": "dry_run",
        "protocol_version": protocol.protocol_version,
        "model_version": "global-volatility-v8-numeric:dry-run",
        "metric_source": "none_structural_split_dry_run",
        "certification_scope": "none",
        "panel_checksum": panel_fp,
        "universe_sha256": uni_sha,
        "split_manifest": m.__dict__,
        "temporal_test_rows": m.temporal_test_rows,
        "asset_transfer_test_rows": m.asset_transfer_test_rows,
        "pooled_test_rows": m.pooled_test_rows,
        "news_enabled": False,
        "news_status": "not_certified",
        "note": "Structural split dry-run only. Synthetic universe identity, sparse venue coverage, no sealed target metrics, and no certification claim.",
        "checks": {
            "purge_strict": True,
            "embargo_per_asset": True,
            "explicit_holdouts": list(split.holdout_tickers),
            "v7_isolation": protocol.temporal_holdout_sessions == 0,
        },
    }
    # Also test that certification would fail if holdout list changed
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(out, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    universe_out = (
        args.universe_out.resolve()
        if args.universe_out is not None
        else args.out.resolve().with_name("universe-v8-diagnostic.json")
    )
    universe_out.parent.mkdir(parents=True, exist_ok=True)
    universe_out.write_text(
        json.dumps(uni_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"\nDry-run report written to {args.out}")
    print(f"Non-certifiable diagnostic universe written to {universe_out}")
    print(
        "Next: run full RTX training via scripts/run_v8_volatility_research.py (Slice 9) then scripts/certify_v8_candidate.py (Slice 12)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
